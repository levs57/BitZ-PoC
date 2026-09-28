//! The paper's raw-performance row for the BitZ parity prover at one size.
//!
//! `bitz_bench <n> [--reps R] [--seed S] [--ladder L] [--compare-cached-rounds]`
//! builds a random instance at the
//! scheme's split (`Shape::reference`: `t = ⌈3n/5⌉ − 1`, `s = n − t`,
//! `q = 2^100 − 15`, generator `X`, the dump examples' transcript labels),
//! commits it `R` times (median), proves it once to warm up and then `R`
//! times (medians of the wall time and of every traced phase, the paper's
//! buckets summed from them), verifies every timed proof (median), and
//! prints one `RESULT` line. One size per process so the peak resident set
//! is the shape's; the thread count is rayon's (`RAYON_NUM_THREADS`).
//!
//! Buckets, as the paper's table splits the F2Z prover: grand products =
//! `fold+images` + `gkr` (the integer folds, the images, the batched GKR);
//! ring switch incl. its sumcheck = `sumcheck` + `ring switch` (the
//! reduction of the GKR exit claim to one packed evaluation and the
//! ring-switch message); Ligerito = `ligerito`; Total = commit + prove.
//! Proof bytes: narg = every transcript message (non-Ligerito), hints =
//! the serialized Ligerito proof.
use std::time::{Duration, Instant};

use bitz::wfbitz::fold::{fold_columns, reconstruct};
use bitz::wfbitz::{
    BitZParams, BitZProver, BitZVerifier, LinearClaim, Pcs, Shape, WINDOW, build_prover,
    build_verifier, record_phases, take_phases,
};
use field::Gf128 as Gf;
use bitz::ligerito_flock::LigeritoSelection;
use bitz::pcs::IntegerMatrixLayout;
use flock_core::merkle::HashKind;
use flock_core::pcs::ligerito::LigeritoProfile;

/// `2^100 − 15`, the dump examples' prime.
const Q: u128 = (1u128 << 100) - 15;
const SESSION: &str = "bitz-tests";
const INSTANCE: &str = "fold-round-trip";

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn median(v: &[Duration]) -> Duration {
    let mut sorted = v.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: `getrusage` fills the struct for the calling process.
    let ok = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if ok != 0 {
        return 0;
    }
    // SAFETY: filled by the call above.
    let usage = unsafe { usage.assume_init() };
    usage.ru_maxrss as u64
}

struct ProverRun {
    label: &'static str,
    prover: BitZProver,
    prove_times: Vec<Duration>,
    verify_times: Vec<Duration>,
    phase_times: Vec<(String, Vec<Duration>)>,
    sizes: (usize, usize),
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut n: Option<usize> = None;
    let mut reps = 5usize;
    let mut seed = 1u64;
    let mut compare_cached_rounds = false;
    // `fast` = flock's embedded ladder as shipped; otherwise one of the
    // crate's validated selections (`custom:1:4` = the paper's rate-1/2
    // Johnson ladder, `custom:3:4` = rate 1/8), resolved at 100 bits. No
    // Round 0 either way (the raw harness has no outer transcript to bind
    // it on; the opener path does run it).
    let mut ladder = String::from("fast");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--reps" => {
                reps = args[i + 1].parse().expect("--reps");
                i += 2;
            }
            "--ladder" => {
                ladder = args[i + 1].clone();
                i += 2;
            }
            "--seed" => {
                seed = args[i + 1].parse().expect("--seed");
                i += 2;
            }
            "--compare-cached-rounds" => {
                compare_cached_rounds = true;
                i += 1;
            }
            x => {
                n = Some(x.parse().expect("n"));
                i += 1;
            }
        }
    }
    let n = n.expect("usage: bitz_bench <n> [--reps R] [--seed S] [--ladder fast|custom:r:k] [--compare-cached-rounds]");
    if compare_cached_rounds && !cfg!(feature = "bench-internals") {
        panic!("--compare-cached-rounds requires --features bench-internals");
    }
    let shape = Shape::reference(n).expect("shape");
    let (t, s) = (shape.log_rows(), shape.log_columns());
    let params = BitZParams::new(shape, Q, Gf::from_polynomial_words([2, 0])).expect("params");
    #[cfg(feature = "parallel")]
    let threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let threads = 1;
    println!("bitz_bench n={n} t={t} s={s} threads={threads} reps={reps} seed={seed} ladder={ladder}");

    // The instance.
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let words = (1usize << t) / 64;
    let rows: Vec<Vec<u64>> = (0..1usize << s)
        .map(|_| (0..words).map(|_| xorshift(&mut state)).collect())
        .collect();
    let mut residue = || {
        let hi = u128::from(xorshift(&mut state));
        let lo = u128::from(xorshift(&mut state));
        ((hi << 64) | lo) % Q
    };
    let row_weights: Vec<u128> = (0..1usize << t).map(|_| residue()).collect();
    let column_weights: Vec<u128> = (0..1usize << s).map(|_| residue()).collect();
    let folds = fold_columns(&shape, &rows, &row_weights);
    let unresolved = LinearClaim::new(&params, row_weights.clone(), column_weights.clone(), 0).expect("claim");
    let target = reconstruct(&unresolved, &folds, Q);
    let claim = LinearClaim::new(&params, row_weights, column_weights, target).expect("claim");
    let pcs = if ladder == "fast" {
        Pcs::new(&shape, HashKind::Blake3).expect("pcs")
    } else {
        let layout = IntegerMatrixLayout { row_vars: t, col_vars: s, word_bits: 1 };
        let resolved = LigeritoSelection::parse(&ladder, 100)
            .and_then(|selection| selection.resolve(bitz::ligerito::packed_vars(&layout), 100))
            .expect("--ladder");
        Pcs::with_security(&shape, resolved.security(), LigeritoProfile::Fast).expect("pcs")
    };

    // Commit: the median of `reps` commits, the last one kept.
    let mut commit_times = Vec::with_capacity(reps);
    let mut committed = None;
    for _ in 0..reps {
        let input = rows.clone();
        let started = Instant::now();
        let (root, hint) = pcs.commit(&shape, input).expect("commit");
        commit_times.push(started.elapsed());
        committed = Some((root, hint));
    }
    drop(rows);
    let (root, hint) = committed.expect("committed");
    let commit = median(&commit_times);
    println!("commit: median {:.2} ms over {reps}", ms(commit));

    // Prove: one warm-up per mode, then interleaved timed and verified runs.
    let mut runs = Vec::new();
    #[cfg(feature = "bench-internals")]
    if compare_cached_rounds {
        runs.push(ProverRun {
            label: "ordinary",
            prover: BitZProver::new(params, WINDOW).with_cached_forest_rounds(false),
            prove_times: Vec::with_capacity(reps),
            verify_times: Vec::with_capacity(reps),
            phase_times: Vec::new(),
            sizes: (0, 0),
        });
    }
    let cached = BitZProver::new(params, WINDOW);
    #[cfg(feature = "bench-internals")]
    let cached = cached.with_cached_forest_rounds(true);
    runs.push(ProverRun {
        label: "cached-k2",
        prover: cached,
        prove_times: Vec::with_capacity(reps),
        verify_times: Vec::with_capacity(reps),
        phase_times: Vec::new(),
        sizes: (0, 0),
    });
    let verifier = BitZVerifier::new(params, WINDOW);
    for rep in 0..=reps {
        for turn in 0..runs.len() {
            let i = if rep & 1 == 0 { turn } else { runs.len() - 1 - turn };
            let run = &mut runs[i];
            record_phases(true);
            let started = Instant::now();
            let mut transcript = build_prover(SESSION, INSTANCE);
            run.prover.prove(&claim, &pcs, &hint, &mut transcript, None).expect("prove");
            let proof = transcript.finish();
            let elapsed = started.elapsed();
            let phases = take_phases();
            record_phases(false);
            if rep == 0 {
                continue;
            }
            run.prove_times.push(elapsed);
            for (label, d) in phases {
                match run.phase_times.iter_mut().find(|(l, _)| *l == label) {
                    Some((_, v)) => v.push(d),
                    None => run.phase_times.push((label, vec![d])),
                }
            }
            let started = Instant::now();
            verifier
                .verify(&claim, &pcs, root, build_verifier(SESSION, INSTANCE, &proof), None)
                .expect("verify");
            run.verify_times.push(started.elapsed());
            run.sizes = (proof.narg_string.len(), proof.hints.len());
        }
    }
    let rss = peak_rss_bytes();
    for run in runs {
        let prove = median(&run.prove_times);
        let verify = median(&run.verify_times);
        let phase = |label: &str| -> Duration {
            run.phase_times
                .iter()
                .find(|(l, _)| l == label)
                .map_or(Duration::ZERO, |(_, v)| median(v))
        };
        let grand = phase("fold+images") + phase("gkr");
        let ring = phase("sumcheck") + phase("ring switch");
        let lig = phase("ligerito");
        println!("\nforest mode: {}", run.label);
        println!("prove: median {:.1} ms over {reps} (min {:.1}); phases:", ms(prove), ms(*run.prove_times.iter().min().expect("reps")));
        for (label, v) in &run.phase_times {
            println!("  {label:<24} {:>8.2} ms", ms(median(v)));
        }
        println!(
            "buckets: grand products {:.1} | ring switch incl. sumcheck {:.1} | ligerito {:.1} | total {:.1} (commit {:.2} + prove {:.1})",
            ms(grand), ms(ring), ms(lig), ms(commit + prove), ms(commit), ms(prove)
        );
        println!("verify: median {:.2} ms; proof: narg {} B + hints {} B = {} B", ms(verify), run.sizes.0, run.sizes.1, run.sizes.0 + run.sizes.1);
        println!("peak rss: {:.2} GB", rss as f64 / 1e9);
        println!(
            "RESULT schema=bitz-bench/1 forest={} n={n} t={t} s={s} threads={threads} reps={reps} seed={seed} ladder={ladder} commit_ms={:.3} prove_ms={:.3} grand_ms={:.3} ring_ms={:.3} lig_ms={:.3} verify_ms={:.3} narg_bytes={} hints_bytes={} peak_rss_bytes={rss}",
            run.label, ms(commit), ms(prove), ms(grand), ms(ring), ms(lig), ms(verify), run.sizes.0, run.sizes.1
        );
    }
}
