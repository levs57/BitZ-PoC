//! BitZ transcript parity — `f2z-benchmark`'s clean-room implementation of
//! F2Z ("BitZ", protocol id `bitz/v1`), driven from this crate's field and
//! flock engine so the two produce the same narg string, hint stream and
//! challenge sequence on the same instance.
//!
//! The message and challenge order, the framing labels and the wire codecs
//! mirror `worldfnd/f2z-benchmark` at `0c75fd8` (branch `bitz-k4` for the
//! k = 4 Ligerito ladder) step for step; nothing here touches the crate's
//! own protocol, which stays byte-identical to what it was. Where the two
//! agree on the values — the GF(2^128) representation, the packed-witness
//! layout, the flock commit and the Ligerito engine — this module reuses
//! the crate's code; where they differ only in how the value is computed,
//! it follows their algorithm first (a per-layer grand-product GKR that
//! binds the in-tree variables MSB-first, a dense degree-2 reduction
//! sumcheck) so the transcript can be pinned before the fast kernels are
//! re-derived for their variable order.
//!
//! Layout, one file per protocol layer of theirs:
//! - [`transcript`]: the spongefish wrapper with the hint channel.
//! - [`params`]: shape and parameter gates, the linear claim, the frames.
//! - [`fold`]: step 3, the integer column folds and the batching point.
//! - [`gkr`]: step 4, the batched grand-product GKR.
//! - [`reduce`]: the reduction of the GKR exit claim to an inner product.
//! - [`sumcheck`]: the dense degree-2 sumcheck to one MLE claim.
//! - [`pcs`]: step 6, statement binding, ring switch and the Ligerito
//!   opening through a challenger that frames flock's events their way.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

pub mod chained;
pub mod codec;
pub mod fold;
pub mod forest;
pub mod gkr;
pub(crate) mod kernels;
pub mod params;
pub mod pcs;
pub mod reduce;
pub mod sumcheck;
pub mod transcript;
pub mod virt;

use crate::ligerito_flock::FlockCommitHint;
use field::Gf128 as Gf;

pub use params::{BitZParams, LinearClaim, ParamsError, Root, Shape, ShapeError, SumClaimGf};
pub use pcs::{OpeningQuery, Pcs, StatementBinding};
pub use transcript::{Proof, ProverState, VerifierState, build_prover, build_verifier};
pub use chained::{ChainedError, ChainedGeometry, ChainedStatement};
pub use virt::{VirtualError, VirtualStatement};

/// Their comb: a fixed-base exponentiation table over the generator
/// (`field::FixedBasePow` with public, variable-time exponents — every
/// exponent it sees is a transcript value or a sent fold).
pub struct FixedBasePow(field::FixedBasePow<field::Gf128Ops, 2>);

impl FixedBasePow {
    pub fn new(alpha: Gf, _max_bits: usize, window: usize) -> Self {
        Self(field::FixedBasePow::new_public(field::Gf128Ops, alpha, window))
    }

    #[inline]
    pub fn pow(&self, exp: u128) -> Gf {
        self.0
            .pow_public(&field::Uint::<2>::from_words([exp as u64, (exp >> 64) as u64]))
    }
}

/// Their comb window: `FixedBasePow` covers the full 128-bit exponent range
/// either way; the window only trades table size against multiplies.
pub const WINDOW: usize = 8;

/// The prover's derived setup: the parameters and the comb over their
/// generator.
pub struct BitZProver {
    params: BitZParams,
    comb: FixedBasePow,
    #[cfg(feature = "bench-internals")]
    cached_forest_rounds: bool,
}

/// The verifier's derived setup, the same two things.
pub struct BitZVerifier {
    params: BitZParams,
    comb: FixedBasePow,
}

/// A proof the prover cannot produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProveError {
    /// The bit rows are not the length the shape calls for.
    Witness,
    /// The fold round failed.
    Fold(fold::SendError),
    /// The derived GKR weight counts do not match the table shape.
    Reduction(params::ClaimError),
    /// The opening failed, so the reduction's claim was never discharged.
    Opening(pcs::ProveError),
    /// The virtual statement, map or parameters do not fit.
    Virtual(virt::VirtualError),
    /// The structured (chained-map) statement does not fit.
    Chained(chained::ChainedError),
}

/// A proof the verifier rejects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// The fold round failed its own checks.
    Fold(fold::ReceiveError),
    /// The reduction failed.
    Reduction(reduce::ReduceError),
    /// The opening did not discharge the reduction's claim.
    Opening(pcs::VerifyError),
    /// A stream held bytes the protocol never read.
    TrailingData,
    /// The virtual statement, map or parameters do not fit.
    Virtual(virt::VirtualError),
    /// The structured (chained-map) statement does not fit.
    Chained(chained::ChainedError),
}

impl BitZProver {
    pub fn new(params: BitZParams, window: usize) -> Self {
        let comb = FixedBasePow::new(params.generator(), 128, window);
        Self {
            params,
            comb,
            #[cfg(feature = "bench-internals")]
            cached_forest_rounds: true,
        }
    }

    /// Selects the ordinary or cached pivotal-round prover in controlled
    /// benchmarks. Production builds always use the cached prover.
    #[cfg(feature = "bench-internals")]
    pub fn with_cached_forest_rounds(mut self, cached: bool) -> Self {
        self.cached_forest_rounds = cached;
        self
    }

    pub fn params(&self) -> &BitZParams {
        &self.params
    }

    pub(crate) fn comb(&self) -> &FixedBasePow {
        &self.comb
    }

    /// Their `BitZProver::prove`: bind the root and the parameters, fold,
    /// reduce through GKR, open.
    ///
    /// `rows` are the committed per-column bit rows (the layout
    /// [`crate::ligerito_flock::commit_rs_ligerito_rows`] takes), `hint` the
    /// commitment they produced under `pcs`. `ood` is an out-of-domain
    /// evaluation claim `MLE[P](point) = y` on the packed message that a
    /// composed caller bound before its own challenges (the crate's Round
    /// 0); it is batched into the final Ligerito opening. `None` is their
    /// protocol as shipped.
    pub fn prove(
        &self,
        claim: &LinearClaim,
        pcs: &Pcs,
        hint: &FlockCommitHint,
        transcript: &mut ProverState,
        ood: Option<(&[Gf], Gf)>,
    ) -> Result<(), ProveError> {
        let shape = *self.params.shape();
        let rows = hint.rows();
        if rows.len() != shape.columns()
            || rows.iter().any(|row| row.len() * 64 != shape.rows())
        {
            return Err(ProveError::Witness);
        }
        let root = Root(*hint.root());

        // Step 1: bind. The root and the parameters; the claim itself enters
        // through the caller's own events.
        transcript.public_message(&root.0);
        transcript.public_message(&self.params);

        // Steps 3 and 4: integer column folds, then GKR to a factored bit claim.
        trace_start();
        let started = std::time::Instant::now();
        let fold = self
            .send_fold(claim, rows, transcript)
            .map_err(ProveError::Fold)?;
        trace("fold+images", started);
        let started = std::time::Instant::now();
        #[cfg(feature = "bench-internals")]
        let query = reduce::gkr_reduce_prove_with_cached_rounds(
            transcript,
            &fold,
            &shape,
            hint,
            self.cached_forest_rounds,
        )
        .map_err(ProveError::Reduction)?;
        #[cfg(not(feature = "bench-internals"))]
        let query = reduce::gkr_reduce_prove(transcript, &fold, &shape, hint)
            .map_err(ProveError::Reduction)?;
        trace("gkr", started);

        // Step 6: inner-product sumcheck, ring switching, and the opening.
        let started = std::time::Instant::now();
        let result = pcs
            .prove_lin(hint, &query, StatementBinding::Bind, transcript, ood)
            .map_err(ProveError::Opening);
        trace("opening (all)", started);
        result
    }
}

impl BitZVerifier {
    pub fn new(params: BitZParams, window: usize) -> Self {
        let comb = FixedBasePow::new(params.generator(), 128, window);
        Self { params, comb }
    }

    pub fn params(&self) -> &BitZParams {
        &self.params
    }

    pub(crate) fn comb(&self) -> &FixedBasePow {
        &self.comb
    }

    /// Their `BitZVerifier::verify`, consuming the transcript so both streams
    /// are checked for exhaustion here. `ood` as in [`BitZProver::prove`].
    pub fn verify(
        &self,
        claim: &LinearClaim,
        pcs: &Pcs,
        com: Root,
        mut transcript: VerifierState<'_>,
        ood: Option<(&[Gf], Gf)>,
    ) -> Result<(), VerifyError> {
        transcript.public_message(&com.0);
        transcript.public_message(&self.params);

        let started = std::time::Instant::now();
        let fold = self
            .receive_fold(claim, &mut transcript)
            .map_err(VerifyError::Fold)?;
        trace("v: fold", started);
        let started = std::time::Instant::now();
        let query = reduce::gkr_reduce_verify(&mut transcript, &fold, self.params.shape())
            .map_err(VerifyError::Reduction)?;
        trace("v: gkr", started);

        let started = std::time::Instant::now();
        pcs.verify_lin(&com, &query, StatementBinding::Bind, &mut transcript, ood)
            .map_err(VerifyError::Opening)?;
        trace("v: opening", started);

        transcript
            .check_eof()
            .map_err(|_| VerifyError::TrailingData)?;
        Ok(())
    }
}

/// Phase timing to stderr when `BITZ_TRACE` is set (diagnostic only), with
/// the minor page faults taken since the previous trace line: fresh pages
/// cost ≈ 0.5 µs each (≈ 30 ms per GB) on this platform and do not
/// parallelise, so the count says how much of a phase is first touch.
pub(crate) fn trace(label: &str, started: std::time::Instant) {
    let elapsed = started.elapsed();
    if let Ok(mut phases) = PHASES.lock() {
        if let Some(recorded) = phases.as_mut() {
            recorded.push((label.trim().to_string(), elapsed));
        }
    }
    if std::env::var_os("BITZ_TRACE").is_some() {
        let faults = minor_faults();
        let previous = LAST_FAULTS.swap(faults, std::sync::atomic::Ordering::Relaxed);
        let delta = faults.saturating_sub(previous);
        eprintln!(
            "bitz: {label:<24} {elapsed:>9.1?}  {:>6.1} MB faulted",
            delta as f64 * 16.0 / 1024.0
        );
    }
}

static LAST_FAULTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The phase timings a bench collects programmatically: while recording
/// is on, every [`trace`] call also appends `(label, elapsed)` here.
static PHASES: std::sync::Mutex<Option<Vec<(String, std::time::Duration)>>> =
    std::sync::Mutex::new(None);

/// Starts (`true`) or stops (`false`) collecting phase timings; starting
/// discards anything collected before.
pub fn record_phases(on: bool) {
    if let Ok(mut phases) = PHASES.lock() {
        *phases = if on { Some(Vec::new()) } else { None };
    }
}

/// The phases recorded since the last call (or since recording started),
/// in completion order, labels trimmed.
pub fn take_phases() -> Vec<(String, std::time::Duration)> {
    PHASES
        .lock()
        .ok()
        .and_then(|mut phases| phases.as_mut().map(std::mem::take))
        .unwrap_or_default()
}

/// Resets the fault baseline so the first trace line of a run counts only
/// its own faults.
pub(crate) fn trace_start() {
    if std::env::var_os("BITZ_TRACE").is_some() {
        LAST_FAULTS.store(minor_faults(), std::sync::atomic::Ordering::Relaxed);
    }
}

/// The process's minor page faults so far (16 KB pages on this platform).
fn minor_faults() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: `getrusage` fills the struct for the calling process.
    let ok = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if ok != 0 {
        return 0;
    }
    // SAFETY: filled by the call above.
    let usage = unsafe { usage.assume_init() };
    usage.ru_minflt as u64
}

/// `eq(r, z)` for one coordinate pair, the way their `poly::eq::eq_eval`
/// computes it: `r·z + (1 − r)(1 − z)`.
pub(crate) fn eq_factor(r: Gf, z: Gf) -> Gf {
    let one = Gf::one();
    r * z + (one - r) * (one - z)
}
