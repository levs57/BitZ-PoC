
# BitZ 🫜

PoC implementation of BitZ PCS and BitZ-SNARK, from the paper [insert_link]. 


## Reproducing the paper's benchmarks

Run all commands below from the repository root. Dependency checkouts use
sibling directories such as `../limber-impl`.

Cross-system comparisons measure complete native proofs: witness generation,
commitment, constraint proving, PCS opening, and verification.

Install the native Perfetto trace processor locally, then set its path in each
shell used for benchmarks. Run these commands from the repository root:

```bash
bash scripts/install_trace_processor.sh
export PERFETTO_TRACE_PROCESSOR="$PWD/.tools/perfetto/trace_processor_shell"
export RUSTFLAGS="-C target-cpu=native"
```

The installer downloads Perfetto **v58.2** for macOS or Linux, verifies its
SHA-256 checksum, and reuses an existing matching installation. It requires
`curl` and `sha256sum` or `shasum`. The binary stays in this checkout's ignored
`.tools/perfetto/` directory; no system installation is needed.

### The benchmark campaigns

The campaign instructions live in [BENCH_INSTRUCTIONS.md](BENCH_INSTRUCTIONS.md) and are copied
below verbatim by `scripts/sync_readme_campaigns.py` (the artifact workspace's README carries the
same copy); edit the source file, then run the script.

<!-- bench-campaigns:begin (generated from BENCH_INSTRUCTIONS.md by scripts/sync_readme_campaigns.py; edit that file) -->
This file is the single source of the benchmark instructions. `README.md`
carries a verbatim copy between its two `bench-campaigns` HTML-comment
markers, refreshed by `python3 scripts/sync_readme_campaigns.py` and checked
by `python3 -m unittest discover -s scripts -p test_readme_sync.py`; the
anonymised artifact workspace splices its own filtered copy (without the
Zinc+ campaign) into its README the same way. Edit here, then sync.

Run the following commands from the repository root in **Bash**. They execute
benchmarks and verify the generated proofs. Use a fresh `RUN_DIR` for each
campaign and run the workloads sequentially on an otherwise idle machine:
`scripts/bench_gate.py` serialises campaigns with a machine-wide lock, waits
for 88% CPU idle held for 120 s before starting, and aborts a campaign whose
swap growth passes its guard. Every scheme, including the competitors, is
built at fat LTO with one codegen unit and `-C target-cpu=native`; the
runners resolve the executables they measure from Cargo's
`--message-format=json` output, and out-of-crate workers carry their own
release profile with the same settings (verify a worker's effective flags in
its `--build-info` before trusting its numbers).

### Setup

```bash
set -euo pipefail

rustup toolchain install 1.98.1
rustup toolchain install nightly-2026-07-01
bash scripts/install_trace_processor.sh

export RUSTFLAGS="-C target-cpu=native"
export PERFETTO_TRACE_PROCESSOR="$PWD/.tools/perfetto/trace_processor_shell"
unset BITZ_LIG_PROFILE CARGO_ENCODED_RUSTFLAGS CARGO_TARGET_DIR

mkdir -p bench_results
export RUN_DIR="$(mktemp -d "$PWD/bench_results/all-benchmarks-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
# scripts/prepare_matched_limber.py fetches the Cargo.toml-pinned Limber
# commit here on demand; set LIMBER_DIR to reuse an existing checkout.
LIMBER_DIR="${LIMBER_DIR:-$PWD/.tools/limber}"
gate() { python3 scripts/bench_gate.py run --label "$1" --swap-grow-gb "${2:-12}" -- "${@:3}"; }
echo "Results: $RUN_DIR"
```

### 1. SHA-256 + ECDSA over secp256k1: BitZ (matched circuit), Binius64, Binius64-Ligerito

The head-to-head. BitZ runs the circuit matched to Binius64's stock verifier
schedule (`sha256-chain-secp256k1/bitz-binius64-matched/v1`); Binius64 and
the BitZ-opener rows run the pinned fork's upstream `ecdsa::bitcoin_verify`
(`sha256-chain-secp256k1/binius64-bitcoin-verify/v1`). Every row records its
`curve` and `circuit_profile`, and the runner refuses any other pairing. See
`docs/sha256-ecdsa-comparison.md` for what "matched" means and what differs.

```bash
gate sha-ecdsa-secp256k1 12 python3 scripts/run_sha256_ecdsa_compare.py \
  --curve secp256k1 \
  --output "$RUN_DIR/sha256-ecdsa-secp256k1" \
  --methods bitz-split binius64 binius64-ligerito \
  --exponents 4 5 6 7 \
  --targets 100 \
  --threads 1 10 \
  --reps 5 \
  --bitz-profiles custom:1:4 custom:3:4 \
  --binius-rates 1 3 \
  --timing perfetto \
  2>&1 | tee "$RUN_DIR/sha256-ecdsa-secp256k1.log"
```

### 2. SHA-256 + ECDSA over P-256: BitZ alone

The paper's P-256 circuit (`sha256-chain-p256/bitz-lean-port/v1`). The pinned
Binius64 fork has no P-256 verifier of its own (its P-256 gadget was written
for this comparison), so no Binius row exists on this curve and the runner
refuses to record one.

```bash
gate sha-ecdsa-p256 12 python3 scripts/run_sha256_ecdsa_compare.py \
  --curve p256 \
  --output "$RUN_DIR/sha256-ecdsa-p256" \
  --methods bitz-split \
  --exponents 4 5 6 7 \
  --targets 100 \
  --threads 1 10 \
  --reps 5 \
  --bitz-profiles custom:1:4 custom:3:4 \
  --timing perfetto \
  2>&1 | tee "$RUN_DIR/sha256-ecdsa-p256.log"
```

### 3. SHA-256 chains: BitZ, Binius64, Binius64-Ligerito

```bash
gate sha256-chain 12 python3 scripts/run_sha256_chain_compare.py \
  --methods bitz binius64 binius64-ligerito \
  --exponents 7 8 9 10 11 12 13 14 15 16 \
  --threads 1 10 \
  --reps 5 \
  --bitz-profiles custom:1:4 custom:3:4 \
  --binius-rates 1 3 \
  --output "$RUN_DIR/sha256-chain" \
  2>&1 | tee "$RUN_DIR/sha256-chain.log"
```

### 4. Multiplication comparisons: BitZ, Binius64, Binius64-Ligerito, Limber

The launcher builds once, validates every selection with the Rust case
planner, runs one worker process per case, records a separate single-proof
peak-RSS trial (`--memory rss`), and generates combined reports under
`<output>/reports` (schema `mul-bench/v2`). Run one gated
invocation per workload; the launcher gates each campaign itself, so omit
`--no-gate` unless an outer gate already holds the lock. Cells whose prover
exceeds the machine's memory are excluded, not measured while paging:
`scripts/mul_memory_probe.py` runs one such cell alone under a small
swap-growth guard and appends a record with the verdict, the observed peak
and the installed memory to a JSONL file that the table generator reads (see
"Tables and figures"), for example

```bash
python3 scripts/mul_memory_probe.py --workload u128 --backend binius64 --log-n 21 \
  --threads 10 --log-inv-rate 1 --record "$RUN_DIR/mul-exclusions.jsonl" \
  --output "$RUN_DIR/multiplication-probes/u128-binius64-rate2-t10-n21"
```

```bash
for workload in u32-mod32 u64 u128; do
  python3 scripts/run_multiplication_benchmarks.py compare \
    --output "$RUN_DIR/multiplication-$workload" -- \
    proof --workload "$workload" --backends bitz,binius64,binius64-ligerito,limber \
    --log-n 15,17,19 --threads 1,10 --reps 5 --memory rss --skip-unsupported \
    --binius-ligerito-accounting rbr \
    2>&1 | tee "$RUN_DIR/multiplication-$workload.log"
done
```

Sizes above `2^19` are a per-backend decision on a 24 GiB machine: BitZ and
Binius64 at rate 1/2 reach `2^21`; BitZ reaches `2^23`
for u32 and u64; the others page first. Add `--log-n 21` (or `23`) runs for
the backends that fit and record the rest as exclusions.

### 5. Multiplication: the Zinc+ rows

Zinc+ cannot be linked into this crate (it pins `crypto-bigint = 0.7.0-rc.9`
against `vendor/field`'s 0.7.5), so its rows come from an external worker
built inside a pinned zinc-plus checkout. The runner exports the exact BitZ
corpora with `examples/mul_corpus_export`, builds a single-threaded and a
`parallel` worker (features `simd,unchecked[,parallel]`, rate 1/4, 100 bits;
the first build also runs once in CHECKED mode at the smallest size), runs
one process per case under `/usr/bin/time -l`, and imports the logs into a
`mul-bench/v2` campaign whose rows the table treats like every other
backend's. Zinc+ reports no security accounting of its own; the imported
configuration records the inverse rate, column openings, projecting-prime
width, grinding bits and the LogUp range-check term per case.

```bash
cargo +1.98.1 build --release --locked --example mul_corpus_export
gate zinc-plus 12 python3 scripts/run_zinc_plus_campaign.py \
  --output "$RUN_DIR/zinc-plus" \
  --workdir "$RUN_DIR/zinc-plus-checkout" \
  --revision 878fbd8292472dcb13b25e2c9c0209406b5fb671 \
  --corpus-exporter target/release/examples/mul_corpus_export \
  --workloads u32-mod32 u64 u128 --exponents 15 17 19 --threads 1 10 --reps 5 \
  2>&1 | tee "$RUN_DIR/zinc-plus.log"
```

### 6. BitZ full-product u32 × u32 → u64, with component breakdown

```bash
cargo +1.98.1 run --release --locked --bin bitz \
  --features unchecked,span-metrics -- \
  --mul-sweep 15-22 \
  --threads 10 \
  --reps 5 \
  --profile custom:1:4 \
  --cooldown 20 \
  --latex "$RUN_DIR/u32-full-product.tex" \
  2>&1 | tee "$RUN_DIR/u32-full-product.log"
```

### 7. MultiSwap: BitZ, Limber-Hyrax, Limber-Brakedown

This uses the Limber checkout in `LIMBER_DIR`. `MSCFG=paper` is a workload
name and does not require a manuscript directory. `--draft` runs proofs and
the local comparison checks while marking canonical trace validation as
pending.

```bash
gate multiswap 10 python3 scripts/run_matched_multiswap_campaign.py \
  --draft \
  --limber-root "$LIMBER_DIR" \
  --security-bits 114 \
  --batch-counts 1,2,4,8,16 \
  --all-threads 10 \
  --warmups 1 \
  --samples 10 \
  --rustflags="-C target-cpu=native" \
  --output-dir "$RUN_DIR/multiswap" \
  2>&1 | tee "$RUN_DIR/multiswap.log"
```

### 8. SHA-256 layout parameter sweep over s and t

The `sha256_product_layout` benchmark holds the workload at `2^14` SHA-256
compressions and sweeps the layout split with `s + t = 29`. By default, it runs
`t = 7..27` (`s = 29 - t`), with one warmup and 21 measured samples per split.
The `t = 28` case is skipped because its projected peak memory exceeds 60 GiB.
This is a controlled fixed-prime layout experiment.

```bash
RUSTFLAGS="-C target-cpu=native" \
RAYON_NUM_THREADS=10 \
cargo +1.98.1 bench --locked \
  --bench sha256_product_layout \
  --features unchecked,span-metrics,bench-internals
```

To select particular splits and change the sample count, prepend
`BITZ_SHA_PRODUCT_TS="13 17" BITZ_BENCH_REPS=5` to the command. This selects
`(t, s) = (13, 16)` and `(17, 12)`, with five measured samples per split.
Add `--no-run` to the Cargo command to compile without executing the sweep.

### 9. Hybrid SHA-256 chain + multiplication modulo 2^32

This is the paper's **"Modular multiplications and bit operations"** experiment
(table label `tab:hybrid-sha256-mul`). It proves `N` relations
`x*y = z + 2^32*w`, with four u32 limbs, together with `M = N/256` chained SHA-256
compressions. The two branches have equal packed witness sizes; their witness
values are independent. Shape `15:7`, for example, means `2^15` multiplications
and `2^7` compressions.

| CLI mode | Multiplication and SHA proof |
|---|---|
| `hybrid` | BitZ multiplication PIOP + Binius64 SHA PIOP, with one shared BitZ opening |
| `all-binius` | Both relations in Binius64, using BaseFold/FRI (paper: Binius UDR) |
| `binius-ligerito` | Both relations in Binius64, using the BitZ/Ligerito opener (paper: Binius Johnson) |

Every other BitZ benchmark in this document is built with the `unchecked`
feature (release integer arithmetic, no overflow guards); the hybrid bench is
the one exception: its constructor refuses `unchecked` because the hybrid
proof relies on checked arithmetic and constraints, so it is built with
`--features hybrid` alone and its rows are checked-arithmetic numbers.

The sweep below runs all three modes at rates 1/2 and 1/8, with 1 and 10
threads, one warmup and five measured iterations per shape. The BitZ hybrid
checks a 100-bit whole-protocol union bound; the Binius/Ligerito mode uses
100-bit round-by-round accounting. The BaseFold query target is explicitly
set to 100 below, overriding the hybrid CLI's default of 112.

Build the benchmark once and obtain its executable path from Cargo's artifact
record, then run each configuration in a separate sweep. On macOS, the sampler
also records per-case peak RSS and swap-outs for the table's memory column.

```bash
cargo +1.98.1 bench --locked --no-run \
  --bench hybrid_u32_sha256 --features hybrid --message-format=json \
  > "$RUN_DIR/hybrid-build.jsonl"

HYBRID_BIN="$(python3 - "$RUN_DIR/hybrid-build.jsonl" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    artifacts = [json.loads(line) for line in stream]
executables = [entry["executable"] for entry in artifacts
               if entry.get("reason") == "compiler-artifact"
               and entry.get("target", {}).get("name") == "hybrid_u32_sha256"
               and entry.get("executable")]
if len(executables) != 1:
    raise SystemExit("expected exactly one hybrid benchmark executable")
print(executables[0])
PY
)"

HYBRID_SHAPES="15:7,16:8,17:9,18:10,19:11,20:12"
HYBRID_ROOT="$RUN_DIR/hybrid-witness"
mkdir -p "$HYBRID_ROOT"

hybrid_sweep() {
  local mode="$1" rate="$2" threads="$3"
  local dir="$HYBRID_ROOT/$mode-rate$rate-t$threads"
  local tsv="$dir-peak-rss-and-swap.tsv"
  local command=("$HYBRID_BIN" --sweep --mode "$mode"
    --shapes "$HYBRID_SHAPES" --iterations 5 --results-dir "$dir")
  if [[ "$mode" == hybrid ]]; then
    command+=(--profile "custom:$rate:4")
  fi
  if [[ "$(uname -s)" == Darwin ]]; then
    command=(python3 scripts/rss_sampler.py --output "$tsv" -- "${command[@]}")
  fi
  env RAYON_NUM_THREADS="$threads" \
    BITZ_HYBRID_BINIUS_LOG_INV_RATE="$rate" \
    BITZ_HYBRID_BINIUS_SECURITY_BITS=100 \
    BITZ_BINIUS_LOG_INV_RATE="$rate" \
    BITZ_BINIUS_LIGERITO_ACCOUNTING=rbr \
    "${command[@]}" 2>&1 | tee "$dir.log"
  if [[ -f "$tsv" ]]; then
    mv "$tsv" "$dir/peak-rss-and-swap.tsv"
  fi
}

for threads in 1 10; do
  for rate in 1 3; do
    for mode in hybrid all-binius binius-ligerito; do
      hybrid_sweep "$mode" "$rate" "$threads"
    done
  done
done

echo "Completed. Results: $RUN_DIR"
```

For the optional **equal-operation-count** experiment (`N = M`), set the
following variables, then repeat the three nested `for` loops above:

```bash
HYBRID_SHAPES="9:9,10:10,11:11,12:12,13:13,14:14"
HYBRID_ROOT="$RUN_DIR/hybrid-counts"
mkdir -p "$HYBRID_ROOT"
```

This is a separate workload from the paper's equal-witness table. Each sweep's
result directory must not already exist. It will contain `summary.csv`,
`run.txt`, and per-case CSV/log files.

### Tables and figures

Every table is generated from recorded campaign output; none is typed by
hand, and each carries in its header comments the generator, the run
directories, the machine, and the per-row medians needed to regenerate it.

```bash
# SHA-256 + ECDSA: one table per curve (mixing curves is refused).
python3 scripts/sha256_ecdsa_table.py "$RUN_DIR/sha256-ecdsa-secp256k1" \
  --out "$RUN_DIR/sha256-ecdsa-secp256k1-table.tex"
python3 scripts/sha256_ecdsa_table.py "$RUN_DIR/sha256-ecdsa-p256" \
  --out "$RUN_DIR/sha256-ecdsa-p256-table.tex"

# Multiplication: one table per workload from every campaign that measured it
# (the launcher's directories and the imported Zinc+ campaign). Excluded cells
# come from the memory probe's JSONL records (or a JSON list) with their
# reason, observed peak and machine memory; the file may be empty.
touch "$RUN_DIR/mul-exclusions.jsonl"
for workload in u32-mod32 u64 u128; do
  python3 scripts/mul_table.py "$RUN_DIR/multiplication-$workload" "$RUN_DIR/zinc-plus/campaign" \
    --workload "$workload" --exclusions "$RUN_DIR/mul-exclusions.jsonl" \
    --out "$RUN_DIR/native-mul-$workload-table.tex"
done

# Hybrid witness sweeps.
hybrid_rows=()
for threads in 1 10; do
  for rate in 1 3; do
    for mode in hybrid all-binius binius-ligerito; do
      hybrid_rows+=(--row "$mode@$rate:$threads=$RUN_DIR/hybrid-witness/$mode-rate$rate-t$threads")
    done
  done
done
python3 scripts/hybrid_table.py --variant witness "${hybrid_rows[@]}" \
  --output "$RUN_DIR/hybrid-witness.tex"
```

For equal-count results, use `--variant counts` and the `hybrid-counts`
directories instead. Generation of a table does not rerun the proofs. The
bundled `scripts/zk_trace.py` (artifact workspace) provides trace report
tooling; tables default to `outputs/tables/` and figures to `outputs/figures/`
there.
<!-- bench-campaigns:end -->

### Cached-round forest benchmark

For a same-process comparison of the cached two-round prover with the ordinary
round path, run:

```bash
RUSTFLAGS="-C target-cpu=native -A warnings" RAYON_NUM_THREADS=1 \
  cargo run --release --no-default-features \
  --features bitz-parity,bench-internals \
  --example wfbitz_bench -- 26 --reps 5 --ladder fast \
  --compare-cached-rounds
```

Both modes reuse one instance and commitment. Their samples run in alternating
order; compare the separately reported `gkr` and `grand products` medians.

### Raw performance of BitZ PCS on the core LinBitsRings relation

```sh
RUSTFLAGS="-C target-cpu=native" cargo run --release --features unchecked -- \
    --sweep 20-30 --threads 8 --reps 5 --profile custom:1:4
```

### Comparison with fields-witch (Soukhanov's characteristic-2 field switch)

[fields-witch](https://github.com/morgana-proofs/fields-witch) is Lev
Soukhanov's implementation of his "Char 2 fieldswitch" note: it commits
`2^k` entries of `F_{2^127}` (integers below `2^127`) densely over
`F_{2^128}` and proves their multilinear evaluation over `F_p`,
`p = 2^127 - 1`. The comparison is bit-matched: `2^k` entries of 127 bits
are the same 16 bytes per entry as BitZ at `n = k + 7` with `W = 1`.
`scripts/run_fields_witch_compare.py` derives fields-witch's per-round limb
schedules from its README rule (it reproduces the README's `2^20` schedule
exactly), runs every (scheme, size, threads) cell in a fresh process under
`/usr/bin/time -l` on a quiet box (CPU-idle gate), and writes
`PerfRuns/<stamp>-fields-witch-compare/{results.jsonl,summary.md,fields-witch-table.tex}`;
see `docs/fields-witch-compare.md` for the measured comparison.

```sh
git clone https://github.com/morgana-proofs/fields-witch ../fields-witch   # measured at 30cca8c
(cd ../fields-witch && CARGO_TARGET_DIR=target \
    RUSTFLAGS="-C target-cpu=native" cargo build --release --examples)
RUSTFLAGS="-C target-cpu=native" cargo build --release --features unchecked --bin bitz
python3 scripts/run_fields_witch_compare.py \
    --fw-bin ../fields-witch/target/release/examples/protocol_profile \
    --sizes 14,16,18,20,22 --threads 1,8 --reps 5 \
    --word-rows 20:32,20:64 --latex paper/fields-witch-table.tex
```

`--fw-bin` / `--bitz-bin` override the binaries (the BitZ default follows
`CARGO_TARGET_DIR`); `--bitz-profile udr:1:4` measures BitZ in fields-witch's
unique-decoding regime; `--render-latex <results.jsonl> --latex <path>`
regenerates the paper table from a finished run.

### Integer multiplication

Choose `bitz` for standalone BitZ experiments or `compare` for cross-system
comparisons. Launcher options go before `--`; Rust benchmark flags go after it.
Standalone proof timings exclude native witness generation; comparison proof
timings include it.

```sh
python3 scripts/run_multiplication_benchmarks.py bitz -- \
  proof --workload u32-full,u64,u128 --log-n 15..=20 --w 1,3,8 \
  --split=0,1 --threads 1,8 --reps 5 --skip-unsupported --dry-run

python3 scripts/run_multiplication_benchmarks.py compare --output results/compare -- \
  proof --workload u32-mod32,u64,u128 --backends all --log-n 15,17,19 \
  --threads 1,8 --skip-unsupported --memory rss

python3 scripts/run_multiplication_benchmarks.py bitz -- \
  witness --workload u32-full,u64,u128,baby-bear --log-n 10 --threads 1

# the worldfnd/BitZ scheme as the opener (feature bitz-parity, added by the
# launcher): `--ligerito fast` is its ladder as shipped; see docs/wfbitz-opener.md
python3 scripts/run_multiplication_benchmarks.py bitz -- \
  proof --workload u64 --opener wfbitz --ligerito fast --bitz-profile 100 \
  --log-n 15,17,19,21 --threads 1,10 --reps 5 --memory rss
```

The launcher builds one executable, runs under the machine lock and swap guard,
and writes shared reports under `<output>/reports`. It defaults to a fresh
`PerfRuns/<timestamp>-multiplication` directory. `--no-gate` disables the gate;
`--swap-grow-gb` changes its default 12 GiB limit. Build settings are preserved,
with native CPU compilation used when no Rust flags are supplied.

Launcher `--dry-run` before `--` prints commands without building or writing
files. Benchmark `--dry-run` after `--` builds and validates the expanded cases
without proving. Forwarded `--help` shows the selected benchmark's flags.
Both targets support `proof`, `witness`, and `pcs`; `bitz` additionally supports
`piop`, `outer`, and `bounds`. See the
[multiplication benchmark guide](docs/native-mul-compare.md) for configuration,
measurement boundaries, memory passes, and the `mul-bench/v2` result format.

### RSA MultiSwap — matched 114-bit comparison

Compare **BitZ/Ligerito, Limber-Hyrax, and Limber-Brakedown** on Limber's
Table 1 fixture. One circuit copy contains **4 exponentiations with 352-bit
exponents modulo a 2048-bit RSA modulus**, with 6,209 live integer constraint
rows. The RSA chains execute; hash and Poseidon operations contribute modeled
costs. The fixture has no application public inputs (`count=0`, `values=[]`)
and does not prove a complete public accumulator transition.

The campaign fixes `k=0` and proves **1, 2, 4, 8, or 16 complete circuit
copies in one proof**: 4–64 RSA exponentiations. It checks matching canonical
statements and witness data across backends. Each modeled security check must
reach **at least 114 bits**; the shared 128-bit prime fingerprint retains its
roughly 114-bit bound. This accounting is per check/round, not a combined
whole-proof soundness bound or an RSA key-strength claim. Limber retains its
native 128-bit integer target and 117-bit integer challenge bound target.

Run these commands from the repository root. MultiSwap benchmarks the existing
Limber checkout supplied through `--limber-root` and records its revision for
provenance. If you need a checkout, the optional setup helper clones the Cargo
dependency revision into a destination that does not already exist:

```sh
python3 scripts/prepare_matched_limber.py ../limber-impl
```

This clones the dependency revision directly, with no patching or local commits.
Use `--limber-root ../limber-impl` for the sibling
checkout. The runner does not require its revision to match the Cargo dependency.
The setup and campaign scripts require Python 3.11 or newer.

The runner requires this repository's pinned Rust toolchain and Limber's
`nightly-2026-07-01`. It sets `MSCFG=paper` and each backend's security
parameters, overriding inherited workload/security settings. Run the full
sweep with **1 and 16 threads**, one warmup, and ten measured proofs per
configuration (**30 configurations**):

```sh
python3 scripts/run_matched_multiswap_campaign.py \
  --draft \
  --limber-root ../limber-impl \
  --security-bits 114 \
  --batch-counts 1,2,4,8,16 \
  --all-threads 16 \
  --warmups 1 \
  --samples 10 \
  --rustflags="-C target-cpu=native"
```

`--draft` runs proof verification and repository comparison checks, but marks
the results as **pending canonical validation**. The external
`zk-proof-profiler/scripts/zk_trace.py` validator is not bundled here. For
canonical execution, replace `--draft` with `--profiler` followed by the
actual path to that file. A placeholder path will fail preflight.

Add `--dry-run` to preview the commands without compiling or running proofs.
For a smoke run, change to `--batch-counts 1 --samples 1` (six configurations).
Results appear under `bench_results/<campaign>/reports/combined/` as
`summary.json`, `metrics.csv`, and `intervals.html`. They report witness,
commitment-plus-proving, combined prover, and verification times, along with
proof sizes including commitments and process peak memory. Compilation and
setup are excluded from headline proving times; analytical proof-size
estimates are marked.

See the [campaign guide](docs/matched-multiswap-campaign.md) for the statement,
security accounting, toolchain setup, and validation requirements. The
[historical comparison rows](#historical-multiswap-comparison-rows-limber-zinc)
below predate this matched campaign.


### SHA-256 
```sh
CARGO_TARGET_DIR=target RUSTFLAGS="-C target-cpu=native" \
RAYON_NUM_THREADS=8 \
BITZ_SHA_COMPARE_EXPONENTS="4 5 6 7 9 10 11 12" \
BITZ_SHA_COMPARE_REPS=5 \
BITZ_SHA_COMPARE_BACKENDS="bitz binius64" \
BITZ_SHA_COMPARE_OUTPUT_DIR="PerfRuns/$(date -u +%Y-%m-%dT%H-%M-%SZ)-sha256-compare" \
  cargo bench --bench sha256_e2e_compare --features bench-internals,native-sha256-compare
```

## License

MIT. See [LICENSE](LICENSE). Vendored and pinned dependencies (`vendor/`, the
`flock-core`, Binius64, Plonky3, and Limber pins) carry their own licenses.
