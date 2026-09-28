//! The dense-round arithmetic of the parity prover: the Gruen round sums,
//! the fused fold-and-round pass, and the just-in-time variants of the
//! first two dense rounds that read their entries through the per-position
//! value tables instead of a materialised layer.
//!
//! NEON kernels on aarch64 (register-resident 256-bit accumulators, the
//! pass-fixed fold multiplier preprocessed, two independent chains per
//! loop), generic field arithmetic elsewhere. Both compute the same field
//! elements: only exact operations are reordered — the unreduced carryless
//! products are XOR-combined and reduced once per accumulator, and
//! reduction is `F₂`-linear — so every transcript byte is unchanged.

use std::mem::MaybeUninit;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::cfg_into_iter;
use field::Gf128 as Gf;
use crate::utils::wide_mul::WideMulAcc;

/// The longest run one task handles when a round has a single row (the
/// flat rounds and the `s = 0` layers).
const MAX_TASK: usize = 1 << 12;

/// Pattern blocks come transposed ([`super::forest::transpose_blocks`]):
/// position `m` of a 64-byte block holds column `((m & 7) << 3) | (m >> 3)`
/// of its group. Weights are handed over in the same order (`eq_t`), zero
/// past the group's last column, so only stores need the column back.
#[inline(always)]
pub(crate) fn col_of(m: usize) -> usize {
    ((m & 7) << 3) | (m >> 3)
}

/// The weight of a dense round: `eq_y[y] · eq_c[c]` over rows `y` of
/// `eq_c.len()` entries (`eq_y = [1]` for a single row).
#[derive(Clone, Copy)]
pub(crate) struct Weights<'a> {
    pub eq_c: &'a [Gf],
    pub eq_y: &'a [Gf],
}

impl Weights<'_> {
    fn width(&self) -> usize {
        self.eq_c.len()
    }

    fn rows(&self) -> usize {
        self.eq_y.len()
    }

    /// The task length: whole rows when there are several, runs of at
    /// most [`MAX_TASK`] entries when there is one.
    fn task_len(&self) -> usize {
        if self.rows() > 1 {
            self.width()
        } else {
            self.width().min(MAX_TASK).max(1)
        }
    }

    /// The row and the column offset of task `i`.
    fn task_pos(&self, i: usize) -> (usize, usize) {
        let start = i * self.task_len();
        let y = start / self.width();
        (y, start - y * self.width())
    }
}

/// Sums the per-task partials — exact in any order (characteristic two).
fn total(partials: Vec<(Gf, Gf)>) -> (Gf, Gf) {
    partials
        .into_iter()
        .fold((Gf::zero(), Gf::zero()), |(a, b), (x, y)| (a + x, b + y))
}

/// `Σ eq·E_end·O_end` and `Σ eq·(E_hi − E_lo)(O_hi − O_lo)` over the four
/// halves, `E_end = E_hi` when `send_one`.
pub(crate) fn round_sums(
    lo_l: &[Gf],
    hi_l: &[Gf],
    lo_r: &[Gf],
    hi_r: &[Gf],
    w: Weights<'_>,
    send_one: bool,
) -> (Gf, Gf) {
    let h = w.width() * w.rows();
    assert_eq!(lo_l.len(), h);
    assert_eq!(hi_l.len(), h);
    assert_eq!(lo_r.len(), h);
    assert_eq!(hi_r.len(), h);
    let task = w.task_len();
    let tasks = h.div_ceil(task);
    let partials: Vec<(Gf, Gf)> = cfg_into_iter!(0..tasks, 1)
        .map(|i| {
            let (y, c0) = w.task_pos(i);
            let start = i * task;
            let end = (start + task).min(h);
            let ec = &w.eq_c[c0..c0 + (end - start)];
            let (e, f) = round_sums_task(
                &lo_l[start..end],
                &hi_l[start..end],
                &lo_r[start..end],
                &hi_r[start..end],
                ec,
                send_one,
            );
            let wy = w.eq_y[y];
            (e * wy, f * wy)
        })
        .collect();
    total(partials)
}

/// The deferred fold of the previous round fused with this round's sums:
/// the four quarters `q0..q3` of each half (the previous round's top index
/// bit selects `q0 q1` against `q2 q3`, this round's the odd quarters) are
/// folded with `rho` — `q0 + rho·(q2 − q0)` into `q0`, `q1 + rho·(q3 − q1)`
/// into `q1` — and the round sums are accumulated over the folded pairs
/// `(q0, q1)` in the same pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_fold_round(
    q0_l: &mut [Gf],
    q1_l: &mut [Gf],
    q2_l: &[Gf],
    q3_l: &[Gf],
    q0_r: &mut [Gf],
    q1_r: &mut [Gf],
    q2_r: &[Gf],
    q3_r: &[Gf],
    rho: Gf,
    w: Weights<'_>,
    send_one: bool,
) -> (Gf, Gf) {
    let h = w.width() * w.rows();
    for q in [&*q0_l, &*q1_l, q2_l, q3_l, &*q0_r, &*q1_r, q2_r, q3_r] {
        assert_eq!(q.len(), h);
    }
    let task = w.task_len();
    let partials: Vec<(Gf, Gf)> = crate::cfg_chunks_mut!(q0_l, task)
        .zip(crate::cfg_chunks_mut!(q1_l, task))
        .zip(crate::cfg_chunks!(q2_l, task))
        .zip(crate::cfg_chunks!(q3_l, task))
        .zip(crate::cfg_chunks_mut!(q0_r, task))
        .zip(crate::cfg_chunks_mut!(q1_r, task))
        .zip(crate::cfg_chunks!(q2_r, task))
        .zip(crate::cfg_chunks!(q3_r, task))
        .enumerate()
        .map(|(i, (((((((a0, a1), a2), a3), b0), b1), b2), b3))| {
            let (y, c0) = w.task_pos(i);
            let ec = &w.eq_c[c0..c0 + a0.len()];
            let (e, f) = fused_task(a0, a1, a2, a3, b0, b1, b2, b3, &rho, ec, send_one);
            let wy = w.eq_y[y];
            (e * wy, f * wy)
        })
        .collect();
    total(partials)
}

/// The running Gruen sums of one row task — unreduced accumulators.
pub(crate) struct Sums(SumsInner);

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
type SumsInner = neon::Sums;
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
type SumsInner = generic::Sums;

impl Sums {
    pub(crate) fn zero() -> Self {
        Self(SumsInner::zero())
    }

    /// `(Σ end, Σ inf)`, reduced.
    pub(crate) fn finish(self) -> (Gf, Gf) {
        self.0.finish()
    }

    #[inline(always)]
    fn slot(&mut self, w: Gf, l0: Gf, l1: Gf, r0: Gf, r1: Gf, send_one: bool) {
        self.0.slot(w, l0, l1, r0, r1, send_one);
    }
}

/// One group of the first just-in-time dense round, slot by slot: `tab[i]`
/// and `pat[i]` (transposed) give the values of corner `i ∈ (E_lo, E_hi,
/// O_lo, O_hi)` at each position (`tab[i][pat[i][m]]`); accumulates the
/// Gruen sums with the transposed weights `eq_t` (64 entries). The prover
/// runs the bucketed form ([`jit_bucket_group`]); this is its reference.
#[allow(dead_code)]
pub(crate) fn jit_sums_group(
    tab: [&[Gf]; 4],
    pat: &[[u8; 64]; 4],
    eq_t: &[Gf],
    send_one: bool,
    sums: &mut Sums,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    neon::jit_sums_group(tab, pat, eq_t, send_one, &mut sums.0);
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    generic::jit_sums_group(tab, pat, eq_t, send_one, &mut sums.0);
}

/// The per-task scratch of the bucketed first table round: three
/// 256-entry tables of unreduced accumulators, keyed by an E pattern —
/// `end` (the endpoint corner's), `inf_lo` and `inf_hi` (E_lo's and
/// E_hi's, both fed `eq·ΔO`).
pub(crate) struct SumBuckets(SumBucketsInner);

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
type SumBucketsInner = neon::SumBuckets;
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
type SumBucketsInner = generic::SumBuckets;

impl SumBuckets {
    pub(crate) fn new() -> Self {
        Self(SumBucketsInner::new())
    }

    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }
}

/// One group of the first just-in-time dense round, bucketed: with
/// `pat[i]` the transposed patterns of the corners `(E_lo, E_hi, O_lo,
/// O_hi)` and `tab_o` the two O tables, folds each column's `eq·O_end`
/// into `end[pat_E_end]` and `eq·(O_hi − O_lo)` into `inf_lo[pat_E_lo]`
/// and `inf_hi[pat_E_hi]`, unreduced — one carryless product per term
/// instead of two multiplies. [`jit_bucket_finish`] contracts the buckets
/// with the E tables.
pub(crate) fn jit_bucket_group(
    tab_o: [&[Gf]; 2],
    pat: &[[u8; 64]; 4],
    eq_t: &[Gf],
    send_one: bool,
    bk: &mut SumBuckets,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    neon::jit_bucket_group(tab_o, pat, eq_t, send_one, &mut bk.0);
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    generic::jit_bucket_group(tab_o, pat, eq_t, send_one, &mut bk.0);
}

/// `Σ_a T_E_end[a]·end[a]` and `Σ_a T_E_lo[a]·inf_lo[a] + Σ_a T_E_hi[a]·inf_hi[a]`,
/// each bucket reduced once — the row's `(Σ end, Σ inf)`.
pub(crate) fn jit_bucket_finish(tab_e: [&[Gf]; 2], send_one: bool, bk: &SumBuckets) -> (Gf, Gf) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        neon::jit_bucket_finish(tab_e, send_one, &bk.0)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        generic::jit_bucket_finish(tab_e, send_one, &bk.0)
    }
}

/// One group of the second just-in-time dense round: corner
/// `i = p·4 + b1·2 + b2` (`p` the half, `b1` the bit folded with `rho`,
/// `b2` this round's bit); writes the folded values
/// `E'(b2) = E(b1=0, b2) + rho·(E(1, b2) − E(0, b2))` to `out_l[b2]` (and
/// `O'` to `out_r`, both `out_l[b].len()` columns wide) and accumulates
/// the round sums over the folded pairs. With `PRE_SCALED`, the tables
/// already include the `1 + rho` and `rho` weights, so a fold is an addition.
#[allow(clippy::too_many_arguments)]
pub(crate) fn jit_fold_group<const PRE_SCALED: bool>(
    tab: [&[Gf]; 8],
    pat: &[[u8; 64]; 8],
    rho: &Gf,
    eq_t: &[Gf],
    send_one: bool,
    out_l: [&mut [MaybeUninit<Gf>]; 2],
    out_r: [&mut [MaybeUninit<Gf>]; 2],
    sums: &mut Sums,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    neon::jit_fold_group::<PRE_SCALED>(tab, pat, rho, eq_t, send_one, out_l, out_r, &mut sums.0);
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    generic::jit_fold_group::<PRE_SCALED>(tab, pat, rho, eq_t, send_one, out_l, out_r, &mut sums.0);
}

/// One group after a bivariate pivotal round. The sixteen tables are ordered
/// by `(half, b1, b2, b3)`; their `b1,b2` weights have already been absorbed.
/// XOR the four lookups for each `(half,b3)`, write the ordinary round's four
/// corners, and accumulate that round's Gruen message.
#[allow(clippy::too_many_arguments)]
pub(crate) fn jit_fold2_group(
    tab: [&[Gf]; 16],
    pat: &[[u8; 64]; 16],
    eq_t: &[Gf],
    send_one: bool,
    out_l: [&mut [MaybeUninit<Gf>]; 2],
    out_r: [&mut [MaybeUninit<Gf>]; 2],
    sums: &mut Sums,
) {
    assert_eq!(eq_t.len(), 64);
    let [out_l0, out_l1] = out_l;
    let [out_r0, out_r1] = out_r;
    let width = out_l0.len();
    assert!(out_l1.len() == width && out_r0.len() == width && out_r1.len() == width);
    for m in 0..64 {
        let c = col_of(m);
        if c >= width {
            continue;
        }
        let v = |i: usize| tab[i][pat[i][m] as usize];
        let l0 = v(0b0000) + v(0b0010) + v(0b0100) + v(0b0110);
        let l1 = v(0b0001) + v(0b0011) + v(0b0101) + v(0b0111);
        let r0 = v(0b1000) + v(0b1010) + v(0b1100) + v(0b1110);
        let r1 = v(0b1001) + v(0b1011) + v(0b1101) + v(0b1111);
        out_l0[c].write(l0);
        out_l1[c].write(l1);
        out_r0[c].write(r0);
        out_r1[c].write(r1);
        sums.slot(eq_t[m], l0, l1, r0, r1, send_one);
    }
}

/// Multiply every value by the same scalar.
pub(crate) fn scale_in_place(values: &mut [Gf], scalar: &Gf) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    neon::scale_in_place(values, scalar);
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    generic::scale_in_place(values, scalar);
}

/// One group of a product level: `out[c] = tab[0][pat[0][m]] · tab[1][pat[1][m]]`
/// over the `out.len()` columns.
pub(crate) fn jit_product_group(tab: [&[Gf]; 2], pat: &[[u8; 64]; 2], out: &mut [MaybeUninit<Gf>]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    neon::jit_product_group(tab, pat, out);
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    generic::jit_product_group(tab, pat, out);
}

/// `out[i] = a[i] · b[i]` over `out.len()` entries — one row of a product
/// level from the two rows below it.
pub(crate) fn product_into(a: &[Gf], b: &[Gf], out: &mut [MaybeUninit<Gf>]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    neon::product_into(a, b, out);
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    generic::product_into(a, b, out);
}

/// `bucket[idx[m]] += eq_t[m]` over a transposed block — the bit rounds'
/// one addition per term. `bucket` must hold 256 entries so every byte
/// index is in bounds; `eq_t` has 64 entries.
pub(crate) fn scatter_add(bucket: &mut [Gf], idx: &[u8; 64], eq_t: &[Gf]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    neon::scatter_add(bucket, idx, eq_t);
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    generic::scatter_add(bucket, idx, eq_t);
}

/// `Σ_k a[k] · b[k]`, accumulated unreduced and reduced once.
pub(crate) fn dot(a: &[Gf], b: &[Gf]) -> Gf {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        neon::dot(a, b)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        generic::dot(a, b)
    }
}

/// `Σ_a t_e[a] · Σ_b t_o[b] · bucket[a·n + b]` over `n = t_e.len()`
/// entries, the inner sums accumulated unreduced and reduced once each.
pub(crate) fn contract(t_e: &[Gf], t_o: &[Gf], bucket: &[Gf]) -> Gf {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        neon::contract(t_e, t_o, bucket)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        generic::contract(t_e, t_o, bucket)
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn round_sums_task(
    lo_l: &[Gf],
    hi_l: &[Gf],
    lo_r: &[Gf],
    hi_r: &[Gf],
    w: &[Gf],
    send_one: bool,
) -> (Gf, Gf) {
    neon::round_sums_task(lo_l, hi_l, lo_r, hi_r, w, send_one)
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
fn round_sums_task(
    lo_l: &[Gf],
    hi_l: &[Gf],
    lo_r: &[Gf],
    hi_r: &[Gf],
    w: &[Gf],
    send_one: bool,
) -> (Gf, Gf) {
    generic::round_sums_task(lo_l, hi_l, lo_r, hi_r, w, send_one)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[allow(clippy::too_many_arguments)]
fn fused_task(
    q0_l: &mut [Gf],
    q1_l: &mut [Gf],
    q2_l: &[Gf],
    q3_l: &[Gf],
    q0_r: &mut [Gf],
    q1_r: &mut [Gf],
    q2_r: &[Gf],
    q3_r: &[Gf],
    rho: &Gf,
    w: &[Gf],
    send_one: bool,
) -> (Gf, Gf) {
    neon::fused_task(q0_l, q1_l, q2_l, q3_l, q0_r, q1_r, q2_r, q3_r, rho, w, send_one)
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[allow(clippy::too_many_arguments)]
fn fused_task(
    q0_l: &mut [Gf],
    q1_l: &mut [Gf],
    q2_l: &[Gf],
    q3_l: &[Gf],
    q0_r: &mut [Gf],
    q1_r: &mut [Gf],
    q2_r: &[Gf],
    q3_r: &[Gf],
    rho: &Gf,
    w: &[Gf],
    send_one: bool,
) -> (Gf, Gf) {
    generic::fused_task(q0_l, q1_l, q2_l, q3_l, q0_r, q1_r, q2_r, q3_r, rho, w, send_one)
}

/// The portable kernels: the field's own operators and the delayed-
/// reduction accumulator. Also the reference the NEON kernels are tested
/// against.
#[allow(dead_code)]
pub(crate) mod generic {
    use std::mem::MaybeUninit;

    use super::Gf;
    use super::WideMulAcc;

    type Wide = <Gf as WideMulAcc>::Wide;

    pub(crate) struct Sums {
        end: Wide,
        inf: Wide,
    }

    impl Sums {
        pub(crate) fn zero() -> Self {
            let zero = Gf::zero();
            Self {
                end: <Gf as WideMulAcc>::wide_zero(&zero),
                inf: <Gf as WideMulAcc>::wide_zero(&zero),
            }
        }

        pub(crate) fn finish(self) -> (Gf, Gf) {
            (
                <Gf as WideMulAcc>::from_wide(self.end),
                <Gf as WideMulAcc>::from_wide(self.inf),
            )
        }

        /// `end += (w·l_end)·r_end`, `inf += (w·(l1 − l0))·(r1 − r0)`.
        #[inline(always)]
        pub(super) fn slot(&mut self, w: Gf, l0: Gf, l1: Gf, r0: Gf, r1: Gf, send_one: bool) {
            let (le, re) = if send_one { (l1, r1) } else { (l0, r0) };
            let el = w * le;
            <Gf as WideMulAcc>::wide_add_assign(&mut self.end, &<Gf as WideMulAcc>::mul_wide(&el, &re));
            let ed = w * (l1 - l0);
            <Gf as WideMulAcc>::wide_add_assign(
                &mut self.inf,
                &<Gf as WideMulAcc>::mul_wide(&ed, &(r1 - r0)),
            );
        }
    }

    #[inline(always)]
    fn fold(rho: Gf, v0: Gf, v1: Gf) -> Gf {
        v0 + rho * (v1 - v0)
    }

    pub(crate) fn round_sums_task(
        lo_l: &[Gf],
        hi_l: &[Gf],
        lo_r: &[Gf],
        hi_r: &[Gf],
        w: &[Gf],
        send_one: bool,
    ) -> (Gf, Gf) {
        let mut sums = Sums::zero();
        for c in 0..w.len() {
            sums.slot(w[c], lo_l[c], hi_l[c], lo_r[c], hi_r[c], send_one);
        }
        sums.finish()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fused_task(
        q0_l: &mut [Gf],
        q1_l: &mut [Gf],
        q2_l: &[Gf],
        q3_l: &[Gf],
        q0_r: &mut [Gf],
        q1_r: &mut [Gf],
        q2_r: &[Gf],
        q3_r: &[Gf],
        rho: &Gf,
        w: &[Gf],
        send_one: bool,
    ) -> (Gf, Gf) {
        let mut sums = Sums::zero();
        for c in 0..w.len() {
            let fl0 = fold(*rho, q0_l[c], q2_l[c]);
            let fl1 = fold(*rho, q1_l[c], q3_l[c]);
            let fr0 = fold(*rho, q0_r[c], q2_r[c]);
            let fr1 = fold(*rho, q1_r[c], q3_r[c]);
            q0_l[c] = fl0;
            q1_l[c] = fl1;
            q0_r[c] = fr0;
            q1_r[c] = fr1;
            sums.slot(w[c], fl0, fl1, fr0, fr1, send_one);
        }
        sums.finish()
    }

    #[allow(dead_code)]
    pub(crate) fn jit_sums_group(
        tab: [&[Gf]; 4],
        pat: &[[u8; 64]; 4],
        eq_t: &[Gf],
        send_one: bool,
        sums: &mut Sums,
    ) {
        assert_eq!(eq_t.len(), 64);
        for m in 0..64 {
            let l0 = tab[0][pat[0][m] as usize];
            let l1 = tab[1][pat[1][m] as usize];
            let r0 = tab[2][pat[2][m] as usize];
            let r1 = tab[3][pat[3][m] as usize];
            sums.slot(eq_t[m], l0, l1, r0, r1, send_one);
        }
    }

    pub(crate) struct SumBuckets {
        end: Vec<Wide>,
        inf_lo: Vec<Wide>,
        inf_hi: Vec<Wide>,
    }

    impl SumBuckets {
        pub(crate) fn new() -> Self {
            let zero = Gf::zero();
            let fresh = || vec![<Gf as WideMulAcc>::wide_zero(&zero); 256];
            Self {
                end: fresh(),
                inf_lo: fresh(),
                inf_hi: fresh(),
            }
        }

        pub(crate) fn clear(&mut self) {
            let zero = <Gf as WideMulAcc>::wide_zero(&Gf::zero());
            for b in [&mut self.end, &mut self.inf_lo, &mut self.inf_hi] {
                b.fill(zero.clone());
            }
        }
    }

    pub(crate) fn jit_bucket_group(
        tab_o: [&[Gf]; 2],
        pat: &[[u8; 64]; 4],
        eq_t: &[Gf],
        send_one: bool,
        bk: &mut SumBuckets,
    ) {
        assert_eq!(eq_t.len(), 64);
        for m in 0..64 {
            let (a_lo, a_hi) = (pat[0][m] as usize, pat[1][m] as usize);
            let o_lo = tab_o[0][pat[2][m] as usize];
            let o_hi = tab_o[1][pat[3][m] as usize];
            let w = eq_t[m];
            let (a_end, o_end) = if send_one { (a_hi, o_hi) } else { (a_lo, o_lo) };
            <Gf as WideMulAcc>::wide_add_assign(&mut bk.end[a_end], &<Gf as WideMulAcc>::mul_wide(&w, &o_end));
            let d = <Gf as WideMulAcc>::mul_wide(&w, &(o_hi - o_lo));
            <Gf as WideMulAcc>::wide_add_assign(&mut bk.inf_lo[a_lo], &d);
            <Gf as WideMulAcc>::wide_add_assign(&mut bk.inf_hi[a_hi], &d);
        }
    }

    pub(crate) fn jit_bucket_finish(tab_e: [&[Gf]; 2], send_one: bool, bk: &SumBuckets) -> (Gf, Gf) {
        let zero = Gf::zero();
        let contract = |t: &[Gf], b: &[Wide]| -> Wide {
            let mut acc = <Gf as WideMulAcc>::wide_zero(&zero);
            for a in 0..256 {
                let v = <Gf as WideMulAcc>::from_wide(b[a].clone());
                <Gf as WideMulAcc>::wide_add_assign(&mut acc, &<Gf as WideMulAcc>::mul_wide(&t[a], &v));
            }
            acc
        };
        let t_end = if send_one { tab_e[1] } else { tab_e[0] };
        let end = contract(t_end, &bk.end);
        let mut inf = contract(tab_e[0], &bk.inf_lo);
        <Gf as WideMulAcc>::wide_add_assign(&mut inf, &contract(tab_e[1], &bk.inf_hi));
        (<Gf as WideMulAcc>::from_wide(end), <Gf as WideMulAcc>::from_wide(inf))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn jit_fold_group<const PRE_SCALED: bool>(
        tab: [&[Gf]; 8],
        pat: &[[u8; 64]; 8],
        rho: &Gf,
        eq_t: &[Gf],
        send_one: bool,
        out_l: [&mut [MaybeUninit<Gf>]; 2],
        out_r: [&mut [MaybeUninit<Gf>]; 2],
        sums: &mut Sums,
    ) {
        assert_eq!(eq_t.len(), 64);
        let [out_l0, out_l1] = out_l;
        let [out_r0, out_r1] = out_r;
        let width = out_l0.len();
        let combine = |v0, v1| if PRE_SCALED { v0 + v1 } else { fold(*rho, v0, v1) };
        for m in 0..64 {
            let c = super::col_of(m);
            if c >= width {
                continue;
            }
            let v = |i: usize| tab[i][pat[i][m] as usize];
            let fl0 = combine(v(0b000), v(0b010));
            let fl1 = combine(v(0b001), v(0b011));
            let fr0 = combine(v(0b100), v(0b110));
            let fr1 = combine(v(0b101), v(0b111));
            out_l0[c].write(fl0);
            out_l1[c].write(fl1);
            out_r0[c].write(fr0);
            out_r1[c].write(fr1);
            sums.slot(eq_t[m], fl0, fl1, fr0, fr1, send_one);
        }
    }

    pub(crate) fn scale_in_place(values: &mut [Gf], scalar: &Gf) {
        for value in values {
            *value = *value * *scalar;
        }
    }

    pub(crate) fn jit_product_group(tab: [&[Gf]; 2], pat: &[[u8; 64]; 2], out: &mut [MaybeUninit<Gf>]) {
        let width = out.len();
        for m in 0..64 {
            let c = super::col_of(m);
            if c < width {
                out[c].write(tab[0][pat[0][m] as usize] * tab[1][pat[1][m] as usize]);
            }
        }
    }

    pub(crate) fn product_into(a: &[Gf], b: &[Gf], out: &mut [MaybeUninit<Gf>]) {
        let n = out.len();
        assert!(a.len() >= n && b.len() >= n);
        for i in 0..n {
            out[i].write(a[i] * b[i]);
        }
    }

    pub(crate) fn scatter_add(bucket: &mut [Gf], idx: &[u8; 64], eq_t: &[Gf]) {
        assert!(bucket.len() >= 256);
        assert_eq!(eq_t.len(), 64);
        for (&i, &e) in idx.iter().zip(eq_t) {
            bucket[i as usize] += e;
        }
    }

    pub(crate) fn dot(a: &[Gf], b: &[Gf]) -> Gf {
        assert_eq!(a.len(), b.len());
        let mut acc = <Gf as WideMulAcc>::wide_zero(&Gf::zero());
        for (x, y) in a.iter().zip(b) {
            <Gf as WideMulAcc>::wide_add_assign(&mut acc, &<Gf as WideMulAcc>::mul_wide(x, y));
        }
        <Gf as WideMulAcc>::from_wide(acc)
    }

    pub(crate) fn contract(t_e: &[Gf], t_o: &[Gf], bucket: &[Gf]) -> Gf {
        let n = t_e.len();
        assert_eq!(t_o.len(), n);
        assert!(bucket.len() >= n * n);
        let zero = Gf::zero();
        let mut acc = <Gf as WideMulAcc>::wide_zero(&zero);
        for a in 0..n {
            let mut inner = <Gf as WideMulAcc>::wide_zero(&zero);
            for b in 0..n {
                <Gf as WideMulAcc>::wide_add_assign(
                    &mut inner,
                    &<Gf as WideMulAcc>::mul_wide(&t_o[b], &bucket[a * n + b]),
                );
            }
            let inner = <Gf as WideMulAcc>::from_wide(inner);
            <Gf as WideMulAcc>::wide_add_assign(&mut acc, &<Gf as WideMulAcc>::mul_wide(&t_e[a], &inner));
        }
        <Gf as WideMulAcc>::from_wide(acc)
    }
}

/// The aarch64 kernels. Every loop keeps its accumulators in vector
/// registers, folds with the pass-fixed multiplier preprocessed (5 PMULLs
/// per product instead of 7) and, where the body allows, alternates two
/// independent accumulator sets so the PMULL pipes stay fed.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) mod neon {
    use core::arch::aarch64::{uint64x2_t, vdupq_n_u64, veorq_u64, vextq_u64, vld1q_u64, vst1q_u64};
    use std::mem::MaybeUninit;

    use super::Gf;
    use field::gf128::neon::{
        clmul_256, fold_x64, ld, mul_fixed_wide, pmull_hi, pmull_lo, prep_fixed, reduce_256,
    };

    /// An unreduced 256-bit accumulator.
    type Acc = (uint64x2_t, uint64x2_t);

    #[inline(always)]
    unsafe fn acc_zero() -> Acc {
        // SAFETY: a plain NEON constant.
        unsafe {
            let z = vdupq_n_u64(0);
            (z, z)
        }
    }

    #[inline(always)]
    unsafe fn acc_add(a: &mut Acc, p: (uint64x2_t, uint64x2_t)) {
        // SAFETY: plain NEON XORs.
        unsafe {
            a.0 = veorq_u64(a.0, p.0);
            a.1 = veorq_u64(a.1, p.1);
        }
    }

    #[inline(always)]
    unsafe fn to_elt(a: Acc) -> Gf {
        // SAFETY: as `neon::pmull_lo`; `out` is a valid 16-byte word pair.
        unsafe {
            let r = reduce_256(a.0, a.1);
            let mut out = [0u64; 2];
            vst1q_u64(out.as_mut_ptr(), r);
            Gf::from_polynomial_words(out)
        }
    }

    #[inline(always)]
    unsafe fn st(x: &mut Gf, v: uint64x2_t) {
        // SAFETY: `out` is a valid 16-byte word pair.
        unsafe {
            let mut out = [0u64; 2];
            vst1q_u64(out.as_mut_ptr(), v);
            *x = Gf::from_polynomial_words(out);
        }
    }

    #[inline(always)]
    unsafe fn st_uninit(x: &mut MaybeUninit<Gf>, v: uint64x2_t) {
        // SAFETY: `out` is a valid 16-byte word pair.
        unsafe {
            let mut out = [0u64; 2];
            vst1q_u64(out.as_mut_ptr(), v);
            x.write(Gf::from_polynomial_words(out));
        }
    }

    /// Reduced multiply on register-resident vectors: 4-PMULL schoolbook
    /// to the raw 3-limb product, two `fold_x64` stages (6 PMULLs).
    #[inline(always)]
    unsafe fn mul_red(a: uint64x2_t, b: uint64x2_t, g: uint64x2_t, z: uint64x2_t) -> uint64x2_t {
        // SAFETY: as `neon::pmull_lo`.
        unsafe {
            let t00 = pmull_lo(a, b);
            let t11 = pmull_hi(a, b);
            let bsw = vextq_u64(b, b, 1);
            let mid = veorq_u64(pmull_lo(a, bsw), pmull_hi(a, bsw));
            let t1 = fold_x64(mid, t11, g, z);
            fold_x64(t00, t1, g, z)
        }
    }

    /// Reduced multiply by the preprocessed pass-fixed scalar `(rl, rh)`.
    #[inline(always)]
    unsafe fn mul_fixed(
        av: uint64x2_t,
        rl: uint64x2_t,
        rh: uint64x2_t,
        g: uint64x2_t,
        z: uint64x2_t,
    ) -> uint64x2_t {
        // SAFETY: as `neon::pmull_lo`.
        unsafe {
            let (tl, tm) = mul_fixed_wide(av, rl, rh);
            fold_x64(tl, tm, g, z)
        }
    }

    /// `v0 + rho·(v1 − v0)` with `rho` preprocessed.
    #[inline(always)]
    unsafe fn fold1(
        rl: uint64x2_t,
        rh: uint64x2_t,
        g: uint64x2_t,
        z: uint64x2_t,
        v0: uint64x2_t,
        v1: uint64x2_t,
    ) -> uint64x2_t {
        // SAFETY: as `neon::pmull_lo`.
        unsafe { veorq_u64(v0, mul_fixed(veorq_u64(v1, v0), rl, rh, g, z)) }
    }

    /// The Gruen slot: `end += (w·l_end)⊗r_end`, `inf += (w·(l1⊕l0))⊗(r1⊕r0)`
    /// with the outer products left unreduced.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn slot(
        w: uint64x2_t,
        l0: uint64x2_t,
        l1: uint64x2_t,
        r0: uint64x2_t,
        r1: uint64x2_t,
        send_one: bool,
        g: uint64x2_t,
        z: uint64x2_t,
        end: &mut Acc,
        inf: &mut Acc,
    ) {
        // SAFETY: as `neon::pmull_lo`.
        unsafe {
            let (le, re) = if send_one { (l1, r1) } else { (l0, r0) };
            let el = mul_red(w, le, g, z);
            acc_add(end, clmul_256(el, re));
            let ed = mul_red(w, veorq_u64(l1, l0), g, z);
            acc_add(inf, clmul_256(ed, veorq_u64(r1, r0)));
        }
    }

    pub(crate) struct Sums {
        end: Acc,
        inf: Acc,
    }

    impl Sums {
        pub(crate) fn zero() -> Self {
            // SAFETY: plain NEON constants.
            unsafe {
                Self {
                    end: acc_zero(),
                    inf: acc_zero(),
                }
            }
        }

        pub(crate) fn finish(self) -> (Gf, Gf) {
            // SAFETY: as `neon::pmull_lo`.
            unsafe { (to_elt(self.end), to_elt(self.inf)) }
        }

        #[inline(always)]
        pub(super) fn slot(&mut self, w: Gf, l0: Gf, l1: Gf, r0: Gf, r1: Gf, send_one: bool) {
            // SAFETY: all values are ordinary field elements loaded into
            // vectors; the accumulators are owned by `self`.
            unsafe {
                let g = vdupq_n_u64(0x87);
                let z = vdupq_n_u64(0);
                slot(
                    ld(&w), ld(&l0), ld(&l1), ld(&r0), ld(&r1), send_one,
                    g, z, &mut self.end, &mut self.inf,
                );
            }
        }
    }

    pub(crate) fn round_sums_task(
        lo_l: &[Gf],
        hi_l: &[Gf],
        lo_r: &[Gf],
        hi_r: &[Gf],
        w: &[Gf],
        send_one: bool,
    ) -> (Gf, Gf) {
        let n = w.len();
        assert!(lo_l.len() >= n && hi_l.len() >= n && lo_r.len() >= n && hi_r.len() >= n);
        // SAFETY: as `neon::pmull_lo`; every index is below `n`, which
        // the assertion bounds by each slice's length.
        unsafe {
            let g = vdupq_n_u64(0x87);
            let z = vdupq_n_u64(0);
            let (mut ea, mut ia) = (acc_zero(), acc_zero());
            let (mut eb, mut ib) = (acc_zero(), acc_zero());
            let mut c = 0usize;
            while c + 2 <= n {
                slot(
                    ld(w.get_unchecked(c)),
                    ld(lo_l.get_unchecked(c)),
                    ld(hi_l.get_unchecked(c)),
                    ld(lo_r.get_unchecked(c)),
                    ld(hi_r.get_unchecked(c)),
                    send_one,
                    g,
                    z,
                    &mut ea,
                    &mut ia,
                );
                let d = c + 1;
                slot(
                    ld(w.get_unchecked(d)),
                    ld(lo_l.get_unchecked(d)),
                    ld(hi_l.get_unchecked(d)),
                    ld(lo_r.get_unchecked(d)),
                    ld(hi_r.get_unchecked(d)),
                    send_one,
                    g,
                    z,
                    &mut eb,
                    &mut ib,
                );
                c += 2;
            }
            if c < n {
                slot(
                    ld(w.get_unchecked(c)),
                    ld(lo_l.get_unchecked(c)),
                    ld(hi_l.get_unchecked(c)),
                    ld(lo_r.get_unchecked(c)),
                    ld(hi_r.get_unchecked(c)),
                    send_one,
                    g,
                    z,
                    &mut ea,
                    &mut ia,
                );
            }
            acc_add(&mut ea, eb);
            acc_add(&mut ia, ib);
            (to_elt(ea), to_elt(ia))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fused_task(
        q0_l: &mut [Gf],
        q1_l: &mut [Gf],
        q2_l: &[Gf],
        q3_l: &[Gf],
        q0_r: &mut [Gf],
        q1_r: &mut [Gf],
        q2_r: &[Gf],
        q3_r: &[Gf],
        rho: &Gf,
        w: &[Gf],
        send_one: bool,
    ) -> (Gf, Gf) {
        let n = w.len();
        assert!(
            q0_l.len() >= n
                && q1_l.len() >= n
                && q2_l.len() >= n
                && q3_l.len() >= n
                && q0_r.len() >= n
                && q1_r.len() >= n
                && q2_r.len() >= n
                && q3_r.len() >= n
        );
        // SAFETY: as `neon::pmull_lo`; every index is below `n`, bounded
        // by the assertion. Each column's four quarter entries are loaded
        // before its two folded values are stored, and the stores only
        // ever target `q0`/`q1`, which no other column reads.
        unsafe {
            let (rl, rh) = prep_fixed(rho);
            let g = vdupq_n_u64(0x87);
            let z = vdupq_n_u64(0);
            let (mut ea, mut ia) = (acc_zero(), acc_zero());
            let (mut eb, mut ib) = (acc_zero(), acc_zero());
            macro_rules! column {
                ($c:expr, $end:expr, $inf:expr) => {{
                    let c = $c;
                    let fl0 = fold1(rl, rh, g, z, ld(q0_l.get_unchecked(c)), ld(q2_l.get_unchecked(c)));
                    let fl1 = fold1(rl, rh, g, z, ld(q1_l.get_unchecked(c)), ld(q3_l.get_unchecked(c)));
                    let fr0 = fold1(rl, rh, g, z, ld(q0_r.get_unchecked(c)), ld(q2_r.get_unchecked(c)));
                    let fr1 = fold1(rl, rh, g, z, ld(q1_r.get_unchecked(c)), ld(q3_r.get_unchecked(c)));
                    st(q0_l.get_unchecked_mut(c), fl0);
                    st(q1_l.get_unchecked_mut(c), fl1);
                    st(q0_r.get_unchecked_mut(c), fr0);
                    st(q1_r.get_unchecked_mut(c), fr1);
                    slot(ld(w.get_unchecked(c)), fl0, fl1, fr0, fr1, send_one, g, z, $end, $inf);
                }};
            }
            let mut c = 0usize;
            while c + 2 <= n {
                column!(c, &mut ea, &mut ia);
                column!(c + 1, &mut eb, &mut ib);
                c += 2;
            }
            if c < n {
                column!(c, &mut ea, &mut ia);
            }
            acc_add(&mut ea, eb);
            acc_add(&mut ia, ib);
            (to_elt(ea), to_elt(ia))
        }
    }

    #[allow(dead_code)]
    pub(crate) fn jit_sums_group(
        tab: [&[Gf]; 4],
        pat: &[[u8; 64]; 4],
        eq_t: &[Gf],
        send_one: bool,
        sums: &mut Sums,
    ) {
        assert_eq!(eq_t.len(), 64);
        assert!(tab.iter().all(|t| t.len() >= 256));
        // SAFETY: as `neon::pmull_lo`; positions are below 64 and the
        // pattern bytes index tables of at least 256 entries.
        unsafe {
            let g = vdupq_n_u64(0x87);
            let z = vdupq_n_u64(0);
            let (mut eb, mut ib) = (acc_zero(), acc_zero());
            macro_rules! column {
                ($m:expr, $end:expr, $inf:expr) => {{
                    let m = $m;
                    let l0 = ld(tab[0].get_unchecked(pat[0][m] as usize));
                    let l1 = ld(tab[1].get_unchecked(pat[1][m] as usize));
                    let r0 = ld(tab[2].get_unchecked(pat[2][m] as usize));
                    let r1 = ld(tab[3].get_unchecked(pat[3][m] as usize));
                    slot(ld(eq_t.get_unchecked(m)), l0, l1, r0, r1, send_one, g, z, $end, $inf);
                }};
            }
            let mut m = 0usize;
            while m < 64 {
                column!(m, &mut sums.end, &mut sums.inf);
                column!(m + 1, &mut eb, &mut ib);
                m += 2;
            }
            acc_add(&mut sums.end, eb);
            acc_add(&mut sums.inf, ib);
        }
    }

    /// Three 256-entry tables of unreduced `(lo, hi)` accumulators, flat
    /// `[u64; 4]` words: `end`, then `inf_lo`, then `inf_hi`.
    pub(crate) struct SumBuckets {
        data: Vec<[u64; 4]>,
    }

    impl SumBuckets {
        pub(crate) fn new() -> Self {
            Self {
                data: vec![[0u64; 4]; 3 * 256],
            }
        }

        pub(crate) fn clear(&mut self) {
            self.data.fill([0u64; 4]);
        }
    }

    /// `bucket[idx] ^= (lo, hi)` on the flat word array.
    #[inline(always)]
    unsafe fn bucket_xor(base: *mut u64, idx: usize, lo: uint64x2_t, hi: uint64x2_t) {
        // SAFETY: the caller keeps `idx` below the table's 256 entries.
        unsafe {
            let p = base.add(4 * idx);
            vst1q_u64(p, veorq_u64(vld1q_u64(p), lo));
            vst1q_u64(p.add(2), veorq_u64(vld1q_u64(p.add(2)), hi));
        }
    }

    pub(crate) fn jit_bucket_group(
        tab_o: [&[Gf]; 2],
        pat: &[[u8; 64]; 4],
        eq_t: &[Gf],
        send_one: bool,
        bk: &mut SumBuckets,
    ) {
        assert_eq!(eq_t.len(), 64);
        assert!(tab_o[0].len() >= 256 && tab_o[1].len() >= 256);
        assert_eq!(bk.data.len(), 3 * 256);
        // SAFETY: as `neon::pmull_lo`; every byte index lands inside a
        // 256-entry table and positions are below 64.
        unsafe {
            let base = bk.data.as_mut_ptr().cast::<u64>();
            let end = base;
            let inf_lo = base.add(4 * 256);
            let inf_hi = base.add(8 * 256);
            for m in 0..64 {
                let (a_lo, a_hi) = (pat[0][m] as usize, pat[1][m] as usize);
                let o_lo = ld(tab_o[0].get_unchecked(pat[2][m] as usize));
                let o_hi = ld(tab_o[1].get_unchecked(pat[3][m] as usize));
                let w = ld(eq_t.get_unchecked(m));
                let (a_end, o_end) = if send_one { (a_hi, o_hi) } else { (a_lo, o_lo) };
                let (pl, ph) = clmul_256(w, o_end);
                bucket_xor(end, a_end, pl, ph);
                let (dl, dh) = clmul_256(w, veorq_u64(o_hi, o_lo));
                bucket_xor(inf_lo, a_lo, dl, dh);
                bucket_xor(inf_hi, a_hi, dl, dh);
            }
        }
    }

    pub(crate) fn jit_bucket_finish(tab_e: [&[Gf]; 2], send_one: bool, bk: &SumBuckets) -> (Gf, Gf) {
        assert!(tab_e[0].len() >= 256 && tab_e[1].len() >= 256);
        assert_eq!(bk.data.len(), 3 * 256);
        // SAFETY: as `neon::pmull_lo`; indices below 256.
        unsafe {
            let contract = |t: &[Gf], table: &[[u64; 4]]| -> Acc {
                let mut acc_a = acc_zero();
                let mut acc_b = acc_zero();
                let mut a = 0usize;
                while a < 256 {
                    let e0 = table.get_unchecked(a);
                    let v0 = reduce_256(vld1q_u64(e0.as_ptr()), vld1q_u64(e0.as_ptr().add(2)));
                    acc_add(&mut acc_a, clmul_256(ld(t.get_unchecked(a)), v0));
                    let e1 = table.get_unchecked(a + 1);
                    let v1 = reduce_256(vld1q_u64(e1.as_ptr()), vld1q_u64(e1.as_ptr().add(2)));
                    acc_add(&mut acc_b, clmul_256(ld(t.get_unchecked(a + 1)), v1));
                    a += 2;
                }
                acc_add(&mut acc_a, acc_b);
                acc_a
            };
            let t_end = if send_one { tab_e[1] } else { tab_e[0] };
            let end = contract(t_end, &bk.data[..256]);
            let mut inf = contract(tab_e[0], &bk.data[256..512]);
            acc_add(&mut inf, contract(tab_e[1], &bk.data[512..768]));
            (to_elt(end), to_elt(inf))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn jit_fold_group<const PRE_SCALED: bool>(
        tab: [&[Gf]; 8],
        pat: &[[u8; 64]; 8],
        rho: &Gf,
        eq_t: &[Gf],
        send_one: bool,
        out_l: [&mut [MaybeUninit<Gf>]; 2],
        out_r: [&mut [MaybeUninit<Gf>]; 2],
        sums: &mut Sums,
    ) {
        assert_eq!(eq_t.len(), 64);
        assert!(tab.iter().all(|t| t.len() >= 256));
        let [out_l0, out_l1] = out_l;
        let [out_r0, out_r1] = out_r;
        let width = out_l0.len();
        assert!(width <= 64 && out_l1.len() == width && out_r0.len() == width && out_r1.len() == width);
        // SAFETY: as `jit_sums_group`; a store's column is checked
        // against `width` (never taken for a full group).
        unsafe {
            let g = vdupq_n_u64(0x87);
            let z = vdupq_n_u64(0);
            let (rl, rh) = if PRE_SCALED { (z, z) } else { prep_fixed(rho) };
            let combine = |v0, v1| {
                if PRE_SCALED { veorq_u64(v0, v1) } else { fold1(rl, rh, g, z, v0, v1) }
            };
            let (mut eb, mut ib) = (acc_zero(), acc_zero());
            macro_rules! column {
                ($m:expr, $end:expr, $inf:expr) => {{
                    let m = $m;
                    let c = super::col_of(m);
                    if c < width {
                        let v = |i: usize| ld(tab[i].get_unchecked(pat[i][m] as usize));
                        let fl0 = combine(v(0b000), v(0b010));
                        let fl1 = combine(v(0b001), v(0b011));
                        let fr0 = combine(v(0b100), v(0b110));
                        let fr1 = combine(v(0b101), v(0b111));
                        st_uninit(out_l0.get_unchecked_mut(c), fl0);
                        st_uninit(out_l1.get_unchecked_mut(c), fl1);
                        st_uninit(out_r0.get_unchecked_mut(c), fr0);
                        st_uninit(out_r1.get_unchecked_mut(c), fr1);
                        slot(ld(eq_t.get_unchecked(m)), fl0, fl1, fr0, fr1, send_one, g, z, $end, $inf);
                    }
                }};
            }
            let mut m = 0usize;
            while m < 64 {
                column!(m, &mut sums.end, &mut sums.inf);
                column!(m + 1, &mut eb, &mut ib);
                m += 2;
            }
            acc_add(&mut sums.end, eb);
            acc_add(&mut sums.inf, ib);
        }
    }

    pub(crate) fn scale_in_place(values: &mut [Gf], scalar: &Gf) {
        // SAFETY: as `neon::pmull_lo`; every index is bounded by `values`.
        unsafe {
            let (rl, rh) = prep_fixed(scalar);
            let g = vdupq_n_u64(0x87);
            let z = vdupq_n_u64(0);
            let mut i = 0usize;
            while i + 2 <= values.len() {
                let p0 = mul_fixed(ld(values.get_unchecked(i)), rl, rh, g, z);
                let p1 = mul_fixed(ld(values.get_unchecked(i + 1)), rl, rh, g, z);
                st(values.get_unchecked_mut(i), p0);
                st(values.get_unchecked_mut(i + 1), p1);
                i += 2;
            }
            if i < values.len() {
                let p = mul_fixed(ld(values.get_unchecked(i)), rl, rh, g, z);
                st(values.get_unchecked_mut(i), p);
            }
        }
    }

    pub(crate) fn jit_product_group(tab: [&[Gf]; 2], pat: &[[u8; 64]; 2], out: &mut [MaybeUninit<Gf>]) {
        let width = out.len();
        assert!(width <= 64);
        assert!(tab[0].len() >= 256 && tab[1].len() >= 256);
        // SAFETY: as `jit_sums_group`; stores are checked against `width`.
        unsafe {
            let g = vdupq_n_u64(0x87);
            let z = vdupq_n_u64(0);
            for m in 0..64 {
                let c = super::col_of(m);
                if c < width {
                    let a = ld(tab[0].get_unchecked(pat[0][m] as usize));
                    let b = ld(tab[1].get_unchecked(pat[1][m] as usize));
                    st_uninit(out.get_unchecked_mut(c), mul_red(a, b, g, z));
                }
            }
        }
    }

    pub(crate) fn product_into(a: &[Gf], b: &[Gf], out: &mut [MaybeUninit<Gf>]) {
        let n = out.len();
        assert!(a.len() >= n && b.len() >= n);
        // SAFETY: as `neon::pmull_lo`; every index is below `n`, bounded
        // by the assertion and `out`'s length.
        unsafe {
            let g = vdupq_n_u64(0x87);
            let z = vdupq_n_u64(0);
            let mut i = 0usize;
            while i + 2 <= n {
                let p0 = mul_red(ld(a.get_unchecked(i)), ld(b.get_unchecked(i)), g, z);
                let p1 = mul_red(ld(a.get_unchecked(i + 1)), ld(b.get_unchecked(i + 1)), g, z);
                st_uninit(out.get_unchecked_mut(i), p0);
                st_uninit(out.get_unchecked_mut(i + 1), p1);
                i += 2;
            }
            if i < n {
                let p = mul_red(ld(a.get_unchecked(i)), ld(b.get_unchecked(i)), g, z);
                st_uninit(out.get_unchecked_mut(i), p);
            }
        }
    }

    pub(crate) fn scatter_add(bucket: &mut [Gf], idx: &[u8; 64], eq_t: &[Gf]) {
        assert_eq!(eq_t.len(), 64);
        assert!(bucket.len() >= 256);
        // SAFETY: `Gf` is `repr(transparent)` over two `u64` limbs, so the
        // bucket is a valid `2·256`-word array; every byte index lands
        // inside it, and positions are below 64.
        unsafe {
            let base = bucket.as_mut_ptr().cast::<u64>();
            let mut m = 0usize;
            while m < 64 {
                let i0 = *idx.get_unchecked(m) as usize;
                let i1 = *idx.get_unchecked(m + 1) as usize;
                let p0 = base.add(2 * i0);
                let e0 = ld(eq_t.get_unchecked(m));
                vst1q_u64(p0, veorq_u64(vld1q_u64(p0), e0));
                let p1 = base.add(2 * i1);
                let e1 = ld(eq_t.get_unchecked(m + 1));
                vst1q_u64(p1, veorq_u64(vld1q_u64(p1), e1));
                m += 2;
            }
        }
    }

    pub(crate) fn dot(a: &[Gf], b: &[Gf]) -> Gf {
        let n = a.len();
        assert_eq!(b.len(), n);
        // SAFETY: as `neon::pmull_lo`; indices below `n`.
        unsafe {
            let mut ia = acc_zero();
            let mut ib = acc_zero();
            let mut k = 0usize;
            while k + 2 <= n {
                acc_add(&mut ia, clmul_256(ld(a.get_unchecked(k)), ld(b.get_unchecked(k))));
                acc_add(&mut ib, clmul_256(ld(a.get_unchecked(k + 1)), ld(b.get_unchecked(k + 1))));
                k += 2;
            }
            if k < n {
                acc_add(&mut ia, clmul_256(ld(a.get_unchecked(k)), ld(b.get_unchecked(k))));
            }
            acc_add(&mut ia, ib);
            to_elt(ia)
        }
    }

    pub(crate) fn contract(t_e: &[Gf], t_o: &[Gf], bucket: &[Gf]) -> Gf {
        let n = t_e.len();
        assert_eq!(t_o.len(), n);
        assert!(bucket.len() >= n * n);
        // SAFETY: as `neon::pmull_lo`; indices bounded by the assertions.
        unsafe {
            let mut acc = acc_zero();
            for a in 0..n {
                let row = bucket.get_unchecked(a * n..(a + 1) * n);
                let mut ia = acc_zero();
                let mut ib = acc_zero();
                let mut b = 0usize;
                while b + 2 <= n {
                    acc_add(&mut ia, clmul_256(ld(t_o.get_unchecked(b)), ld(row.get_unchecked(b))));
                    acc_add(
                        &mut ib,
                        clmul_256(ld(t_o.get_unchecked(b + 1)), ld(row.get_unchecked(b + 1))),
                    );
                    b += 2;
                }
                if b < n {
                    acc_add(&mut ia, clmul_256(ld(t_o.get_unchecked(b)), ld(row.get_unchecked(b))));
                }
                acc_add(&mut ia, ib);
                let inner = reduce_256(ia.0, ia.1);
                acc_add(&mut acc, clmul_256(ld(t_e.get_unchecked(a)), inner));
            }
            to_elt(acc)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads back buffers the kernels filled (every slot was written).
    fn init(v: &[Vec<MaybeUninit<Gf>>]) -> Vec<Vec<Gf>> {
        v.iter()
            .map(|row| row.iter().map(|x| unsafe { x.assume_init() }).collect())
            .collect()
    }

    fn elements(n: usize, seed: u64) -> Vec<Gf> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                let mut next = || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state
                };
                Gf::from_polynomial_words([next(), next()])
            })
            .collect()
    }

    #[test]
    fn round_sums_matches_the_reference() {
        for n in [1usize, 2, 3, 7, 64, 65, 200] {
            for send_one in [false, true] {
                let lo_l = elements(n, 11);
                let hi_l = elements(n, 12);
                let lo_r = elements(n, 13);
                let hi_r = elements(n, 14);
                let w = elements(n, 15);
                let got = round_sums_task(&lo_l, &hi_l, &lo_r, &hi_r, &w, send_one);
                let want = generic::round_sums_task(&lo_l, &hi_l, &lo_r, &hi_r, &w, send_one);
                assert_eq!(got, want, "n {n} send_one {send_one}");
            }
        }
    }

    #[test]
    fn fused_task_matches_fold_then_round() {
        for n in [1usize, 2, 3, 7, 64, 65, 200] {
            for send_one in [false, true] {
                let mut q_l: Vec<Vec<Gf>> = (0..4).map(|i| elements(n, 20 + i)).collect();
                let mut q_r: Vec<Vec<Gf>> = (0..4).map(|i| elements(n, 30 + i)).collect();
                let w = elements(n, 40);
                let rho = elements(1, 41)[0];
                let mut want_l = q_l.clone();
                let mut want_r = q_r.clone();
                let (a0, rest) = q_l.split_at_mut(1);
                let (a1, rest) = rest.split_at_mut(1);
                let (b0, rest_r) = q_r.split_at_mut(1);
                let (b1, rest_r) = rest_r.split_at_mut(1);
                let got = fused_task(
                    &mut a0[0], &mut a1[0], &rest[0], &rest[1], &mut b0[0], &mut b1[0], &rest_r[0],
                    &rest_r[1], &rho, &w, send_one,
                );
                let (c0, rest) = want_l.split_at_mut(1);
                let (c1, rest) = rest.split_at_mut(1);
                let (d0, rest_r) = want_r.split_at_mut(1);
                let (d1, rest_r) = rest_r.split_at_mut(1);
                let want = generic::fused_task(
                    &mut c0[0], &mut c1[0], &rest[0], &rest[1], &mut d0[0], &mut d1[0], &rest_r[0],
                    &rest_r[1], &rho, &w, send_one,
                );
                assert_eq!(got, want, "n {n} send_one {send_one}");
                assert_eq!(q_l, want_l);
                assert_eq!(q_r, want_r);
            }
        }
    }

    #[test]
    fn scatter_and_contract_match_the_reference() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for seed in [70u64, 71, 72] {
            let mut idx = [0u8; 64];
            for e in idx.iter_mut() {
                *e = (next() >> 20) as u8;
            }
            let eq = elements(64, seed);
            let mut got = vec![Gf::zero(); 256];
            let mut want = got.clone();
            scatter_add(&mut got, &idx, &eq);
            generic::scatter_add(&mut want, &idx, &eq);
            assert_eq!(got, want, "scatter seed {seed}");
        }
        for entries in [2usize, 4, 16] {
            let t_e = elements(entries, 71);
            let t_o = elements(entries, 72);
            let bucket = elements(entries * entries, 73);
            assert_eq!(
                contract(&t_e, &t_o, &bucket),
                generic::contract(&t_e, &t_o, &bucket),
                "contract {entries}"
            );
        }
        for n in [0usize, 1, 2, 15, 16, 33] {
            let a = elements(n, 74);
            let b = elements(n, 75);
            assert_eq!(dot(&a, &b), generic::dot(&a, &b), "dot {n}");
            let want = a.iter().zip(&b).fold(Gf::zero(), |acc, (&x, &y)| acc + x * y);
            assert_eq!(dot(&a, &b), want, "dot vs field {n}");
        }
    }

    #[test]
    fn bucketed_first_round_matches_the_plain_slots() {
        let tabs: Vec<Vec<Gf>> = (0..4).map(|i| elements(256, 80 + i)).collect();
        let mut state = 0x0123_4567_89ab_cdefu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for send_one in [false, true] {
            let mut want = Sums::zero();
            let mut bk = SumBuckets::new();
            bk.clear();
            for g in 0..5 {
                let mut pats = [[0u8; 64]; 4];
                for p in pats.iter_mut() {
                    for e in p.iter_mut() {
                        *e = (next() >> 23) as u8;
                    }
                }
                let eq = elements(64, 90 + g);
                let tab4: [&[Gf]; 4] = std::array::from_fn(|i| &tabs[i][..]);
                jit_sums_group(tab4, &pats, &eq, send_one, &mut want);
                jit_bucket_group([&tabs[2][..], &tabs[3][..]], &pats, &eq, send_one, &mut bk);
            }
            let got = jit_bucket_finish([&tabs[0][..], &tabs[1][..]], send_one, &bk);
            assert_eq!(got, want.finish(), "send_one {send_one}");
        }
    }

    #[test]
    fn jit_groups_match_the_reference() {
        let tabs: Vec<Vec<Gf>> = (0..8).map(|i| elements(256, 50 + i)).collect();
        let mut pats = [[0u8; 64]; 8];
        let mut state = 0x1234_5678_9abc_def1u64;
        for p in pats.iter_mut() {
            for e in p.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *e = (state >> 17) as u8;
            }
        }
        let rho = elements(1, 60)[0];
        for n in [1usize, 2, 5, 63, 64] {
            let w = elements(64, 61);
            for send_one in [false, true] {
                let tab4 = [&tabs[0][..], &tabs[1][..], &tabs[2][..], &tabs[3][..]];
                let pat4 = [pats[0], pats[1], pats[2], pats[3]];
                let mut got = Sums::zero();
                jit_sums_group(tab4, &pat4, &w, send_one, &mut got);
                let mut want = generic::Sums::zero();
                generic::jit_sums_group(tab4, &pat4, &w, send_one, &mut want);
                assert_eq!(got.finish(), want.finish(), "sums n {n}");

                let tab8: [&[Gf]; 8] = std::array::from_fn(|i| &tabs[i][..]);
                let mut out = vec![vec![MaybeUninit::new(Gf::zero()); n]; 4];
                let mut want_out = out.clone();
                let mut got = Sums::zero();
                {
                    let (l, r) = out.split_at_mut(2);
                    let (l0, l1) = l.split_at_mut(1);
                    let (r0, r1) = r.split_at_mut(1);
                    jit_fold_group::<false>(
                        tab8, &pats, &rho, &w, send_one, [&mut l0[0], &mut l1[0]], [&mut r0[0], &mut r1[0]],
                        &mut got,
                    );
                }
                let mut want = generic::Sums::zero();
                {
                    let (l, r) = want_out.split_at_mut(2);
                    let (l0, l1) = l.split_at_mut(1);
                    let (r0, r1) = r.split_at_mut(1);
                    generic::jit_fold_group::<false>(
                        tab8, &pats, &rho, &w, send_one, [&mut l0[0], &mut l1[0]], [&mut r0[0], &mut r1[0]],
                        &mut want,
                    );
                }
                assert_eq!(got.finish(), want.finish(), "fold n {n}");
                assert_eq!(init(&out), init(&want_out));
            }
            let mut got = vec![MaybeUninit::new(Gf::zero()); n];
            let mut want = vec![MaybeUninit::new(Gf::zero()); n];
            let tab2 = [&tabs[0][..], &tabs[1][..]];
            let pat2 = [pats[0], pats[1]];
            jit_product_group(tab2, &pat2, &mut got);
            generic::jit_product_group(tab2, &pat2, &mut want);
            assert_eq!(init(&[got]), init(&[want]), "product n {n}");
        }
    }

    #[test]
    fn pre_scaled_jit_fold_matches_the_original_fold() {
        let tabs: Vec<Vec<Gf>> = (0..8).map(|i| elements(256, 110 + i)).collect();
        let pats: [[u8; 64]; 8] =
            std::array::from_fn(|i| std::array::from_fn(|m| (53 * i + 17 * m) as u8));
        let w = elements(64, 120);
        for rho in [Gf::zero(), Gf::one(), elements(1, 121)[0]] {
            let scaled: Vec<Vec<Gf>> = tabs.iter().enumerate().map(|(i, table)| {
                let weight = if i & 2 == 0 { Gf::one() + rho } else { rho };
                table.iter().map(|&value| value * weight).collect()
            }).collect();
            let tab8: [&[Gf]; 8] = std::array::from_fn(|i| &tabs[i][..]);
            let scaled8: [&[Gf]; 8] = std::array::from_fn(|i| &scaled[i][..]);
            for n in [0usize, 1, 2, 3, 5, 31, 63, 64] {
                for send_one in [false, true] {
                    let mut want_out = vec![vec![MaybeUninit::new(Gf::zero()); n]; 4];
                    let mut want = generic::Sums::zero();
                    {
                        let (l, r) = want_out.split_at_mut(2);
                        let (l0, l1) = l.split_at_mut(1);
                        let (r0, r1) = r.split_at_mut(1);
                        generic::jit_fold_group::<false>(
                            tab8, &pats, &rho, &w, send_one,
                            [&mut l0[0], &mut l1[0]], [&mut r0[0], &mut r1[0]], &mut want,
                        );
                    }
                    let want = want.finish();
                    for portable in [false, true] {
                        let mut out = vec![vec![MaybeUninit::new(Gf::zero()); n]; 4];
                        let (l, r) = out.split_at_mut(2);
                        let (l0, l1) = l.split_at_mut(1);
                        let (r0, r1) = r.split_at_mut(1);
                        let got = if portable {
                            let mut sums = generic::Sums::zero();
                            generic::jit_fold_group::<true>(
                                scaled8, &pats, &rho, &w, send_one,
                                [&mut l0[0], &mut l1[0]], [&mut r0[0], &mut r1[0]], &mut sums,
                            );
                            sums.finish()
                        } else {
                            let mut sums = Sums::zero();
                            jit_fold_group::<true>(
                                scaled8, &pats, &rho, &w, send_one,
                                [&mut l0[0], &mut l1[0]], [&mut r0[0], &mut r1[0]], &mut sums,
                            );
                            sums.finish()
                        };
                        assert_eq!(got, want, "n {n} send_one {send_one} portable {portable}");
                        assert_eq!(init(&out), init(&want_out));
                    }
                }
            }
        }
    }

    #[test]
    fn scale_in_place_matches_the_field_multiply() {
        for n in [0usize, 1, 2, 3, 7, 64, 65, 255, 256, 257] {
            let values = elements(n, 130);
            for scalar in [Gf::zero(), Gf::one(), elements(1, 131)[0]] {
                let want: Vec<Gf> = values.iter().map(|&value| value * scalar).collect();
                let mut got = values.clone();
                scale_in_place(&mut got, &scalar);
                assert_eq!(got, want, "n {n}");
                let mut portable = values.clone();
                generic::scale_in_place(&mut portable, &scalar);
                assert_eq!(portable, want, "portable n {n}");
            }
        }
    }

    #[test]
    fn product_into_matches_the_field_multiply() {
        for n in [0usize, 1, 2, 3, 7, 64, 65, 200] {
            let a = elements(n, 101);
            let b = elements(n, 102);
            let mut got = vec![MaybeUninit::new(Gf::zero()); n];
            product_into(&a, &b, &mut got);
            let want: Vec<Gf> = a.iter().zip(&b).map(|(&x, &y)| x * y).collect();
            assert_eq!(init(&[got])[0], want, "n {n}");
        }
    }
}
