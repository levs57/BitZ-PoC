//! Their batched grand product, computed without materialising the leaves.
//!
//! Level `ℓ` of the tree (level 0 = leaves, `2^{m−ℓ}` entries, in-tree index
//! high, column low) has entries that are products of `2^ℓ` leaves, each
//! `1` or the row image `A(b)`, so the entry at in-tree position `y` of
//! column `c` is a function of the `2^ℓ` bits of column `c` under it —
//! a per-position table with `2^{2^ℓ}` entries. Folding the sumcheck's first
//! `k` rounds into such a table keeps that shape (`2^{2^{ℓ+k}}` entries), so
//! levels below 3 run their first rounds straight off the packed bits, cache
//! the 3-by-3 grid for the next two ordinary rounds, then materialise the
//! following round into one arena shared by all these levels. Level 3 is
//! never materialised either: its dense rounds start through table lookups,
//! and level 4 is built as
//! pairwise products of its table values; the levels above are products of
//! the level below. The round sums are the same field sums their dense pass
//! computes, so every message is byte-identical.

use std::collections::VecDeque;
use std::mem::MaybeUninit;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::eq_factor;
use super::gkr::{Point, eq_table, prove_dense_rounds, prove_layer_tensor};
use super::kernels;
use super::transcript::ProverState;
use field::Gf128 as Gf;
use crate::{cfg_chunks_mut, cfg_into_iter};

/// Levels below this run `MATERIALISED_LEVEL − ℓ` rounds off the bits;
/// this level's own rounds read its tables; the levels above are
/// materialised.
const MATERIALISED_LEVEL: usize = 3;
/// How many levels above [`MATERIALISED_LEVEL`]` + 1` the build pass emits
/// from the rows it just wrote, while they are still in cache; the levels
/// above those are products of the level below, read back from memory.
const FUSED_UPPER_LEVELS: usize = 3;
const PARALLEL_MIN_LANES: usize = 1 << 12;

/// The per-position value tables of one level after some folds, flat:
/// position `q` owns `data[q·len..(q+1)·len]`.
pub(crate) struct Tables {
    data: Vec<Gf>,
    len: usize,
}

impl Tables {
    fn at(&self, q: usize) -> &[Gf] {
        &self.data[q * self.len..(q + 1) * self.len]
    }

    /// Absorb the next fold into each position table once, so columns can
    /// evaluate `(1-rho)·T0[a] + rho·T1[b]` with two lookups and an XOR.
    fn scale_fold(&mut self, low_bits: usize, rho: Gf) {
        let side_len = (1usize << (low_bits - 1)) * self.len;
        let chunk_len = PARALLEL_MIN_LANES.min(side_len);
        let weights = [Gf::one() - rho, rho];
        cfg_chunks_mut!(self.data, chunk_len)
            .enumerate()
            .for_each(|(i, values)| {
                let side = (i * chunk_len / side_len) & 1;
                kernels::scale_in_place(values, &weights[side]);
            });
    }

    /// Absorb two adjacent folds into the four corresponding position
    /// tables after their ordinary round messages have been emitted.
    fn scale_fold2(&mut self, low_bits: usize, r1: Gf, r2: Gf) {
        let side_len = (1usize << (low_bits - 2)) * self.len;
        let chunk_len = PARALLEL_MIN_LANES.min(side_len);
        let one = Gf::one();
        let weights = [
            (one - r1) * (one - r2),
            (one - r1) * r2,
            r1 * (one - r2),
            r1 * r2,
        ];
        cfg_chunks_mut!(self.data, chunk_len)
            .enumerate()
            .for_each(|(i, values)| {
                let corner = (i * chunk_len / side_len) & 3;
                kernels::scale_in_place(values, &weights[corner]);
            });
    }
}

pub(crate) struct Forest<'a> {
    /// `log2` rows (the paper's `t`).
    t: usize,
    /// `log2` columns (`s`).
    s: usize,
    /// `packed_cols[g][b]`: bit `b` of columns `64g..64g+63`.
    packed_cols: &'a [Vec<u64>],
    /// `A(b) = g^{w_b}`, one per row.
    images: &'a [Gf],
}

impl<'a> Forest<'a> {
    pub(crate) fn new(t: usize, s: usize, packed_cols: &'a [Vec<u64>], images: &'a [Gf]) -> Self {
        assert!(t >= MATERIALISED_LEVEL + 1, "the bit-driven levels need t ≥ 4");
        assert_eq!(images.len(), 1 << t);
        assert!(packed_cols.len() >= (1usize << s).div_ceil(64));
        assert!(packed_cols.iter().all(|g| g.len() == 1 << t));
        Self {
            t,
            s,
            packed_cols,
            images,
        }
    }

    /// Whether the table-driven levels have two in-tree rounds left after
    /// their bit rounds (`t − 4 ≥ 2`); below that they are materialised.
    fn jit(&self) -> bool {
        self.t >= MATERIALISED_LEVEL + 3
    }

    /// Their `gpgkr_prove` over this tree: from the claim at `zeta` on the
    /// roots down to the leaf point and the claimed leaf value.
    pub(crate) fn prove(&self, ps: &mut ProverState, zeta: &[Gf]) -> (Vec<Gf>, Gf) {
        self.prove_inner(ps, zeta, true)
    }

    #[cfg(feature = "bench-internals")]
    pub(crate) fn prove_with_cached_rounds(
        &self,
        ps: &mut ProverState,
        zeta: &[Gf],
        cached_rounds: bool,
    ) -> (Vec<Gf>, Gf) {
        self.prove_inner(ps, zeta, cached_rounds)
    }

    fn prove_inner(
        &self,
        ps: &mut ProverState,
        zeta: &[Gf],
        cached_rounds: bool,
    ) -> (Vec<Gf>, Gf) {
        let t = self.t;
        let started = std::time::Instant::now();
        // The materialised path for tiny `t`: level 3 in full, the levels
        // above it by products.
        let mut levels: Vec<Option<Vec<Gf>>> = (0..t).map(|_| None).collect();
        // One arena of `2^{t−4+s}` entries, touched once: it first holds
        // levels 5..t−1 (built from level-4 rows that live only in cache),
        // is rebuilt as level 4 once those are proved, and then takes every
        // table-driven level's once-folded halves. Nothing larger than the
        // level-3 tables is allocated after it, so the prover's peak is the
        // arena plus those tables. (Writing level 4 in the same pass into
        // a second buffer instead saves the 6.3 ms rebuild less the 2.6 ms
        // of writes it skips — 1.2 % of the prove with a warm allocator —
        // for `2^{t−4+s}` more entries of peak: 256 MB here, 4 GB at
        // n = 32.)
        let mut arena: Vec<Gf> = Vec::new();
        let mut tables3 = if self.jit() {
            let tables = self.fold_table(MATERIALISED_LEVEL, 0, &[]);
            super::trace("    L3 value tables", started);
            let started = std::time::Instant::now();
            arena = Vec::with_capacity(1usize << (t - MATERIALISED_LEVEL - 1 + self.s));
            let fused = (t - MATERIALISED_LEVEL - 2).min(FUSED_UPPER_LEVELS);
            self.upper_levels_into(&tables, &mut arena, fused);
            super::trace(&format!("    levels {}..{} build", MATERIALISED_LEVEL + 2, t - 1), started);
            Some(tables)
        } else {
            levels[MATERIALISED_LEVEL] = Some(self.materialise_level(MATERIALISED_LEVEL));
            for ell in MATERIALISED_LEVEL..t - 1 {
                let next = level_up(levels[ell].as_deref().expect("built"));
                levels[ell + 1] = Some(next);
            }
            None
        };
        super::trace("  levels ≥4", started);

        let mut point: Vec<Gf> = zeta.to_owned();
        point.reverse();
        let mut point: Point = VecDeque::from(point);
        let mut claim = Gf::zero();
        for ell in (0..t).rev() {
            let started = std::time::Instant::now();
            (point, claim) = if let Some(mut wnext) = levels[ell].take() {
                let mid = wnext.len() / 2;
                let (l, r) = wnext.split_at_mut(mid);
                prove_layer_tensor(ps, point, l, r, 0, Gf::one(), VecDeque::new(), self.s)
            } else if ell >= MATERIALISED_LEVEL + 2 {
                let region = &mut arena[self.upper_region(ell)];
                let mid = region.len() / 2;
                let (l, r) = region.split_at_mut(mid);
                prove_layer_tensor(ps, point, l, r, 0, Gf::one(), VecDeque::new(), self.s)
            } else if ell == MATERIALISED_LEVEL + 1 {
                // Level 4 rebuilt in place of the proved upper levels. (Its
                // first round fused into the rebuild was measured slower:
                // the rebuild is compute-bound, so the round's slots no
                // longer hide behind a memory stream.)
                let rebuilt = std::time::Instant::now();
                arena.clear();
                self.product_level_into(tables3.as_ref().expect("jit"), &mut arena);
                super::trace("    L4 rebuild", rebuilt);
                let mid = arena.len() / 2;
                let (l, r) = arena.split_at_mut(mid);
                prove_layer_tensor(ps, point, l, r, 0, Gf::one(), VecDeque::new(), self.s)
            } else if self.jit() {
                let tables = if ell == MATERIALISED_LEVEL { tables3.take() } else { None };
                if cached_rounds && ell < MATERIALISED_LEVEL {
                    debug_assert!(tables.is_none());
                    self.prove_merged_level(ps, ell, point, &mut arena)
                } else {
                    self.prove_jit_level(ps, ell, point, tables, &mut arena)
                }
            } else {
                self.prove_bit_level(ps, ell, point)
            };
            super::trace(&format!("  level {ell}"), started);
        }
        let mut point = Vec::from(point);
        point.reverse();
        (point, claim)
    }

    /// Where level `ell ≥ 5` lives in the arena while the upper levels are
    /// proved: level 5 first, each level after the one below it.
    fn upper_region(&self, ell: usize) -> std::ops::Range<usize> {
        let t = self.t;
        debug_assert!(ell >= MATERIALISED_LEVEL + 2 && ell < t);
        let cols = 1usize << self.s;
        let offset = ((1usize << (t - MATERIALISED_LEVEL - 1)) - (1usize << (t - ell + 1))) * cols;
        offset..offset + ((1usize << (t - ell)) * cols)
    }

    /// The `k = 3 − ℓ` bit rounds of level `ℓ`; returns the challenges, the
    /// accumulated eq factor and the next point so far.
    fn bit_rounds(
        &self,
        ps: &mut ProverState,
        ell: usize,
        k: usize,
        point: &Point,
        external: &[Gf],
    ) -> (Vec<Gf>, Gf, VecDeque<Gf>) {
        let mut factor = Gf::one();
        let mut next_point = VecDeque::with_capacity(point.len() + 1);
        let mut challenges: Vec<Gf> = Vec::with_capacity(k);
        for j in 1..=k {
            let started = std::time::Instant::now();
            let z = point[j - 1];
            let send_one = z == Gf::zero();
            let (sum_endpoint, sum_inf) = self.bit_round(ell, j, &challenges, external, send_one);
            super::trace(&format!("    L{ell} bit round {j}"), started);
            ps.prover_message(&[factor * sum_endpoint, factor * sum_inf]);
            let r: Gf = ps.verifier_message();
            next_point.push_back(r);
            challenges.push(r);
            factor = factor * eq_factor(r, z);
        }
        (challenges, factor, next_point)
    }

    /// A level at or below [`MATERIALISED_LEVEL`]: `k` rounds off the bits,
    /// two rounds through the `k`-fold tables (the second writing the
    /// once-folded halves into `arena`), then the dense rounds on the arena.
    fn prove_jit_level(
        &self,
        ps: &mut ProverState,
        ell: usize,
        point: Point,
        tables: Option<Tables>,
        arena: &mut Vec<Gf>,
    ) -> (Point, Gf) {
        let t = self.t;
        let s = self.s;
        let k = MATERIALISED_LEVEL - ell;
        let external: Vec<Gf> = point.iter().rev().copied().collect();
        let (challenges, mut factor, mut next_point) = self.bit_rounds(ps, ell, k, &point, &external);
        let low_bits = t - ell - 1 - k;
        debug_assert!(low_bits >= 2);

        let started = std::time::Instant::now();
        let mut tables = match tables {
            Some(tables) if k == 0 => tables,
            _ => self.fold_table(ell, k, &challenges),
        };
        let eq_c = eq_table(&external[..s]);
        super::trace(&format!("    L{ell} tables"), started);

        // Round k + 1 off the tables.
        let started = std::time::Instant::now();
        let z = point[k];
        let send_one = z == Gf::zero();
        let eq_y = eq_table(&external[s..s + low_bits - 1]);
        let (sum_endpoint, sum_inf) = self.jit_round_sums(&tables, ell, k, &eq_c, &eq_y, send_one);
        super::trace(&format!("    L{ell} jit round"), started);
        ps.prover_message(&[factor * sum_endpoint, factor * sum_inf]);
        let r1: Gf = ps.verifier_message();
        next_point.push_back(r1);
        factor = factor * eq_factor(r1, z);

        // Round k + 2 writes the folded halves to the arena. On wide rows,
        // pre-scale the shared tables instead of multiplying every column;
        // narrow rows retain the in-register fold to avoid the extra pass.
        let started = std::time::Instant::now();
        let z = point[k + 1];
        let send_one = z == Gf::zero();
        let eq_y = eq_table(&external[s..s + low_bits - 2]);
        // Scaling costs one multiply per table entry; the original fold
        // costs one per pair of tables per column. Require at least a 2x
        // reduction in those multiplies to offset the table read/write pass.
        let (sum_endpoint, sum_inf) = if (1usize << s) >= 4 * tables.len {
            tables.scale_fold(low_bits, r1);
            self.jit_fold_round::<true>(&tables, ell, k, r1, &eq_c, &eq_y, send_one, arena)
        } else {
            self.jit_fold_round::<false>(&tables, ell, k, r1, &eq_c, &eq_y, send_one, arena)
        };
        super::trace(&format!("    L{ell} jit fold"), started);
        ps.prover_message(&[factor * sum_endpoint, factor * sum_inf]);
        let r2: Gf = ps.verifier_message();
        next_point.push_back(r2);
        factor = factor * eq_factor(r2, z);

        let started = std::time::Instant::now();
        let half = arena.len() / 2;
        let (l, r) = arena.split_at_mut(half);
        let out = prove_dense_rounds(ps, point, l, r, k + 2, Some(r2), factor, next_point, s);
        super::trace(&format!("    L{ell} dense tail"), started);
        out
    }

    /// Cache the full 3-by-3 grid for two adjacent rounds, then emit the
    /// same two ordinary Gruen messages as [`Forest::prove_jit_level`]. The
    /// transcript and verifier remain unchanged; only the prover avoids a
    /// second packed-bit pass.
    fn prove_merged_level(
        &self,
        ps: &mut ProverState,
        ell: usize,
        point: Point,
        arena: &mut Vec<Gf>,
    ) -> (Point, Gf) {
        let s = self.s;
        let k = MATERIALISED_LEVEL - ell;
        let prior = k - 1;
        let external: Vec<Gf> = point.iter().rev().copied().collect();
        let (challenges, mut factor, mut next_point) =
            self.bit_rounds(ps, ell, prior, &point, &external);
        let low_bits = self.t - ell - 1 - prior;
        debug_assert!(low_bits >= 3);

        let started = std::time::Instant::now();
        let mut tables = self.fold_table(ell, prior, &challenges);
        let eq_c = eq_table(&external[..s]);
        let eq_y = eq_table(&external[s..s + low_bits - 2]);
        super::trace(&format!("    L{ell} cached tables"), started);
        let grid_started = std::time::Instant::now();
        let grid = self.merged_round_sums(&tables, ell, prior, &eq_c, &eq_y);
        super::trace(&format!("    L{ell} cached grid"), grid_started);
        super::trace(&format!("    L{ell} cached rounds"), started);

        let one = Gf::one();
        let z1 = point[prior];
        let z2 = point[prior + 1];
        let first = [
            (one - z2) * grid[0] + z2 * grid[3],
            (one - z2) * grid[1] + z2 * grid[4],
            (one - z2) * grid[2] + z2 * grid[5],
        ];
        let endpoint = if z1 == Gf::zero() { first[1] } else { first[0] };
        ps.prover_message(&[factor * endpoint, factor * first[2]]);
        let r1: Gf = ps.verifier_message();
        next_point.push_back(r1);
        factor = factor * eq_factor(r1, z1);

        let evaluate = |values: [Gf; 3]| {
            values[0] + r1 * ((values[1] - values[0]) + (r1 - one) * values[2])
        };
        let second = [
            evaluate([grid[0], grid[1], grid[2]]),
            evaluate([grid[3], grid[4], grid[5]]),
            evaluate([grid[6], grid[7], grid[8]]),
        ];
        let endpoint = if z2 == Gf::zero() { second[1] } else { second[0] };
        ps.prover_message(&[factor * endpoint, factor * second[2]]);
        let r2: Gf = ps.verifier_message();
        next_point.push_back(r2);
        factor = factor * eq_factor(r2, z2);

        let started = std::time::Instant::now();
        tables.scale_fold2(low_bits, r1, r2);
        let z3 = point[prior + 2];
        let send_one = z3 == Gf::zero();
        let eq_y = eq_table(&external[s..s + low_bits - 3]);
        let (sum_endpoint, sum_inf) =
            self.merged_fold_round(&tables, ell, prior, &eq_c, &eq_y, send_one, arena);
        super::trace(&format!("    L{ell} cached fold"), started);
        ps.prover_message(&[factor * sum_endpoint, factor * sum_inf]);
        let r3: Gf = ps.verifier_message();
        next_point.push_back(r3);
        factor = factor * eq_factor(r3, z3);

        let started = std::time::Instant::now();
        let half = arena.len() / 2;
        let (l, r) = arena.split_at_mut(half);
        let out = prove_dense_rounds(
            ps,
            point,
            l,
            r,
            prior + 3,
            Some(r3),
            factor,
            next_point,
            s,
        );
        super::trace(&format!("    L{ell} dense tail"), started);
        out
    }

    /// The pre-arena path for tiny `t`: `k` rounds off the bits, then the
    /// folded halves are materialised and the dense rounds continue.
    fn prove_bit_level(&self, ps: &mut ProverState, ell: usize, point: Point) -> (Point, Gf) {
        let k = (MATERIALISED_LEVEL - ell).min(self.t - ell - 1);
        let external: Vec<Gf> = point.iter().rev().copied().collect();
        let (challenges, factor, next_point) = self.bit_rounds(ps, ell, k, &point, &external);
        let mut folded = self.materialise_folded(ell, k, &challenges);
        let half = folded.len() / 2;
        let (l, r) = folded.split_at_mut(half);
        prove_layer_tensor(ps, point, l, r, k, factor, next_point, self.s)
    }

    /// The row of corner `(p, y)` of level `ell` after `kk` folds, pattern
    /// bit `(v, u)`.
    #[inline]
    fn corner_row(&self, ell: usize, kk: usize, p: usize, y: usize, v: usize, u: usize) -> usize {
        let t = self.t;
        let low_bits = t - ell - 1 - kk;
        y | (v << low_bits) | (p << (t - ell - 1)) | (u << (t - ell))
    }

    /// The `2^{ell+kk}` words of column group `g` at corner `(p, y)`, in
    /// pattern-bit order.
    #[inline]
    fn corner_words(&self, ell: usize, kk: usize, p: usize, y: usize, g: usize, words: &mut [u64; 8]) {
        let col = &self.packed_cols[g];
        for v in 0..(1usize << kk) {
            for u in 0..(1usize << ell) {
                words[(v << ell) | u] = col[self.corner_row(ell, kk, p, y, v, u)];
            }
        }
    }

    /// The per-position value tables of level `ell` after `kk` folds with
    /// `r`: position `q = (p, y)` (`p` the level's product bit on top, `y`
    /// the `t − ell − 1 − kk` unbound in-tree bits), entry index = the
    /// `2^{ell+kk}`-bit pattern of the column's bits at rows
    /// `y | v ≪ (t−ell−1−kk) | p ≪ (t−ell−1) | u ≪ (t−ell)`, bit `v·2^ell + u`.
    fn fold_table(&self, ell: usize, kk: usize, r: &[Gf]) -> Tables {
        let t = self.t;
        let low_bits = t - ell - 1 - kk;
        let positions = 1usize << (t - ell - kk);
        let sub_entries = 1usize << (1usize << ell);
        let len = 1usize << (1usize << (ell + kk));
        let eq_v: Vec<Gf> = if kk == 0 {
            vec![Gf::one()]
        } else {
            let reversed: Vec<Gf> = r[..kk].iter().rev().copied().collect();
            eq_table(&reversed)
        };
        let mut data: Vec<Gf> = Vec::with_capacity(positions * len);
        let spare = &mut data.spare_capacity_mut()[..positions * len];
        cfg_chunks_mut!(spare, len).enumerate().for_each(|(q, out)| {
            let p = q >> low_bits;
            let y = q & ((1usize << low_bits) - 1);
            let mut table: Vec<Gf> = Vec::new();
            for v in 0..(1usize << kk) {
                let base = y | (v << low_bits) | (p << (t - ell - 1));
                // W_v[sub] = eq_v[v] · Π_{u ∈ sub} A(base + u·2^{t−ell}).
                let mut w = vec![Gf::zero(); sub_entries];
                w[0] = eq_v[v];
                for u in 0..(1usize << ell) {
                    let a = self.images[base | (u << (t - ell))];
                    let lim = 1usize << u;
                    for sub in 0..lim {
                        w[sub | lim] = w[sub] * a;
                    }
                }
                if v == 0 {
                    table = w;
                } else {
                    // Outer sum: new[x | sub ≪ (v·2^ell)] = table[x] + w[sub].
                    let mut next = Vec::with_capacity(table.len() * sub_entries);
                    for &ws in &w {
                        next.extend(table.iter().map(|&x| x + ws));
                    }
                    table = next;
                }
            }
            for (slot, value) in out.iter_mut().zip(table) {
                slot.write(value);
            }
        });
        // SAFETY: every one of the `positions · len` slots was written above.
        unsafe { data.set_len(positions * len) };
        Tables { data, len }
    }

    /// Level `ell` in full: index `y·2^s + c`.
    fn materialise_level(&self, ell: usize) -> Vec<Gf> {
        let tables = self.fold_table(ell, 0, &[]);
        let nb = 1usize << ell;
        let cols = 1usize << self.s;
        let groups = cols.div_ceil(64);
        let mut out = vec![Gf::zero(); (1usize << (self.t - ell)) << self.s];
        cfg_chunks_mut!(out, cols).enumerate().for_each(|(y, chunk)| {
            let table = tables.at(y);
            let mut words = [0u64; 8];
            let mut pats = [0u8; 64];
            for g in 0..groups {
                self.corner_words(ell, 0, 0, y, g, &mut words);
                patterns(&words[..nb], &mut pats);
                let base_c = g << 6;
                for j0 in 0..64.min(cols - base_c) {
                    chunk[base_c + j0] = table[pats[j0] as usize];
                }
            }
        });
        out
    }

    /// The two halves of level `ell` folded over its first `k` rounds:
    /// index `q·2^s + c`, `q = (p, y_k)`, so the first half is E.
    fn materialise_folded(&self, ell: usize, k: usize, r: &[Gf]) -> Vec<Gf> {
        let tables = self.fold_table(ell, k, r);
        let t = self.t;
        let nb = 1usize << (ell + k);
        let low_bits = t - ell - 1 - k;
        let cols = 1usize << self.s;
        let groups = cols.div_ceil(64);
        let mut out = vec![Gf::zero(); (1usize << (t - ell - k)) << self.s];
        cfg_chunks_mut!(out, cols).enumerate().for_each(|(q, chunk)| {
            let table = tables.at(q);
            let p = q >> low_bits;
            let y = q & ((1usize << low_bits) - 1);
            let mut words = [0u64; 8];
            let mut pats = [0u8; 64];
            for g in 0..groups {
                self.corner_words(ell, k, p, y, g, &mut words);
                patterns(&words[..nb], &mut pats);
                let base_c = g << 6;
                for j0 in 0..64.min(cols - base_c) {
                    chunk[base_c + j0] = table[pats[j0] as usize];
                }
            }
        });
        out
    }

    /// Level [`MATERIALISED_LEVEL`]` + 1` as pairwise products of level 3's
    /// table values — entry `(y, c)` is `T[(0, y)][pat] · T[(1, y)][pat]` —
    /// written into the (empty, pre-sized) `out`.
    fn product_level_into(&self, tables: &Tables, out: &mut Vec<Gf>) {
        let t = self.t;
        let ell = MATERIALISED_LEVEL;
        let cols = 1usize << self.s;
        let groups = cols.div_ceil(64);
        let rows = 1usize << (t - ell - 1);
        let len = rows * cols;
        assert!(out.is_empty() && out.capacity() >= len);
        let spare = &mut out.spare_capacity_mut()[..len];
        cfg_chunks_mut!(spare, cols).enumerate().for_each(|(y, chunk)| {
            let tab = [tables.at(y), tables.at(y | (1 << (t - ell - 1)))];
            let mut words = [0u64; 8];
            let mut pats = [[0u8; 64]; 2];
            for g in 0..groups {
                for b in 0..2 {
                    self.corner_words(ell, 0, b, y, g, &mut words);
                    pats[b] = transposed_patterns(&mut words);
                }
                let base_c = g << 6;
                let width = 64.min(cols - base_c);
                kernels::jit_product_group(tab, &pats, &mut chunk[base_c..base_c + width]);
            }
        });
        // SAFETY: every slot of every row chunk was written by the kernel.
        unsafe { out.set_len(len) };
    }

    /// Levels 5..t−1 into the (empty, pre-sized) `arena`, each at its
    /// [`Forest::upper_region`]: the first `fused` of them from level-4
    /// rows computed 64 columns at a time into a task-local buffer and never
    /// stored — a task owns level 4's rows `y0 + b·2^{t−4−fused}` for every
    /// `b`, the whole subtree above row `y0` of level `4 + fused`, so every
    /// product it emits reads rows it computed itself — and the levels
    /// above those as products of the level below. Level 4 itself is
    /// rebuilt into the arena when its turn comes
    /// ([`Forest::product_level_into`]).
    fn upper_levels_into(&self, tables: &Tables, arena: &mut Vec<Gf>, fused: usize) {
        let t = self.t;
        let ell = MATERIALISED_LEVEL;
        let cols = 1usize << self.s;
        let groups = cols.div_ceil(64);
        let rows = 1usize << (t - ell - 1);
        let total = self.upper_region(t - 1).end;
        assert!(arena.is_empty() && arena.capacity() >= total);
        assert!(fused >= 1 && fused + ell + 2 <= t, "level {} does not exist", ell + 1 + fused);
        let spare = &mut arena.spare_capacity_mut()[..total];
        // Rows of level `4 + fused` = tasks; level `4 + i` has `sub ≪ (fused − i)`.
        let sub = rows >> fused;
        let above: Vec<RowsPtr> = (1..=fused)
            .map(|i| RowsPtr::new(&mut spare[self.upper_region(ell + 1 + i)], cols))
            .collect();
        cfg_into_iter!(0..sub, 1).for_each(|y0| {
            let mut words = [0u64; 8];
            let mut pats = [[0u8; 64]; 2];
            let mut l4 = [[MaybeUninit::<Gf>::uninit(); 64]; 1 << FUSED_UPPER_LEVELS];
            for g in 0..groups {
                let base_c = g << 6;
                let width = 64.min(cols - base_c);
                let range = base_c..base_c + width;
                for (b, row) in l4.iter_mut().enumerate().take(1 << fused) {
                    let y = y0 + b * sub;
                    let tab = [tables.at(y), tables.at(y | (1 << (t - ell - 1)))];
                    for p in 0..2 {
                        self.corner_words(ell, 0, p, y, g, &mut words);
                        pats[p] = transposed_patterns(&mut words);
                    }
                    kernels::jit_product_group(tab, &pats, &mut row[..width]);
                }
                for i in 1..=fused {
                    let half = 1usize << (fused - i);
                    for bp in 0..half {
                        let y = y0 + bp * sub;
                        // SAFETY: the level-4 segments were written just
                        // above (their first `width` slots); rows `y` and
                        // `y + half·sub` of level `3 + i > 4` were written by
                        // this task in this group's iteration; row `y` of
                        // level `4 + i` belongs to it alone.
                        unsafe {
                            let (a, b): (&[Gf], &[Gf]) = if i == 1 {
                                (init_prefix(&l4[bp], width), init_prefix(&l4[bp + half], width))
                            } else {
                                let below = &above[i - 2];
                                (&below.row_init(y)[range.clone()], &below.row_init(y + half * sub)[range.clone()])
                            };
                            let o = &mut above[i - 1].row(y)[range.clone()];
                            kernels::product_into(a, b, o);
                        }
                    }
                }
            }
        });
        // The levels above `4 + fused`, each the products of the one below.
        for lower in ell + 1 + fused..t - 1 {
            let (below, rest) = spare.split_at_mut(self.upper_region(lower + 1).start);
            let below = &below[self.upper_region(lower)];
            // SAFETY: level `lower` was fully written (by the pass or the
            // previous iteration) and `MaybeUninit<Gf>` has `Gf`'s layout.
            let below = unsafe { std::slice::from_raw_parts(below.as_ptr().cast::<Gf>(), below.len()) };
            let len = self.upper_region(lower + 1).len();
            level_up_into(below, &mut rest[..len]);
        }
        // SAFETY: the tasks partition the rows of levels 5..4+fused and the
        // loop above writes every level after them in full.
        unsafe { arena.set_len(total) };
    }

    /// The tensor grid for two adjacent rounds through the 16-entry tables
    /// present after `prior = 2 - ell` ordinary bit rounds. Grid order is
    /// `(0,0), (1,0), (inf,0), (0,1), (1,1), (inf,1), (0,inf), (1,inf),
    /// `(inf,inf)`.
    fn merged_round_sums(
        &self,
        tables: &Tables,
        ell: usize,
        prior: usize,
        eq_c: &[Gf],
        eq_y: &[Gf],
    ) -> [Gf; 9] {
        let low_bits = self.t - ell - 1 - prior;
        let cols = 1usize << self.s;
        let groups = cols.div_ceil(64);
        let rows = 1usize << (low_bits - 2);
        debug_assert_eq!(ell + prior, 2);
        debug_assert_eq!(tables.len, 16);
        debug_assert_eq!(eq_y.len(), rows);
        let eq_t = transposed_eq(eq_c);
        let row = |y2: usize, scratch: &mut MergedScratch| {
            let tab: [&[Gf]; 8] = std::array::from_fn(|corner| {
                let (p, b1, b2) = (corner >> 2, (corner >> 1) & 1, corner & 1);
                tables.at(
                    (p << low_bits)
                        | (b1 << (low_bits - 1))
                        | (b2 << (low_bits - 2))
                        | y2,
                )
            });
            for g in 0..groups {
                for corner in 0..8 {
                    let (p, b1, b2) = (corner >> 2, (corner >> 1) & 1, corner & 1);
                    let y = y2 | (b2 << (low_bits - 2)) | (b1 << (low_bits - 1));
                    let mut words = [0u64; 8];
                    self.corner_words(ell, prior, p, y, g, &mut words);
                    scratch.patterns[8 * g + corner] = transposed_patterns(&mut words);
                }
            }

            let mut terms = [Gf::zero(); 16];
            for e in 0..4 {
                for o in 0..4 {
                    scratch.bucket.fill(Gf::zero());
                    for g in 0..groups {
                        let pe = &scratch.patterns[8 * g + e];
                        let po = &scratch.patterns[8 * g + 4 + o];
                        let mut index = [0u8; 64];
                        for m in 0..64 {
                            index[m] = (pe[m] << 4) | po[m];
                        }
                        kernels::scatter_add(
                            &mut scratch.bucket,
                            &index,
                            &eq_t[g << 6..(g + 1) << 6],
                        );
                    }
                    terms[4 * e + o] =
                        kernels::contract(tab[e], tab[4 + o], &scratch.bucket);
                }
            }
            let mut grid = [Gf::zero(); 9];
            grid[0] = terms[0];
            grid[1] = terms[10];
            grid[2] = terms[0] + terms[2] + terms[8] + terms[10];
            grid[3] = terms[5];
            grid[4] = terms[15];
            grid[5] = terms[5] + terms[7] + terms[13] + terms[15];
            grid[6] = terms[0] + terms[1] + terms[4] + terms[5];
            grid[7] = terms[10] + terms[11] + terms[14] + terms[15];
            grid[8] = terms.into_iter().fold(Gf::zero(), |sum, value| sum + value);
            let weight = eq_y[y2];
            grid.map(|value| value * weight)
        };

        #[cfg(feature = "parallel")]
        let partials: Vec<[Gf; 9]> = (0..rows)
            .into_par_iter()
            .with_min_len(1)
            .map_init(|| MergedScratch::new(groups), |scratch, y2| row(y2, scratch))
            .collect();
        #[cfg(not(feature = "parallel"))]
        let partials: Vec<[Gf; 9]> = {
            let mut scratch = MergedScratch::new(groups);
            (0..rows).map(|y2| row(y2, &mut scratch)).collect()
        };
        partials.into_iter().fold([Gf::zero(); 9], |mut total, part| {
            for i in 0..9 {
                total[i] += part[i];
            }
            total
        })
    }

    /// Round `k + 1` of level `ell` through its `k`-fold tables: the sums
    /// over `(y1, c)` of the corners `(p, b1, y1)`, `b1` the bit bound this
    /// round, weighted `eq_y[y1]·eq_c[c]`.
    fn jit_round_sums(
        &self,
        tables: &Tables,
        ell: usize,
        k: usize,
        eq_c: &[Gf],
        eq_y: &[Gf],
        send_one: bool,
    ) -> (Gf, Gf) {
        let low_bits = self.t - ell - 1 - k;
        let cols = 1usize << self.s;
        let groups = cols.div_ceil(64);
        let rows = 1usize << (low_bits - 1);
        debug_assert_eq!(eq_y.len(), rows);
        let eq_t = transposed_eq(eq_c);
        let row = |y1: usize, bk: &mut kernels::SumBuckets| {
            bk.clear();
            let mut words = [0u64; 8];
            let mut pats = [[0u8; 64]; 4];
            let tab: [&[Gf]; 4] = std::array::from_fn(|corner| {
                let (p, b1) = (corner >> 1, corner & 1);
                tables.at((p << low_bits) | (b1 << (low_bits - 1)) | y1)
            });
            for g in 0..groups {
                for corner in 0..4 {
                    let (p, b1) = (corner >> 1, corner & 1);
                    let y = y1 | (b1 << (low_bits - 1));
                    self.corner_words(ell, k, p, y, g, &mut words);
                    pats[corner] = transposed_patterns(&mut words);
                }
                kernels::jit_bucket_group([tab[2], tab[3]], &pats, &eq_t[g << 6..(g + 1) << 6], send_one, bk);
            }
            let (end, inf) = kernels::jit_bucket_finish([tab[0], tab[1]], send_one, bk);
            let w = eq_y[y1];
            (end * w, inf * w)
        };
        #[cfg(feature = "parallel")]
        let partials: Vec<(Gf, Gf)> = (0..rows)
            .into_par_iter()
            .with_min_len(1)
            .map_init(kernels::SumBuckets::new, |bk, y1| row(y1, bk))
            .collect();
        #[cfg(not(feature = "parallel"))]
        let partials: Vec<(Gf, Gf)> = {
            let mut bk = kernels::SumBuckets::new();
            (0..rows).map(|y1| row(y1, &mut bk)).collect()
        };
        partials
            .into_iter()
            .fold((Gf::zero(), Gf::zero()), |(a, b), (x, y)| (a + x, b + y))
    }

    /// Round `k + 2` of level `ell` through its `k`-fold tables with the
    /// previous challenge `rho` absorbed in the tables or folded on the fly:
    /// writes the once-folded
    /// halves (`E'` then `O'`, each `2^{low_bits−1}` rows of `2^s`) into
    /// `arena` and returns this round's sums over the corners `(b2, y2)`.
    #[allow(clippy::too_many_arguments)]
    fn jit_fold_round<const PRE_SCALED: bool>(
        &self,
        tables: &Tables,
        ell: usize,
        k: usize,
        rho: Gf,
        eq_c: &[Gf],
        eq_y: &[Gf],
        send_one: bool,
        arena: &mut Vec<Gf>,
    ) -> (Gf, Gf) {
        let low_bits = self.t - ell - 1 - k;
        let cols = 1usize << self.s;
        let rows = 1usize << (low_bits - 2);
        let half = 2 * rows * cols;
        let len = 2 * half;
        debug_assert_eq!(eq_y.len(), rows);
        assert!(arena.capacity() >= len, "arena too small");
        if arena.len() != len {
            // First fill: through the spare capacity, no memset.
            arena.clear();
            let spare = &mut arena.spare_capacity_mut()[..len];
            let sums = self.jit_fold_into::<PRE_SCALED>(tables, ell, k, rho, eq_c, eq_y, send_one, spare);
            // SAFETY: `jit_fold_into` writes every slot of both halves.
            unsafe { arena.set_len(len) };
            sums
        } else {
            let slots = &mut arena[..len];
            // SAFETY: `MaybeUninit<Gf>` has `Gf`'s layout and only
            // initialised values are ever written through the view.
            let view = unsafe {
                std::slice::from_raw_parts_mut(slots.as_mut_ptr().cast::<MaybeUninit<Gf>>(), len)
            };
            self.jit_fold_into::<PRE_SCALED>(tables, ell, k, rho, eq_c, eq_y, send_one, view)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn jit_fold_into<const PRE_SCALED: bool>(
        &self,
        tables: &Tables,
        ell: usize,
        k: usize,
        rho: Gf,
        eq_c: &[Gf],
        eq_y: &[Gf],
        send_one: bool,
        out: &mut [MaybeUninit<Gf>],
    ) -> (Gf, Gf) {
        let low_bits = self.t - ell - 1 - k;
        let cols = 1usize << self.s;
        let groups = cols.div_ceil(64);
        let rows = 1usize << (low_bits - 2);
        let half = out.len() / 2;
        let (e_half, o_half) = out.split_at_mut(half);
        let (e_lo, e_hi) = e_half.split_at_mut(half / 2);
        let (o_lo, o_hi) = o_half.split_at_mut(half / 2);
        let eq_t = transposed_eq(eq_c);
        let partials: Vec<(Gf, Gf)> = cfg_chunks_mut!(e_lo, cols)
            .zip(cfg_chunks_mut!(e_hi, cols))
            .zip(cfg_chunks_mut!(o_lo, cols))
            .zip(cfg_chunks_mut!(o_hi, cols))
            .enumerate()
            .map(|(y2, (((el, eh), ol), oh))| {
                debug_assert!(y2 < rows);
                let mut sums = kernels::Sums::zero();
                let mut words = [0u64; 8];
                let mut pats = [[0u8; 64]; 8];
                let tab: [&[Gf]; 8] = std::array::from_fn(|corner| {
                    let (p, b1, b2) = (corner >> 2, (corner >> 1) & 1, corner & 1);
                    tables.at((p << low_bits) | (b1 << (low_bits - 1)) | (b2 << (low_bits - 2)) | y2)
                });
                for g in 0..groups {
                    for corner in 0..8 {
                        let (p, b1, b2) = (corner >> 2, (corner >> 1) & 1, corner & 1);
                        let y = y2 | (b2 << (low_bits - 2)) | (b1 << (low_bits - 1));
                        self.corner_words(ell, k, p, y, g, &mut words);
                        pats[corner] = transposed_patterns(&mut words);
                    }
                    let base_c = g << 6;
                    let width = 64.min(cols - base_c);
                    let range = base_c..base_c + width;
                    kernels::jit_fold_group::<PRE_SCALED>(
                        tab,
                        &pats,
                        &rho,
                        &eq_t[g << 6..(g + 1) << 6],
                        send_one,
                        [&mut el[range.clone()], &mut eh[range.clone()]],
                        [&mut ol[range.clone()], &mut oh[range]],
                        &mut sums,
                    );
                }
                let (end, inf) = sums.finish();
                let w = eq_y[y2];
                (end * w, inf * w)
            })
            .collect();
        partials
            .into_iter()
            .fold((Gf::zero(), Gf::zero()), |(a, b), (x, y)| (a + x, b + y))
    }

    /// Materialise the following ordinary round after two cached rounds,
    /// and accumulate that round's Gruen message in the same pass.
    #[allow(clippy::too_many_arguments)]
    fn merged_fold_round(
        &self,
        tables: &Tables,
        ell: usize,
        prior: usize,
        eq_c: &[Gf],
        eq_y: &[Gf],
        send_one: bool,
        arena: &mut Vec<Gf>,
    ) -> (Gf, Gf) {
        let low_bits = self.t - ell - 1 - prior;
        let cols = 1usize << self.s;
        let rows = 1usize << (low_bits - 3);
        let len = 4 * rows * cols;
        debug_assert_eq!(eq_y.len(), rows);
        assert!(arena.capacity() >= len, "arena too small");
        if arena.len() != len {
            arena.clear();
            let spare = &mut arena.spare_capacity_mut()[..len];
            let sums = self.merged_fold_into(tables, ell, prior, eq_c, eq_y, send_one, spare);
            // SAFETY: `merged_fold_into` writes every slot.
            unsafe { arena.set_len(len) };
            sums
        } else {
            let slots = &mut arena[..len];
            // SAFETY: the view is used only to overwrite every slot.
            let view = unsafe {
                std::slice::from_raw_parts_mut(slots.as_mut_ptr().cast::<MaybeUninit<Gf>>(), len)
            };
            self.merged_fold_into(tables, ell, prior, eq_c, eq_y, send_one, view)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn merged_fold_into(
        &self,
        tables: &Tables,
        ell: usize,
        prior: usize,
        eq_c: &[Gf],
        eq_y: &[Gf],
        send_one: bool,
        out: &mut [MaybeUninit<Gf>],
    ) -> (Gf, Gf) {
        let low_bits = self.t - ell - 1 - prior;
        let cols = 1usize << self.s;
        let groups = cols.div_ceil(64);
        let rows = 1usize << (low_bits - 3);
        let quarter = rows * cols;
        let (e_lo, rest) = out.split_at_mut(quarter);
        let (e_hi, rest) = rest.split_at_mut(quarter);
        let (o_lo, o_hi) = rest.split_at_mut(quarter);
        let eq_t = transposed_eq(eq_c);
        let partials: Vec<(Gf, Gf)> = cfg_chunks_mut!(e_lo, cols)
            .zip(cfg_chunks_mut!(e_hi, cols))
            .zip(cfg_chunks_mut!(o_lo, cols))
            .zip(cfg_chunks_mut!(o_hi, cols))
            .enumerate()
            .map(|(y3, (((el, eh), ol), oh))| {
                let tab: [&[Gf]; 16] = std::array::from_fn(|corner| {
                    let (p, b1, b2, b3) =
                        (corner >> 3, (corner >> 2) & 1, (corner >> 1) & 1, corner & 1);
                    tables.at(
                        (p << low_bits)
                            | (b1 << (low_bits - 1))
                            | (b2 << (low_bits - 2))
                            | (b3 << (low_bits - 3))
                            | y3,
                    )
                });
                let mut sums = kernels::Sums::zero();
                for g in 0..groups {
                    let mut pats = [[0u8; 64]; 16];
                    for corner in 0..16 {
                        let (p, b1, b2, b3) =
                            (corner >> 3, (corner >> 2) & 1, (corner >> 1) & 1, corner & 1);
                        let y = y3
                            | (b3 << (low_bits - 3))
                            | (b2 << (low_bits - 2))
                            | (b1 << (low_bits - 1));
                        let mut words = [0u64; 8];
                        self.corner_words(ell, prior, p, y, g, &mut words);
                        pats[corner] = transposed_patterns(&mut words);
                    }
                    let base_c = g << 6;
                    let width = 64.min(cols - base_c);
                    let range = base_c..base_c + width;
                    kernels::jit_fold2_group(
                        tab,
                        &pats,
                        &eq_t[g << 6..(g + 1) << 6],
                        send_one,
                        [&mut el[range.clone()], &mut eh[range.clone()]],
                        [&mut ol[range.clone()], &mut oh[range]],
                        &mut sums,
                    );
                }
                let (end, inf) = sums.finish();
                let weight = eq_y[y3];
                (end * weight, inf * weight)
            })
            .collect();
        partials
            .into_iter()
            .fold((Gf::zero(), Gf::zero()), |(a, b), (x, y)| (a + x, b + y))
    }

    /// Round `j` (1-based, `j ≤ k`) of level `ell`'s sumcheck off the bits:
    /// `Σ eq·E_end·O_end` and `Σ eq·(E_hi − E_lo)(O_hi − O_lo)` over the
    /// `(y_j, c)` terms, with `E_·`, `O_·` read from the `(j−1)`-fold tables.
    ///
    /// Within a row `y` each of `E_lo, E_hi, O_lo, O_hi` is a function of the
    /// column's `nb`-bit pattern, so a row's sum is `Σ_{a,b} T_E[a]·T_O[b]·
    /// Σ_{c: pat_E(c)=a, pat_O(c)=b} eq_c[c]`: the terms only bucket `eq_c`
    /// by their `(E, O)` pattern pair — one 16-byte addition per term for
    /// widths 1 and 2 (the whole `(E_lo, E_hi, O_lo, O_hi)` tuple, one byte
    /// straight out of the bit-block transpose, marginalised afterwards),
    /// four for width 4 (the pair bytes come out of two transposes, the
    /// cross pairs by swapping nibbles) — and the `2^{2nb} + 2^{nb}`
    /// products of the contraction are paid once per row.
    fn bit_round(
        &self,
        ell: usize,
        j: usize,
        challenges: &[Gf],
        external: &[Gf],
        send_one: bool,
    ) -> (Gf, Gf) {
        let t = self.t;
        let s = self.s;
        let kk = j - 1;
        let tables = self.fold_table(ell, kk, challenges);
        let nb = 1usize << (ell + kk);
        debug_assert!(nb <= 4, "the pair buckets need nb ≤ 4");
        let y_bits = t - ell - 1 - j;
        let eq_t = transposed_eq(&eq_table(&external[..s]));
        let eq_y = eq_table(&external[s..s + y_bits]);
        let rows = 1usize << y_bits;
        // Width 1: two rows share one byte index, so one scatter serves
        // two terms ([`Forest::bit_row_pair`]).
        let paired = nb == 1 && rows >= 2;
        let tasks = if paired { rows / 2 } else { rows };
        let task = |i: usize, buckets: &mut Buckets| {
            if paired {
                self.bit_row_pair(&tables, ell, kk, 2 * i, y_bits, &eq_t, &eq_y, send_one, buckets)
            } else {
                let (end, inf) = self.bit_row(&tables, ell, kk, nb, i, y_bits, &eq_t, send_one, buckets);
                let w = eq_y[i];
                (end * w, inf * w)
            }
        };
        #[cfg(feature = "parallel")]
        let partials: Vec<(Gf, Gf)> = (0..tasks)
            .into_par_iter()
            .with_min_len(1)
            .map_init(Buckets::new, |buckets, i| task(i, buckets))
            .collect();
        #[cfg(not(feature = "parallel"))]
        let partials: Vec<(Gf, Gf)> = {
            let mut buckets = Buckets::new();
            (0..tasks).map(|i| task(i, &mut buckets)).collect()
        };
        partials
            .into_iter()
            .fold((Gf::zero(), Gf::zero()), |(a, b), (x, y)| (a + x, b + y))
    }

    /// Rows `y0` and `y0 + 1` of a width-1 [`Forest::bit_round`] together,
    /// weighted: the two rows' 4-bit tuples `(E_lo, E_hi, O_lo, O_hi)` fill
    /// one byte index out of one block transpose (row `y0` in the low
    /// nibble), so a column's weight is scattered once for both rows; the
    /// per-row coefficient of a tuple is `w·E_end·O_end` (`w·(E_hi + E_lo)
    /// (O_hi + O_lo)` for the ∞ sum), and since the pair's coefficient is
    /// the sum of the two rows' the contraction only needs the two 16-entry
    /// marginals of the 256 buckets.
    #[allow(clippy::too_many_arguments)]
    fn bit_row_pair(
        &self,
        tables: &Tables,
        ell: usize,
        kk: usize,
        y0: usize,
        y_bits: usize,
        eq_t: &[Gf],
        eq_y: &[Gf],
        send_one: bool,
        bk: &mut Buckets,
    ) -> (Gf, Gf) {
        debug_assert_eq!(ell + kk, 0);
        let low_bits = y_bits + 1;
        let cols = 1usize << self.s;
        let groups = cols.div_ceil(64);
        let zero = Gf::zero();
        bk.tuples.fill(zero);
        let corner_y = |y: usize, corner: usize| y | ((corner & 1) << y_bits);
        let corner_p = |corner: usize| corner >> 1;
        let mut tmp = [0u64; 8];
        let mut words = [0u64; 8];
        for g in 0..groups {
            for i in 0..2 {
                for corner in 0..4 {
                    self.corner_words(ell, kk, corner_p(corner), corner_y(y0 + i, corner), g, &mut tmp);
                    words[4 * i + corner] = tmp[0];
                }
            }
            let idx = transposed_patterns(&mut words);
            kernels::scatter_add(&mut bk.tuples, &idx, &eq_t[g << 6..(g + 1) << 6]);
        }
        // The marginals: row `y0`'s tuple is the low nibble.
        let mut marginal = [[zero; 16]; 2];
        for (k, &v) in bk.tuples.iter().enumerate() {
            marginal[0][k & 15] += v;
            marginal[1][k >> 4] += v;
        }
        let (mut end, mut inf) = (zero, zero);
        for (i, m) in marginal.iter().enumerate() {
            let y = y0 + i;
            let q = |corner: usize| (corner_p(corner) << low_bits) | corner_y(y, corner);
            let (t_e_lo, t_e_hi) = (tables.at(q(0)), tables.at(q(1)));
            let (t_o_lo, t_o_hi) = (tables.at(q(2)), tables.at(q(3)));
            let w = eq_y[y];
            let (t_e_end, t_o_end) = if send_one { (t_e_hi, t_o_hi) } else { (t_e_lo, t_o_lo) };
            let we: [Gf; 2] = [w * t_e_end[0], w * t_e_end[1]];
            let mut c_end = [zero; 16];
            let mut c_inf = [zero; 16];
            for (t, (c_end_t, c_inf_t)) in c_end.iter_mut().zip(c_inf.iter_mut()).enumerate() {
                let (e_lo, e_hi, o_lo, o_hi) = (t & 1, (t >> 1) & 1, (t >> 2) & 1, (t >> 3) & 1);
                let (e_end, o_end) = if send_one { (e_hi, o_hi) } else { (e_lo, o_lo) };
                *c_end_t = we[e_end] * t_o_end[o_end];
                *c_inf_t = w * (t_e_hi[e_hi] + t_e_lo[e_lo]) * (t_o_hi[o_hi] + t_o_lo[o_lo]);
            }
            end = end + kernels::dot(&c_end, m);
            inf = inf + kernels::dot(&c_inf, m);
        }
        (end, inf)
    }

    /// One row of [`Forest::bit_round`]: bucket the row's columns, then
    /// contract with the four corner tables.
    #[allow(clippy::too_many_arguments)]
    fn bit_row(
        &self,
        tables: &Tables,
        ell: usize,
        kk: usize,
        nb: usize,
        y: usize,
        y_bits: usize,
        eq_t: &[Gf],
        send_one: bool,
        bk: &mut Buckets,
    ) -> (Gf, Gf) {
        let low_bits = y_bits + 1;
        let entries = 1usize << nb;
        let pairs = entries * entries;
        let cols = 1usize << self.s;
        let groups = cols.div_ceil(64);
        bk.reset(nb);
        // The four (p, bit) corners: E_lo, E_hi, O_lo, O_hi.
        let corner_y = |corner: usize| y | ((corner & 1) << y_bits);
        let corner_p = |corner: usize| corner >> 1;
        let mut tmp = [0u64; 8];
        let mut words = [0u64; 8];
        let mut lh = [0u8; 64];
        let mut hl = [0u8; 64];
        for g in 0..groups {
            let eq_g = &eq_t[g << 6..(g + 1) << 6];
            if nb <= 2 {
                // Tuple byte: corner `i`'s pattern at bits `i·nb..`.
                words = [0u64; 8];
                for corner in 0..4 {
                    self.corner_words(ell, kk, corner_p(corner), corner_y(corner), g, &mut tmp);
                    words[corner * nb..(corner + 1) * nb].copy_from_slice(&tmp[..nb]);
                }
                let idx = transposed_patterns(&mut words);
                kernels::scatter_add(&mut bk.tuples, &idx, eq_g);
            } else {
                // Pair bytes `pat_E·16 + pat_O`: the O words in the low
                // nibble, the E words in the high one.
                let mut idx = [[0u8; 64]; 2];
                for (out, e_corner, o_corner) in [(0usize, 0usize, 2usize), (1, 1, 3)] {
                    self.corner_words(ell, kk, corner_p(o_corner), corner_y(o_corner), g, &mut tmp);
                    words[..4].copy_from_slice(&tmp[..4]);
                    self.corner_words(ell, kk, corner_p(e_corner), corner_y(e_corner), g, &mut tmp);
                    words[4..8].copy_from_slice(&tmp[..4]);
                    idx[out] = transposed_patterns(&mut words);
                }
                for m in 0..64 {
                    lh[m] = (idx[0][m] & 0xF0) | (idx[1][m] & 0x0F);
                    hl[m] = (idx[1][m] & 0xF0) | (idx[0][m] & 0x0F);
                }
                kernels::scatter_add(&mut bk.ll, &idx[0], eq_g);
                kernels::scatter_add(&mut bk.hh, &idx[1], eq_g);
                kernels::scatter_add(&mut bk.lh, &lh, eq_g);
                kernels::scatter_add(&mut bk.hl, &hl, eq_g);
            }
        }
        if nb <= 2 {
            // Marginalise the tuple buckets into the four pair buckets.
            let mask = entries - 1;
            for (tuple, &value) in bk.tuples[..1 << (4 * nb)].iter().enumerate() {
                let pe_lo = tuple & mask;
                let pe_hi = (tuple >> nb) & mask;
                let po_lo = (tuple >> (2 * nb)) & mask;
                let po_hi = (tuple >> (3 * nb)) & mask;
                bk.ll[pe_lo * entries + po_lo] += value;
                bk.lh[pe_lo * entries + po_hi] += value;
                bk.hl[pe_hi * entries + po_lo] += value;
                bk.hh[pe_hi * entries + po_hi] += value;
            }
        }
        let q = |corner: usize| (corner_p(corner) << low_bits) | corner_y(corner);
        let (t_e_lo, t_e_hi) = (tables.at(q(0)), tables.at(q(1)));
        let (t_o_lo, t_o_hi) = (tables.at(q(2)), tables.at(q(3)));
        let ll = kernels::contract(t_e_lo, t_o_lo, &bk.ll[..pairs]);
        let hh = kernels::contract(t_e_hi, t_o_hi, &bk.hh[..pairs]);
        let end = if send_one { hh } else { ll };
        // (E_hi − E_lo)(O_hi − O_lo) expands to the four combinations;
        // characteristic two makes every sign a plus.
        let inf = hh
            + kernels::contract(t_e_hi, t_o_lo, &bk.hl[..pairs])
            + kernels::contract(t_e_lo, t_o_hi, &bk.lh[..pairs])
            + ll;
        (end, inf)
    }
}

/// Per-task scratch for the cached two-round grid. Patterns are retained so
/// one 256-entry pair bucket stays hot while all column groups feed it.
struct MergedScratch {
    bucket: Vec<Gf>,
    patterns: Vec<[u8; 64]>,
}

impl MergedScratch {
    fn new(groups: usize) -> Self {
        Self {
            bucket: vec![Gf::zero(); 256],
            patterns: vec![[0u8; 64]; 8 * groups],
        }
    }
}

/// The per-task bucket store of the bit rounds: 256 entries each so every
/// byte index is in bounds; only the used prefixes are cleared per row.
struct Buckets {
    tuples: Vec<Gf>,
    ll: Vec<Gf>,
    lh: Vec<Gf>,
    hl: Vec<Gf>,
    hh: Vec<Gf>,
}

impl Buckets {
    fn new() -> Self {
        let fresh = || vec![Gf::zero(); 256];
        Self {
            tuples: fresh(),
            ll: fresh(),
            lh: fresh(),
            hl: fresh(),
            hh: fresh(),
        }
    }

    fn reset(&mut self, nb: usize) {
        let pairs = 1usize << (2 * nb);
        let zero = Gf::zero();
        if nb <= 2 {
            self.tuples[..1 << (4 * nb)].fill(zero);
        }
        self.ll[..pairs].fill(zero);
        self.lh[..pairs].fill(zero);
        self.hl[..pairs].fill(zero);
        self.hh[..pairs].fill(zero);
    }
}

/// One level up: `out[i] = lower[i] · lower[i + half]`.
fn level_up(lower: &[Gf]) -> Vec<Gf> {
    let half = lower.len() / 2;
    let mut out: Vec<Gf> = Vec::with_capacity(half);
    level_up_into(lower, &mut out.spare_capacity_mut()[..half]);
    // SAFETY: `level_up_into` writes every one of the `half` slots.
    unsafe { out.set_len(half) };
    out
}

/// [`level_up`] into pre-sized slots.
fn level_up_into(lower: &[Gf], out: &mut [MaybeUninit<Gf>]) {
    let half = lower.len() / 2;
    assert_eq!(out.len(), half);
    let (l, r) = lower.split_at(half);
    cfg_chunks_mut!(out, PARALLEL_MIN_LANES)
        .enumerate()
        .for_each(|(i, chunk)| {
            let start = i * PARALLEL_MIN_LANES;
            kernels::product_into(&l[start..start + chunk.len()], &r[start..start + chunk.len()], chunk);
        });
}

/// The first `len` slots of `buf` as values.
///
/// # Safety
/// Those slots must have been written.
#[inline]
unsafe fn init_prefix(buf: &[MaybeUninit<Gf>; 64], len: usize) -> &[Gf] {
    debug_assert!(len <= 64);
    // SAFETY: in bounds; `MaybeUninit<Gf>` has `Gf`'s layout and the prefix
    // is initialised per the contract.
    unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<Gf>(), len) }
}

/// Row access into an uninitialised level for tasks that own disjoint
/// rows: `row(y)` is row `y`'s slots, `row_init(y)` the same row once
/// written.
#[derive(Clone, Copy)]
struct RowsPtr {
    ptr: *mut MaybeUninit<Gf>,
    rows: usize,
    cols: usize,
}

// SAFETY: the pointer is only dereferenced through the `unsafe` accessors,
// whose callers keep the rows they touch disjoint across tasks.
unsafe impl Send for RowsPtr {}
unsafe impl Sync for RowsPtr {}

impl RowsPtr {
    fn new(slots: &mut [MaybeUninit<Gf>], cols: usize) -> Self {
        debug_assert_eq!(slots.len() % cols, 0);
        Self {
            ptr: slots.as_mut_ptr(),
            rows: slots.len() / cols,
            cols,
        }
    }

    /// Row `y`'s slots.
    ///
    /// # Safety
    /// No other live reference to row `y` may exist while this one does.
    #[inline]
    unsafe fn row<'b>(&self, y: usize) -> &'b mut [MaybeUninit<Gf>] {
        assert!(y < self.rows);
        // SAFETY: in bounds by the assertion; exclusivity is the caller's.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(y * self.cols), self.cols) }
    }

    /// Row `y` as initialised values.
    ///
    /// # Safety
    /// Every slot of row `y` must have been written, and no `&mut` to the
    /// row may be live.
    #[inline]
    unsafe fn row_init<'b>(&self, y: usize) -> &'b [Gf] {
        assert!(y < self.rows);
        // SAFETY: in bounds by the assertion; `MaybeUninit<Gf>` has `Gf`'s
        // layout and the row is initialised per the contract.
        unsafe { std::slice::from_raw_parts(self.ptr.add(y * self.cols).cast::<Gf>(), self.cols) }
    }
}

/// Transposes the eight 8×8 bit blocks of `w` in place, one per byte lane:
/// afterwards bit `i` of byte `k` of `w[j]` is bit `8k + j` of the original
/// `w[i]`, so byte `k` of `w[j]` is the pattern of column `8k + j`. Twelve
/// delta swaps (4-, 2-, then 1-bit sub-blocks).
#[inline]
pub(crate) fn transpose_blocks(w: &mut [u64; 8]) {
    for i in 0..4 {
        let t = ((w[i] >> 4) ^ w[i + 4]) & 0x0F0F_0F0F_0F0F_0F0F;
        w[i + 4] ^= t;
        w[i] ^= t << 4;
    }
    for i in [0, 1, 4, 5] {
        let t = ((w[i] >> 2) ^ w[i + 2]) & 0x3333_3333_3333_3333;
        w[i + 2] ^= t;
        w[i] ^= t << 2;
    }
    for i in [0, 2, 4, 6] {
        let t = ((w[i] >> 1) ^ w[i + 1]) & 0x5555_5555_5555_5555;
        w[i + 1] ^= t;
        w[i] ^= t << 1;
    }
}

/// The patterns of a group's 64 columns in transposed order: position
/// `8j + k` holds column `8k + j` ([`kernels::col_of`]). Consumes `words`.
#[inline]
fn transposed_patterns(words: &mut [u64; 8]) -> [u8; 64] {
    transpose_blocks(words);
    let mut out = [0u8; 64];
    for (j, w) in words.iter().enumerate() {
        out[8 * j..8 * j + 8].copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// The column weights regrouped the way the transposed pattern blocks
/// visit them: entry `64g + m` is `eq_c[64g + col_of(m)]`, zero past the
/// last column.
fn transposed_eq(eq_c: &[Gf]) -> Vec<Gf> {
    let groups = eq_c.len().div_ceil(64);
    let mut out = vec![Gf::zero(); groups << 6];
    for g in 0..groups {
        for m in 0..64 {
            let c = (g << 6) | kernels::col_of(m);
            if c < eq_c.len() {
                out[(g << 6) | m] = eq_c[c];
            }
        }
    }
    out
}

/// Bit-slices `words` (one 64-column word per pattern bit, at most 8) into
/// one pattern per column: `out[c]` bit `i` = bit `c` of `words[i]`. Eight
/// 8×8 bit-block transposes (delta swaps), one per byte lane.
fn patterns(words: &[u64], out: &mut [u8; 64]) {
    debug_assert!(words.len() <= 8);
    // One or two words: walking the set bits (~32 per word) beats eight
    // block transposes; from four words on the transposes win.
    if words.len() <= 2 {
        *out = [0u8; 64];
        for (i, &w) in words.iter().enumerate() {
            let mut x = w;
            while x != 0 {
                let c = x.trailing_zeros() as usize;
                out[c] |= 1 << i;
                x &= x - 1;
            }
        }
        return;
    }
    for k in 0..8 {
        // Row i of the block = byte k of words[i] (rows past `words` are 0).
        let mut block = 0u64;
        for (i, &w) in words.iter().enumerate() {
            block |= ((w >> (8 * k)) & 0xFF) << (8 * i);
        }
        let t = transpose8x8(block);
        for j in 0..8 {
            out[8 * k + j] = ((t >> (8 * j)) & 0xFF) as u8;
        }
    }
}

/// Transposes the 8×8 bit matrix whose row `i` is byte `i` of `x` (bit `j`
/// of that byte = element `(i, j)`): the result's byte `j` holds column `j`,
/// with element `(i, j)` at bit `i`.
#[inline]
fn transpose8x8(mut x: u64) -> u64 {
    // Swap 1×1 elements across the diagonal within each 2×2 block, then
    // 2×2 blocks within 4×4, then 4×4 within 8×8. Row-major bit (8i + j)
    // and column-major bit (8j + i) differ by 7·(j − i) at each scale.
    let mut t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x ^= t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
    x ^= t ^ (t << 28);
    x
}

#[cfg(test)]
#[path = "forest_tests.rs"]
mod parity_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_patterns(words: &[u64]) -> [u8; 64] {
        let mut out = [0u8; 64];
        for (i, &w) in words.iter().enumerate() {
            for c in 0..64 {
                out[c] |= (((w >> c) & 1) as u8) << i;
            }
        }
        out
    }

    #[test]
    fn block_transpose_matches_the_bit_loop() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..64 {
            let mut words = [0u64; 8];
            for w in words.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *w = state;
            }
            let want = naive_patterns(&words);
            let got = transposed_patterns(&mut words.clone());
            for m in 0..64 {
                assert_eq!(got[m], want[kernels::col_of(m)], "position {m}");
            }
        }
    }

    #[test]
    fn transposed_patterns_match_the_bit_loop() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for len in 1..=8 {
            for _ in 0..64 {
                let words: Vec<u64> = (0..len)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        state
                    })
                    .collect();
                let mut out = [0u8; 64];
                patterns(&words, &mut out);
                assert_eq!(out, naive_patterns(&words), "len {len}");
            }
        }
    }
}
