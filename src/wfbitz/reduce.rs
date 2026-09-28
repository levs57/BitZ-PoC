//! Their grand-product reduction (step 4) to a factored inner-product claim
//! over the committed bits: each leaf is `1 + (y_row − 1)·bit(column, row)`,
//! and at GKR's terminal point subtracting one leaves
//! `Σ u1[row]·u2[column]·bit(column, row)`.

use super::fold::Fold;
use super::forest::Forest;
use super::gkr::gpgkr_verify;
use super::params::{ClaimError, LinearClaimGf, Shape};
use super::pcs::OpeningQuery;
use super::transcript::{ProverState, VerifierState};
use crate::ligerito_flock::FlockCommitHint;
use field::Gf128 as Gf;
use crate::poly::utils::build_eq_x_r_vec;

/// A reduction the verifier rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceError {
    /// The GKR replay failed.
    GKR,
    /// The derived weight counts do not match the table shape.
    Claim(ClaimError),
}

fn eq_table(point: &[Gf]) -> Vec<Gf> {
    build_eq_x_r_vec(point, &()).expect("non-empty point")
}

/// Splits the terminal point into column and row coordinates (columns occupy
/// the low index bits) and derives the two weight factors.
fn query_from_terminal(
    fold: &Fold,
    shape: &Shape,
    point: Vec<Gf>,
    claim: Gf,
) -> Result<OpeningQuery, ClaimError> {
    let (u1, alfa_c, inner_product_claim) = exit_from_terminal(fold, point, claim);
    let u2 = eq_table(&alfa_c);
    let claim = LinearClaimGf::from_shape(shape, u1, u2, inner_product_claim)?;
    Ok(OpeningQuery::InnerProduct { claim })
}

/// The GKR's exit split the way a composition consumes it: the row-bit
/// weights `(A(b) − 1)·eq(b, α_b)`, the column point `α_c` and the
/// inner-product target `claim − 1` (the multilinear extension of the
/// constant-one table is one everywhere). GKR's terminal point is
/// `[α_c (r2 column vars) | α_b (r1 row vars)]`.
pub(crate) fn exit_from_terminal(fold: &Fold, mut point: Vec<Gf>, claim: Gf) -> (Vec<Gf>, Vec<Gf>, Gf) {
    let inner_product_claim = claim - Gf::one();
    let r1 = fold.row_images.len().max(1).ilog2() as usize;
    let r2 = point.len() - r1;
    let alfa_b = point.split_off(r2);
    let alfa_c = point;
    let u1: Vec<Gf> = fold
        .row_images
        .iter()
        .zip(eq_table(&alfa_b))
        .map(|(a, b)| (*a - Gf::one()) * b)
        .collect();
    (u1, alfa_c, inner_product_claim)
}

/// The fold's GKR over any grid's packed columns, its exit in the
/// composition's form (see [`exit_from_terminal`]).
pub(crate) fn gkr_exit_prove(
    transcript: &mut ProverState,
    fold: &Fold,
    shape: &Shape,
    packed_cols: &[Vec<u64>],
) -> (Vec<Gf>, Vec<Gf>, Gf) {
    let forest = Forest::new(
        shape.log_rows(),
        shape.log_columns(),
        packed_cols,
        &fold.row_images,
    );
    let (point, claim) = forest.prove(transcript, &fold.zeta);
    exit_from_terminal(fold, point, claim)
}

/// [`gkr_exit_prove`]'s verifier.
pub(crate) fn gkr_exit_verify(
    transcript: &mut VerifierState<'_>,
    fold: &Fold,
    shape: &Shape,
) -> Result<(Vec<Gf>, Vec<Gf>, Gf), ReduceError> {
    let (point, claim) = gpgkr_verify(transcript, fold.e0, &fold.zeta, shape.log_rows() as u32)
        .ok_or(ReduceError::GKR)?;
    Ok(exit_from_terminal(fold, point, claim))
}

pub(crate) fn gkr_reduce_prove(
    transcript: &mut ProverState,
    fold: &Fold,
    shape: &Shape,
    hint: &FlockCommitHint,
) -> Result<OpeningQuery, ClaimError> {
    gkr_reduce_prove_packed(transcript, fold, shape, hint.packed_cols())
}

#[cfg(feature = "bench-internals")]
pub(crate) fn gkr_reduce_prove_with_cached_rounds(
    transcript: &mut ProverState,
    fold: &Fold,
    shape: &Shape,
    hint: &FlockCommitHint,
    cached_rounds: bool,
) -> Result<OpeningQuery, ClaimError> {
    let forest = Forest::new(
        shape.log_rows(),
        shape.log_columns(),
        hint.packed_cols(),
        &fold.row_images,
    );
    let (point, claim) = forest.prove_with_cached_rounds(transcript, &fold.zeta, cached_rounds);
    query_from_terminal(fold, shape, point, claim)
}

/// [`gkr_reduce_prove`] over any grid's 64-lane packed columns
/// (`packed_cols[g][b]` = bit `b` of columns `64g..64g+63`): the committed
/// rows' packing, or a derived grid's for a virtual opening.
pub(crate) fn gkr_reduce_prove_packed(
    transcript: &mut ProverState,
    fold: &Fold,
    shape: &Shape,
    packed_cols: &[Vec<u64>],
) -> Result<OpeningQuery, ClaimError> {
    let forest = Forest::new(
        shape.log_rows(),
        shape.log_columns(),
        packed_cols,
        &fold.row_images,
    );
    let (point, claim) = forest.prove(transcript, &fold.zeta);
    query_from_terminal(fold, shape, point, claim)
}

pub(crate) fn gkr_reduce_verify(
    transcript: &mut VerifierState<'_>,
    fold: &Fold,
    shape: &Shape,
) -> Result<OpeningQuery, ReduceError> {
    let (point, claim) = gpgkr_verify(transcript, fold.e0, &fold.zeta, shape.log_rows() as u32)
        .ok_or(ReduceError::GKR)?;
    query_from_terminal(fold, shape, point, claim).map_err(ReduceError::Claim)
}
