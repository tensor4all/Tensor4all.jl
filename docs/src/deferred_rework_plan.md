# Deferred Rework Plan

The broader rework is intentionally split. Phase 0 only resets the repository
and sets up a review-first docs surface.

The deferred implementation work is tracked in:

- [2026-04-10 tensor4all rework follow-up plan](https://github.com/tensor4all/Tensor4all.jl/blob/tensor4all-rework/docs/plans/2026-04-10-tensor4all-rework-followup.md)

## Deferred Until After Review

- Core skeleton for `Index` and `Tensor`
- TreeTN-general tensor network skeleton with `TensorTrain`, `MPS`, and `MPO` aliases
- Quantics grids, transforms, and QTCI skeleton types
- ITensors and HDF5 extension skeletons
- staged commit and review cadence for the real API skeleton
- expanded smoke tests and later API-reference pages

This split is deliberate: the next wave of implementation should land only after
the architecture and naming choices have been reviewed.
