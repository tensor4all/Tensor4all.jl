# Deferred Rework Plan

The broader rework is intentionally split.

The repository is now in the implementation phase of the restored Julia
frontend. Backend numerics and downstream migration work are still incomplete,
but the public architecture is no longer described as an early-stage surface.

The repository has now moved beyond the pure reset step:

- `Core.Index` and `Core.Tensor` are backend-backed wrappers
- HDF5 persistence lives in `TensorNetworks.save_as_mps` / `load_tt`
- HDF5 compatibility is checked both for Tensor4all roundtrip and
  Tensor4all/ITensorMPS interoperability

Remaining work is now concentrated in the still-deferred helper and execution
surfaces, plus downstream migration work.

The single implementation handoff plan is tracked in:

- [2026-04-10 tensor4all rework follow-up plan](https://github.com/tensor4all/Tensor4all.jl/blob/main/docs/plans/2026-04-10-tensor4all-rework-followup.md)

## Still Deferred

- the remaining `TensorNetworks` helper implementations that still throw
  missing-implementation errors
- transform materialization and QTCI execution
- C API expansion where the Julia implementation still reveals missing
  multi-language primitives
- downstream `BubbleTeaCI` migration onto the new public surface
- beginner-facing tutorials beyond the current review-first documentation

This split remains deliberate: the next implementation wave should focus on
backend enablement and downstream migration, not on widening the public surface
without review.
