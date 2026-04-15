# Deferred Rework Plan

The broader rework is intentionally split.

The repository is now in the implementation phase of the restored Julia
frontend. Backend numerics and downstream migration work are still incomplete,
but the public architecture is no longer described as an early-stage surface.

The active execution plan remains tracked in:

- [2026-04-10 tensor4all rework follow-up plan](https://github.com/tensor4all/Tensor4all.jl/blob/tensor4all-rework/docs/plans/2026-04-10-tensor4all-rework-followup.md)

## Still Deferred

- backend-backed tensor contraction, factorization, and dense materialization
- transform materialization and QTCI execution
- C API expansion where the Julia implementation still reveals missing
  multi-language primitives
- downstream `BubbleTeaCI` migration onto the new public surface
- beginner-facing tutorials beyond the current review-first documentation

This split remains deliberate: the next implementation wave should focus on
backend enablement and downstream migration, not on widening the public surface
without review.
