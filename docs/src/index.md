# Tensor4all.jl

> Current phase: skeleton review / implementation reset.
>
> Old implementation removed intentionally.
>
> API layers will land only after review.

`Tensor4all.jl` is being reset around the design documents in `docs/design/`.
The previous implementation was broad but no longer matched the planned Julia
frontend architecture closely enough to serve as a safe base for future work.

## What Exists Right Now

- a minimal loadable `Tensor4all` module
- a review-first documentation site
- the imported design set under `docs/design/`
- a deferred rework plan for the next implementation stages

## What Has Been Removed For This Phase

- the previous `src/` implementation surface
- the previous `test/` suite
- old API-reference pages and tutorial pages that described behavior no longer present

## Review Entry Points

- [Architecture Status](modules.md)
- [Design Documents](design_documents.md)
- [Deferred Rework Plan](deferred_rework_plan.md)

## Ownership Boundary

- `tensor4all-rs` owns kernels, storage, and performance-critical numerics
- `Tensor4all.jl` will own Julia-side wrappers, TreeTN-general abstractions, and extension glue
- `BubbleTeaCI` owns the reusable `TTFunction` / `GriddedFunction` layer and application workflows

## Immediate Goal

Use this phase to review the architecture, naming, layering, and documentation
before reintroducing any real backend-facing APIs.

## Bootstrap API

```@docs
Tensor4all
Tensor4all.SkeletonPhaseError
```
