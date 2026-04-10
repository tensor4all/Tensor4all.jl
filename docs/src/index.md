# Tensor4all.jl

> Current phase: skeleton review / implementation reset.
>
> Old implementation removed intentionally.
>
> Public metadata layers now exist for review, while backend numerics remain stubbed.

`Tensor4all.jl` is being reset around the design documents in `docs/design/`.
The previous implementation was broad but no longer matched the planned Julia
frontend architecture closely enough to serve as a safe base for future work.

## What Exists Right Now

- a loadable `Tensor4all` module with reviewable metadata types
- `Index`, `Tensor`, and `TreeTensorNetwork` skeletons
- adopted and re-exported quantics grid functionality from `QuanticsGrids.jl`
- local quantics transform and QTCI placeholder types
- extension-only ITensors and HDF5 compatibility stubs
- a review-first documentation site
- the imported design set under `docs/design/`
- an execution plan documenting remaining backend-facing work

## What Has Been Removed For This Phase

- stale pre-reset APIs that no longer matched the planned architecture
- old behavior claims that implied numerics were already implemented
- high-level `TTFunction` functionality that belongs in `BubbleTeaCI`, not here

## Review Entry Points

- [Architecture Status](modules.md)
- [API Reference](api.md)
- [Design Documents](design_documents.md)
- [Deferred Rework Plan](deferred_rework_plan.md)

## Ownership Boundary

- `tensor4all-rs` owns kernels, storage, and performance-critical numerics
- `Tensor4all.jl` owns Julia-side wrappers, TreeTN-general abstractions, quantics integration, and extension glue
- `QuanticsGrids.jl` owns grid semantics and coordinate conversion; `Tensor4all.jl` adopts and re-exports a curated subset for single-import usability
- `BubbleTeaCI` owns the reusable `TTFunction` / `GriddedFunction` layer and application workflows, and should build on lower layers instead of duplicating them

## Immediate Goal

Use this phase to review the architecture, naming, layering, and documentation
of the new skeleton surface before backend numerics are wired in.

## Skeleton API Snapshot

See the [API Reference](api.md) for the full review surface. The homepage stays
focused on architecture and package boundaries so it does not duplicate the
reference listings.
