# Tensor4all.jl

Julia frontend for [tensor4all-rs](https://github.com/tensor4all/tensor4all-rs).

## Current Phase

This repository is in a review-first skeleton phase.

The previous implementation was intentionally cleared so the Julia package could
be rebuilt around the design set in `docs/design/` without carrying stale or
inconsistent APIs forward.

The package now exposes a reviewed skeleton surface:

- metadata-level `Index` and `Tensor` types
- `TreeTensorNetwork` plus `TensorTrain`, `MPS`, and `MPO` aliases
- a curated `QuanticsGrids.jl` re-export for single-import quantics grid usage
- local quantics transform and QTCI placeholder types
- extension-only ITensors and HDF5 compatibility stubs

Backend numerics are still intentionally stubbed. Public APIs that are not yet
implemented should fail with actionable skeleton exceptions rather than silently
pretending to do work.

## What To Read

- [docs/design/README.md](docs/design/README.md)
- [docs/design/julia_ffi.md](docs/design/julia_ffi.md)
- [docs/plans/2026-04-10-tensor4all-rework-followup.md](docs/plans/2026-04-10-tensor4all-rework-followup.md)

## Development Notes

- `QuanticsGrids.jl` remains the owner of grid semantics and coordinate
  conversion. `Tensor4all.jl` adopts and re-exports a curated subset for
  usability.
- `BubbleTeaCI` remains the home of `TTFunction` and high-level workflows. It
  should build on `Tensor4all.jl` rather than duplicating lower-level
  functionality.
- The active docs site is review-oriented and includes an API reference for the
  current skeleton surface.
- Future implementation work is focused on backend enablement and downstream
  migration, not on growing a second high-level function layer here.

## Build Script

The backend build script remains in place for later phases. It looks for
`tensor4all-rs` in this order:

1. `TENSOR4ALL_RS_PATH` environment variable
2. sibling directory `../tensor4all-rs/`
3. clone from GitHub at the pinned fallback commit in [deps/build.jl](deps/build.jl)

If you run the build script directly, use the package project:

```bash
julia --startup-file=no --project=. deps/build.jl
```

## Smoke Checks

```julia
using Pkg
Pkg.activate("/path/to/Tensor4all.jl")
Pkg.test()
```

The current test suite verifies the skeleton metadata layers, the adopted
quantics re-export, and the extension boundaries without requiring working
backend numerics.
