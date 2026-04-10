# Tensor4all.jl

Julia frontend for [tensor4all-rs](https://github.com/tensor4all/tensor4all-rs).

## Current Phase

This repository is in a skeleton-review reset.

The previous implementation has been intentionally removed so the Julia package
can be rebuilt around the design set in `docs/design/` without carrying stale or
inconsistent APIs forward. The package currently provides only a minimal
bootstrap module and review-first documentation.

## What To Read

- [docs/design/README.md](docs/design/README.md)
- [docs/design/julia_ffi.md](docs/design/julia_ffi.md)
- [docs/plans/2026-04-10-tensor4all-rework-followup.md](docs/plans/2026-04-10-tensor4all-rework-followup.md)

## Development Notes

- `src/` and `test/` have been reset on purpose.
- The old API reference and tutorial pages have been removed from the active docs site.
- Future implementation work is intentionally deferred until the architecture has been reviewed.

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

The current test suite is intentionally tiny and only verifies that the package
loads and exposes the reset-phase markers.
