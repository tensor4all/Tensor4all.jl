# Design Documents

This directory contains the active design set for the `Tensor4all.jl` rework.

The current implementation state is intentionally smaller than the scope
described here. For the staged follow-up work that is deferred beyond the reset,
see [../plans/2026-04-10-tensor4all-rework-followup.md](../plans/2026-04-10-tensor4all-rework-followup.md).

The Julia frontend design is split into a hub-and-spoke set:

- [julia_ffi.md](./julia_ffi.md) for the overview and index
- [julia_ffi_core.md](./julia_ffi_core.md) for low-level primitives
- [julia_ffi_tt.md](./julia_ffi_tt.md) for backend TT support
- [julia_ffi_quantics.md](./julia_ffi_quantics.md) for quantics grids and transforms
- [bubbleteaCI.md](./bubbleteaCI.md) for reusable `TTFunction` logic and migration
- [julia_ffi_extensions.md](./julia_ffi_extensions.md) for compatibility and extension glue
- [julia_ffi_roadmap.md](./julia_ffi_roadmap.md) for the implementation plan
