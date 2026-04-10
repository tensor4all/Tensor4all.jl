# Julia Frontend Design for `tensor4all-rs`

## Overview

This directory now uses a hub-and-spoke structure. The Julia frontend owns backend-facing primitives and extension glue, `tensor4all-rs` owns kernels and storage, and the reusable high-level `TTFunction` / `GriddedFunction` logic stays in `BubbleTeaCI`.

This file is only the index. The detailed design lives in the sibling documents linked below.

## Doc Map

| File | Purpose |
|------|---------|
| [julia_ffi_core.md](./julia_ffi_core.md) | `Index`, `Tensor`, ownership, low-level FFI, base Julia APIs |
| [julia_ffi_tt.md](./julia_ffi_tt.md) | backend `TensorTrain` support and TT-level operations |
| [julia_ffi_quantics.md](./julia_ffi_quantics.md) | grids, layouts, coordinates, quantics transforms, multiresolution |
| [bubbleteaCI.md](./bubbleteaCI.md) | reusable `TTFunction` / `GriddedFunction` layer and migration plan |
| [julia_ffi_extensions.md](./julia_ffi_extensions.md) | ITensors/HDF5 compatibility and package extension boundary |
| [julia_ffi_roadmap.md](./julia_ffi_roadmap.md) | phased implementation order and dependencies |

## Ownership Model

- `tensor4all-rs` owns performance-critical kernels, storage, and numerics.
- The Julia frontend owns low-level wrappers, backend-facing abstractions, and extension glue.
- `BubbleTeaCI` owns the reusable high-level `TTFunction` logic and the application code built on top of it.

## Cross-Cutting Questions

- What is the public naming for the high-level function abstraction: `TTFunction`, `QTTFunction`, or an aliasing pair?
- How should we normalize quantics layouts internally while keeping user-facing construction flexible?
- What is the clean boundary between the absorbed raw `TensorTrain` layer and the higher-level `TTFunction` logic in `BubbleTeaCI`?
- Which compatibility and conversion policies belong in Julia extensions versus the core frontend?

## Roadmap Pointer

See [julia_ffi_roadmap.md](./julia_ffi_roadmap.md) for the implementation phases and dependency order.
