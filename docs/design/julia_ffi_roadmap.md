# Julia Frontend Implementation Roadmap

## Purpose

This document gives the phase order for the Julia frontend and the `BubbleTeaCI` migration.

## Phase 1. Low-Level Foundation

- wrap `Index` and `Tensor`
- implement lifecycle, metadata, and basic contraction
- add smoke tests for the foundational layer

Exit criteria:
- stable low-level primitives are available for higher layers.

## Phase 2. Core Backend Coverage for TT Workflows

- add `TensorTrain`
- surface TT-level contraction/compression
- expose TT transforms and backend TT helpers

Exit criteria:
- the backend can support higher-level TT workflow migration.

## Phase 3. Quantics and Backend-Neutral Interfaces

- adopt and re-export `QuanticsGrids.jl` for quantics grids and coordinate conversion
- settle public names for `TTFunction` / `QTTFunction`
- define the backend-neutral shape of TT-like operations
- fix the downstream `BubbleTeaCI` contract before migration phases begin

Exit criteria:
- the Julia frontend knows what the higher-level layer will target, and the adopted quantics boundary plus downstream `BubbleTeaCI` contract are explicit.

## Phase 4. Separate the Reusable BubbleTeaCI Core

- isolate generic `TTFunction`-related code from application modules
- keep ladder-DGammaA and similar workflows outside the reusable core
- preserve behavior with tests while reshaping modules

Exit criteria:
- the reusable core is cleanly separated from application code.

## Phase 5. Migrate the Reusable High-Level Layer

- retarget reusable `TTFunction` logic to the new backend
- preserve evaluation, contraction, integration, embedding, and transform behavior
- close any remaining frontend gaps required by the migrated code

Exit criteria:
- the reusable high-level layer no longer depends on `ITensors.jl` internals.

## Phase 6. Cleanup and Application Migration

- improve naming, readability, and diagnostics
- harden error handling and user ergonomics
- migrate application modules onto the refactored core
- settle whether `BubbleTeaCI` should later re-export a curated `Tensor4all.jl` subset as part of stabilization, not as part of the early skeleton

Exit criteria:
- the migrated ecosystem is stable enough for day-to-day use.

## Phase 7. Tutorial, Validation, and Performance Closure

- add the expanded beginner tutorial
- complete the HDF5 compatibility layer and round-trip tests
- complete compatibility and round-trip tests
- add parity tests for representative BubbleTeaCI workflows
- profile for any remaining Rust-side kernels that are truly needed

Exit criteria:
- the frontend is documented, validated, and ready for sustained use.

## Dependency Order

- Phase 1 must finish before Phase 2.
- Phase 2 and Phase 3 should be in place, including the explicit downstream `BubbleTeaCI` contract, before the reusable BubbleTeaCI core is refactored.
- Phase 4 must finish before the migration in Phase 5.
- Phase 6 and Phase 7 are primarily stabilization and completion work after the migration path is clear.
