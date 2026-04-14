# BubbleTeaCI High-Level QTT Layer

## Purpose

This document is the home of the reusable `TTFunction` / `GriddedFunction` logic. That logic stays in `BubbleTeaCI`; it is not duplicated in the Julia frontend docs.

`BubbleTeaCI` is the migration target for the reusable high-level layer, which should move away from `ITensors.jl` internals and onto the new `tensor4all-rs`-based frontend.

## In Scope

- reusable `TTFunction` / `GriddedFunction` design
- variable-aware contraction APIs
- integration and reduction
- embedding, interpolation, and related multiresolution helpers
- cleanup and stabilization
- beginner tutorial and examples
- migration of reusable code away from `ITensors.jl`
- separation of generic functionality from application code

## Reusable Core and Application Boundary

The reusable parts of `BubbleTeaCI` should be clearly separated from application-specific physics code.

### Reusable Core

- `diagramcomponent`
- `contract_order`
- `grid_operations`
- `affinetransform`
- `ttfunction_interpol_avg`
- the generic pieces of `newgrid`
- any shared `TTFunction` helpers that are not application-specific

### Application Code

- `ladder_DGammaA`
- `BSE`
- `SDE`
- `dyson`
- `Keldysh_PT2`
- other workflow-specific modules that depend on the reusable core

## `TTFunction` Responsibilities

- represent scalar-, vector-, and matrix-valued gridded functions
- track grid metadata together with tensor-train data
- support evaluation, slicing, and collection
- support arithmetic, dot, norm, and truncation
- support variable-aware contraction by names or positions
- support integration, reduction, and weighted volume-element behavior
- support embedding and interpolation on quantics grids

## Cleanup Goals

- decouple the reusable high-level layer from concrete `ITensors.jl` storage types
- improve naming, error handling, and readability
- make grid, layout, and component-leg semantics easier to inspect
- define a stable public surface for the generic `TTFunction` layer

## Tutorial and Onboarding

A beginner-oriented tutorial should live here as part of the reusable layer.

- Use `BubbleTeaCI/scripts/examples/basic_operations.jl` as the starting point.
- Expand it with more explanation and more examples.
- Cover grids, compression, evaluation, contraction orders, transforms, and integration.
- Make the examples runnable so they can also act as regression tests.

## Migration Plan

- Keep the reusable `TTFunction` logic in `BubbleTeaCI`.
- Retarget the reusable layer from `ITensors.jl` to the new Julia frontend.
- Keep application code separate while the core is migrated.
- Preserve behavior with parity tests during the transition.

## Dependency and Re-Export Strategy

- `BubbleTeaCI` should consume lower-level tensor-network and grid functionality from `Tensor4all.jl` and its adopted dependencies rather than duplicating them locally.
- `BubbleTeaCI` may later re-export a curated subset of `Tensor4all.jl` for single-import usability in high-level workflows.
- Such re-export should stay curated and documented. Ownership of lower-level APIs remains with `Tensor4all.jl` or the adopted dependency that originally provides them.
- `TTFunction`, `GriddedFunction`, and other high-level workflow semantics remain owned by `BubbleTeaCI` even if lower-level functionality is re-exported.

## Open Questions

- Should the public abstraction be `TTFunction`, `QTTFunction`, or one name with an alias?
- Should the reusable core eventually be split into its own package after stabilization, or remain within `BubbleTeaCI`?
- Which migration helpers should be implemented first to reduce churn for downstream applications?
- Should `BubbleTeaCI` later re-export all of `Tensor4all.jl` or a curated subset tailored to the high-level workflow story?
