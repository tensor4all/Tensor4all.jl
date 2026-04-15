# Julia-Native Execution Redesign

> Status: approved redesign baseline as of 2026-04-15.
>
> This document supersedes the earlier "POC", "skeleton", and
> "metadata-only placeholder" framing for the active Julia frontend work.

## Goal

Move the branch from a review-first placeholder phase into an implementation
phase where the public Julia APIs are expected to perform real work, and where
tests are translated from `tensor4all-rs` into Julia-native contracts.

## Scope

The redesign focuses on the modules that currently have the largest semantic
gap between their public surface and their tested behavior:

- `Tensor4all.TensorNetworks`
- `Tensor4all.QuanticsTransform`
- supporting documentation and test structure

The following modules remain intentionally lighter-weight:

- `Tensor4all.TensorCI`
- `Tensor4all.QuanticsTCI`
- `Tensor4all.QuanticsGrids`

For those re-export or boundary modules, surface and smoke coverage remains
acceptable unless a concrete regression suggests a deeper Julia-side contract.

## Problem Statement

The current branch still mixes two conflicting stories:

- source and docs increasingly treat `Index`, `Tensor`, HDF5 interop, and
  parts of the chain layer as real implementations
- README, AGENTS guidance, and many tests still frame the package as a POC or
  skeleton whose primary guarantee is that names exist

That mismatch is now the main risk. It weakens review quality, encourages
placeholder implementations to linger, and leaves Julia behavior significantly
under-tested relative to `tensor4all-rs`.

## Design Principles

### 1. Public Julia API stays Julia-native

Julia users should work with Julia collections and Julia data conventions.
Internal Rust/C representations may be more constrained, but those constraints
must be normalized behind the public API.

Examples:

- affine transforms should accept `AbstractMatrix` and `AbstractVector`
  semantics, not flattened coefficient buffers as the primary public interface
- Fourier configuration should use Julia keyword arguments, not a Rust-mirror
  options struct as the required user-facing type
- boundary conditions should remain Julia symbols or similarly idiomatic values
  at the public boundary

### 2. Rust is the semantic reference, not the public API template

`tensor4all-rs` is the source of truth for operator semantics and edge-case
behavior. Its tests should be translated into Julia tests, but the Julia API
should not blindly expose Rust-shaped helper types when a simpler Julia
signature is natural.

### 3. Exported APIs should be executable by default

If an exported function or type is documented as public and general-purpose,
the default expectation should be that it works. Placeholder-only exports are
no longer the preferred steady state.

`SkeletonNotImplemented` should shrink to genuinely temporary or backend-blocked
corners, not define the main user story for `TensorNetworks` and
`QuanticsTransform`.

### 4. Test contracts should be numeric where semantics are numeric

For operator constructors and chain helpers, existence tests are insufficient.
The Julia suite should include:

- constructor validation tests
- shape and index-layout tests
- dense or basis-state numeric equivalence tests
- end-to-end `apply` tests where the operator is meant to execute

## Considered Approaches

### Approach A: Keep the current public API and only deepen tests

Pros:

- smallest short-term diff
- preserves all existing names

Cons:

- locks in placeholder-first API design
- keeps `QuanticsTransform` semantically underspecified
- continues to reward "names exist" testing over numerical behavior

This is not recommended.

### Approach B: Mirror Rust public helper types in Julia

Pros:

- Rust tests translate mechanically
- internal implementation may be straightforward

Cons:

- exposes Rust-shaped parameter packing to Julia users
- weakens ergonomics for matrices, vectors, and keywords
- creates avoidable duplication between Julia-native values and mirror structs

This is not recommended as the primary public design.

### Approach C: Julia-native public API with Rust-aligned semantics

Pros:

- natural Julia call patterns
- semantics remain anchored to `tensor4all-rs`
- tests can be translated by meaning rather than by syntax
- keeps C-API normalization internal

Cons:

- requires a deliberate normalization layer in Julia
- Rust test translation is less mechanical

This is the recommended design.

## Approved Public Direction

### TensorNetworks

`TensorNetworks` should become a genuinely usable indexed chain layer.

Required direction:

- keep `TensorTrain` as `Vector{Tensor} + llim/rlim`
- keep `LinearOperator` owned by `TensorNetworks`
- implement the current helper names instead of treating them as permanent
  stubs
- implement executable operator application where the backend path exists

The intended helper surface includes:

- `findsite`
- `findsites`
- `findallsiteinds_by_tag`
- `findallsites_by_tag`
- `replace_siteinds!`
- `replace_siteinds`
- `replace_siteinds_part!`
- `rearrange_siteinds`
- `makesitediagonal`
- `extractdiagonal`
- `matchsiteinds`
- `apply`

These functions should be validated against small, explicit `TensorTrain`
examples with deterministic index layouts.

### QuanticsTransform

`QuanticsTransform` should expose Julia-native constructors that return
executable `TensorNetworks.LinearOperator` values, not metadata-only
placeholders.

Preferred public signatures:

- `shift_operator(r, offset; bc=:periodic)`
- `shift_operator_multivar(r, offset, nvars, target; bc=:periodic)`
- `flip_operator(r; bc=:periodic)`
- `flip_operator_multivar(r, nvars, target; bc=:periodic)`
- `phase_rotation_operator(r, theta)`
- `phase_rotation_operator_multivar(r, theta, nvars, target)`
- `cumsum_operator(r)`
- `fourier_operator(r; direction=:forward, maxbonddim=typemax(Int), tolerance=0.0)`
- `affine_operator(r, A, b; bc=:periodic)`
- `affine_pullback_operator(r, A, b; bc=:periodic)`
- `binaryop_operator(r, a1, b1, a2, b2; bc1=:periodic, bc2=:periodic)`

Internal normalization may still pack values into Rust/C-compatible forms, but
that normalization should not be the primary user contract.

### Internal Parameter Normalization

To keep the Julia API idiomatic without duplicating backend logic, introduce an
internal normalization layer that:

- validates matrix and vector shapes
- normalizes `bc` into a per-dimension representation
- converts Julia `Int` / `Rational` / floating-point values into the exact
  scalar buffers required by the C API
- centralizes boundary-condition parsing and error messages

This layer may use internal helper structs, but those helpers do not need to be
exported.

## Test Translation Strategy

### 1. Translate Rust constructor-validation tests into Julia unit tests

Rust module tests under:

- `crates/tensor4all-quanticstransform/src/shift/tests/mod.rs`
- `crates/tensor4all-quanticstransform/src/flip/tests/mod.rs`
- `crates/tensor4all-quanticstransform/src/phase_rotation/tests/mod.rs`
- `crates/tensor4all-quanticstransform/src/cumsum/tests/mod.rs`
- `crates/tensor4all-quanticstransform/src/fourier/tests/mod.rs`
- `crates/tensor4all-quanticstransform/src/affine/tests/mod.rs`

should be translated into Julia tests that check:

- accepted inputs
- rejected inputs
- boundary-condition validation
- keyword and shape validation

### 2. Translate Rust numeric operator tests into Julia dense-reference tests

Rust integration and C-API tests include matrix-equivalence checks for:

- shift
- flip
- phase rotation
- cumsum
- Fourier
- affine
- affine pullback
- binary operators

Julia should reproduce those checks using small `r` values and dense reference
matrices or basis-vector application. The goal is to verify semantics, not just
construction.

### 3. Replace `TensorNetworks` skeleton tests with behavior tests

Current tests mainly verify that helper names exist and throw.

They should be replaced by tests that verify:

- tag-based site discovery
- index replacement semantics
- physical-leg rearrangement
- diagonalization helpers on small explicit examples
- `LinearOperator` input/output-space binding
- `apply` on small chain states and operators

### 4. Keep HDF5 as a real interop contract

The HDF5 suite should continue to cover:

- Tensor4all roundtrip
- Tensor4all ↔ `ITensorMPS` interoperability
- schema and storage-order expectations that are visible at the Julia layer

### 5. Keep re-export modules light

`TensorCI`, `QuanticsTCI`, and `QuanticsGrids` do not need broad Julia-native
numeric duplication if their main role is wrapping or re-exporting upstream
functionality. Smoke tests plus key boundary tests remain sufficient.

## Documentation Changes Required

The public docs should stop describing the active branch as a POC or skeleton.

Required wording changes:

- `README.md` should describe the package as being in an implementation phase
  with some remaining unimplemented areas, not as a proof of concept
- `AGENTS.md` should stop instructing authors to describe exported APIs as
  deferred placeholders by default
- docs under `docs/src/` and `docs/design/` should distinguish between
  implemented, partially implemented, and not-yet-implemented surfaces without
  centering the whole repository story on placeholders

## File Organization Direction

The redesign should follow the file-size guidance already recorded in
`AGENTS.md`.

Likely split points:

- `src/QuanticsTransform/`
  - constructor front-end
  - parameter normalization
  - materialization helpers
- `src/TensorNetworks/`
  - chain helper algorithms
  - operator application
  - HDF5 interop
- `test/quanticstransform/`
  - constructor validation
  - numeric equivalence
- `test/tensornetworks/`
  - helper semantics
  - operator application

## Non-Goals

- redesigning `TensorCI` into a Julia-owned numeric engine
- adding deep numeric duplication for `QuanticsTCI` re-exports
- broad tree-tensor-network redesign in this phase
- changing `tensor4all-rs` itself from this repository

## Success Criteria

The redesign is complete when all of the following are true:

- README and AGENTS no longer present the active branch as a POC or skeleton
- `QuanticsTransform` constructors return executable operators with Julia-native
  signatures
- `TensorNetworks` helper tests verify real behavior, not only thrown errors
- translated Rust tests cover constructor validation and small numeric
  equivalence cases in Julia
- `TensorCI` and `QuanticsTCI` remain lightly tested by design rather than by
  omission
