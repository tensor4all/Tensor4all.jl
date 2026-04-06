# Quantics Rust Parity Design

## Goal

Make `Tensor4all.jl` expose the existing pure Rust quantics functionality through the C API and Julia wrapper, instead of rebuilding a legacy `TensorCI` compatibility layer. The immediate downstream target is `ReFrequenTT`, which needs grids, multivariable transforms, dimension embedding, and mixed boundary conditions more than it needs the old `TensorCI2` surface.

## Decision

Adopt `tensor4all-rs` as the source of truth for quantics functionality and make `Tensor4all.jl` a thin, ergonomic wrapper over that surface.

This means:

- Do not revive the old Rust-backed `TensorCI` wrapper.
- Do not make `TensorCrossInterpolation.jl` reexport masquerade as `Tensor4all.TensorCI`.
- Use `SimpleTT.SimpleTensorTrain` as the canonical Julia TT type inside `Tensor4all.jl`.
- Keep interoperability with external TT types in weak-dependency extensions.

---

## A. Why This Direction

Reviving `Tensor4all.TensorCI` would mostly solve naming continuity, but it would not solve the actual migration blockers for `ReFrequenTT`. The missing pieces there are:

- multivariable affine transforms
- dimension-changing embeddings
- per-axis boundary conditions
- a clean path from quantics operators to Julia TT values

Those capabilities belong to the quantics backend surface, not to a legacy `TensorCI2` facade.

By contrast, `QuanticsGrids` is already close to the right architecture: Rust owns the implementation, the C API exposes opaque grid handles and conversion functions, and Julia wraps them. The correct direction is to finish that pattern across the quantics stack.

---

## B. Public API Principles

### 1. Rust/C API parity first

If a capability already exists in `tensor4all-rs`, the Julia package should expose it with minimal semantic drift. Julia convenience is allowed, but the underlying feature boundary should stay aligned with Rust.

### 2. Ergonomic top-level exports are secondary

Top-level exports such as `DiscretizedGrid` or `InherentDiscreteGrid` improve usability, but they are not substitutes for missing backend exposure. Reexports should be treated as API polish, not architecture.

### 3. One canonical TT type inside Tensor4all.jl

`SimpleTT.SimpleTensorTrain` is the internal Julia TT representation that all quantics wrappers should produce and consume where possible.

### 4. Interop lives in extensions

Conversions to external tensor-train ecosystems should stay in weak-dependency extension files:

- `TensorCrossInterpolation.TensorTrain <-> SimpleTensorTrain`
- future `ITensorLike.TensorTrain <-> SimpleTensorTrain`, if a concrete package surface exists

This keeps the core package small and avoids coupling `Tensor4all.jl` to external API churn.

---

## C. QuanticsGrids Scope

`QuanticsGrids` is already C-API-backed, so the work here is parity and cleanup, not reinvention.

Planned direction:

- keep `DiscretizedGrid` and `InherentDiscreteGrid` as opaque Julia wrappers over Rust-owned objects
- expose the full unfolding enum that Rust already supports, including `:grouped`
- make the common grid accessors part of the normal Julia surface
- reexport the most common grid symbols from `Tensor4all` top level for ergonomics

Recommended top-level exports:

- `DiscretizedGrid`
- `InherentDiscreteGrid`
- `localdimensions`
- coordinate conversion helpers only if name collisions remain manageable

Non-goal:

- do not duplicate grid logic in Julia

---

## D. QuanticsTransform Scope

This is the core gap for `ReFrequenTT`.

`tensor4all-rs` already has the relevant operator constructors in `tensor4all-quanticstransform`, including:

- multivariable shift/flip/phase operators
- affine operators
- asymmetric input/output layouts for embedding
- boundary conditions per output variable

The missing piece is exposure through `tensor4all-capi` and then `Tensor4all.jl`.

Julia should expose:

- `BoundaryCondition`
- `shift_operator`, `flip_operator`, `phase_rotation_operator`
- `shift_operator_multivar`, `flip_operator_multivar`, `phase_rotation_operator_multivar`
- `affine_operator`
- `binaryop_operator`
- `apply`

Design requirements:

- support mixed boundary conditions, e.g. frequency axes open and momentum axes periodic
- support dimension-preserving transforms such as `(nu, omega, k, q) -> (nu, nu+omega, k, k+q)`
- support dimension-changing embeddings such as `f(omega, q) -> g(nu, nup, k, kp) = f(nu-nup, k-kp)`
- preserve the current operator-application path onto `SimpleTensorTrain`

This surface is the one that directly enables `ReFrequenTT`-style kernels.

---

## E. Tensor Representation and Interoperability

### Canonical representation

Inside `Tensor4all.jl`, the canonical TT object remains `SimpleTT.SimpleTensorTrain`.

Reasons:

- it is already backed by Rust
- `QuanticsTCI` already converts into it
- arithmetic and partial reductions already exist on it
- it avoids making external packages part of the core abstraction boundary

### TensorCrossInterpolation interoperability

Keep the current weak-dep extension that converts between:

- `TensorCrossInterpolation.TensorTrain`
- `Tensor4all.SimpleTT.SimpleTensorTrain`

This is useful, but it should remain interoperability, not the primary API.

### ITensorLike interoperability

Do not design around `ITensorLike.TensorTrain` yet.

There is no current `ITensorLike` module in this repository, so adding it now would create a second unfinished abstraction boundary. If a concrete downstream need appears, add a weak-dependency extension later with explicit constructors, mirroring the `TensorCrossInterpolation` pattern.

---

## F. Migration Impact

### What becomes smoother

- `ReFrequenTT` can target `Tensor4all.jl` quantics primitives directly
- users can stay within one package family for grid, transform, and TT operations
- Julia wrappers stay stable as long as Rust/C API surfaces stay stable

### What does not become smoother

- code expecting the deleted Rust-backed `TensorCI2` API does not automatically recover
- a `TensorCrossInterpolation.jl` facade under the old `TensorCI` name is still likely to have behavioral mismatches

### Migration recommendation

For downstream ports, migrate toward:

1. grid construction via `QuanticsGrids`
2. interpolation/compression via `QuanticsTCI`
3. TT data via `SimpleTensorTrain`
4. operator construction via `QuanticsTransform`

Do not migrate toward a reintroduced `TensorCI` namespace unless a separate compatibility objective is explicitly accepted.

---

## Scope Exclusions

This design does not yet include:

- a `TTFunction`-style high-level wrapper bundling `(grid, tt, logical variables)`
- variable-aware contraction APIs analogous to `BubbleTeaCI.BasicContractOrder`
- a compatibility facade that recreates the removed Rust `TensorCI2` wrapper
- a new `ITensorLike` module

Those are valid follow-up layers, but they should be built on top of the Rust-parity quantics surface, not before it.
