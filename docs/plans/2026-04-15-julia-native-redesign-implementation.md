# Julia-Native Execution Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** replace the remaining POC/skeleton framing with an implementation-phase Julia frontend, make `QuanticsTransform` executable with Julia-native inputs, and translate the relevant `tensor4all-rs` tests into `TensorNetworks` and `QuanticsTransform` Julia coverage.

**Architecture:** keep the existing public module split, but move `TensorNetworks` and `QuanticsTransform` from placeholder-oriented exports to implemented APIs. Julia owns the public argument normalization and high-level error messages; `tensor4all-rs` remains the semantic reference for operator behavior and the source for translated tests.

**Tech Stack:** Julia 1.11, `tensor4all-capi` from `../tensor4all-rs`, `LinearAlgebra`, `Test`, `HDF5`, `Documenter.jl`.

---

### Task 1: Rewrite the public phase narrative

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/src/index.md`
- Modify: `docs/src/api.md`
- Modify: `docs/src/modules.md`
- Modify: `docs/src/deferred_rework_plan.md`

**Step 1: Write the failing documentation assertions**

Document the exact phrases to remove:

- `POC`
- `skeleton`
- `metadata-only placeholder`
- any text that presents `TensorNetworks` or `QuanticsTransform` as primarily name-only surfaces

**Step 2: Verify the current wording is still present**

Run:

```bash
rg -n "POC|poc|skeleton|metadata-only|placeholder" AGENTS.md README.md docs/src
```

Expected: matches are found in the current docs.

**Step 3: Update the wording**

Rewrite those files so they describe:

- an implementation-phase repository
- implemented versus still-missing behavior explicitly
- Rust-backed semantics with Julia-native APIs

**Step 4: Re-run the wording check**

Run:

```bash
rg -n "POC|poc|metadata-only placeholder" AGENTS.md README.md docs/src
```

Expected: no matches for the removed narrative in the active public docs.

**Step 5: Commit**

```bash
git add AGENTS.md README.md docs/src/index.md docs/src/api.md docs/src/modules.md docs/src/deferred_rework_plan.md
git commit -m "docs: move frontend narrative to implementation phase"
```

### Task 2: Restructure `QuanticsTransform` for real implementation

**Files:**
- Modify: `src/QuanticsTransform.jl`
- Create: `src/QuanticsTransform/Normalization.jl`
- Create: `src/QuanticsTransform/Materialize.jl`
- Create: `src/QuanticsTransform/Types.jl` only if an internal helper type is truly needed

**Step 1: Write failing constructor tests for Julia-native arguments**

Create or expand:

- `test/quanticstransform/constructors.jl`

Cover at least:

- `affine_operator(r, A, b; bc=...)` accepts `AbstractMatrix` / `AbstractVector`
- invalid shapes throw `ArgumentError`
- invalid `r`, `theta`, or boundary conditions throw actionable errors

**Step 2: Run the focused constructor tests**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: constructor-focused tests fail because the current API still returns placeholders or lacks validation.

**Step 3: Implement normalization**

Add a small internal layer that:

- normalizes `bc` keywords into backend-ready arrays
- checks matrix/vector dimensionality
- converts `AbstractMatrix` into column-major flattened buffers only internally
- validates scalar finiteness where Rust expects it

**Step 4: Keep exported docstrings concise**

Ensure every exported constructor has a concise docstring describing:

- Julia-native inputs
- key keyword arguments
- what the returned operator represents

**Step 5: Run the focused constructor tests again**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: constructor tests pass.

**Step 6: Commit**

```bash
git add src/QuanticsTransform.jl src/QuanticsTransform test/quanticstransform
git commit -m "feat: normalize Julia-native quantics transform inputs"
```

### Task 3: Materialize quantics operators instead of returning placeholders

**Files:**
- Modify: `src/QuanticsTransform.jl`
- Modify: `src/Core/CAPI.jl`
- Modify: `src/TensorNetworks.jl`
- Create: `src/TensorNetworks/Apply.jl` if `TensorNetworks.jl` grows too large

**Step 1: Write failing execution tests**

Create or expand:

- `test/quanticstransform/numeric_equivalence.jl`
- `test/tensornetworks/apply.jl`

Start with small cases:

- shift on `r=2`
- flip on `r=2`
- phase rotation on `r=2`
- cumsum on `r=2`

The tests should compare either:

- dense matrices from the Julia operator against dense reference matrices, or
- `apply` against basis-vector expectations

**Step 2: Run the new tests to confirm failure**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: failures due to metadata-only operators and deferred `apply`.

**Step 3: Implement operator materialization**

Use existing C-API entry points from `tensor4all-rs` where available. Keep the
Julia public API unchanged while populating executable `LinearOperator`
instances.

**Step 4: Implement `TensorNetworks.apply`**

Support the operator application path needed by the translated quantics tests.
Validate input/output index compatibility in Julia before calling the backend.

**Step 5: Re-run the focused tests**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: the new shift/flip/phase/cumsum execution tests pass.

**Step 6: Commit**

```bash
git add src/Core/CAPI.jl src/QuanticsTransform.jl src/QuanticsTransform src/TensorNetworks.jl src/TensorNetworks test/quanticstransform test/tensornetworks
git commit -m "feat: materialize quantics operators and apply them"
```

### Task 4: Translate the Rust `QuanticsTransform` validation suite

**Files:**
- Modify: `test/quanticstransform/constructors.jl`
- Use as references:
  - `../tensor4all-rs/crates/tensor4all-quanticstransform/src/shift/tests/mod.rs`
  - `../tensor4all-rs/crates/tensor4all-quanticstransform/src/flip/tests/mod.rs`
  - `../tensor4all-rs/crates/tensor4all-quanticstransform/src/phase_rotation/tests/mod.rs`
  - `../tensor4all-rs/crates/tensor4all-quanticstransform/src/cumsum/tests/mod.rs`
  - `../tensor4all-rs/crates/tensor4all-quanticstransform/src/fourier/tests/mod.rs`
  - `../tensor4all-rs/crates/tensor4all-quanticstransform/src/affine/tests/mod.rs`

**Step 1: Port the simplest constructor-validation cases**

Translate at least:

- zero-site errors
- maximum `r` boundary checks where applicable
- `theta` NaN / infinity checks
- affine boundary-condition length mismatch checks
- affine matrix/vector size mismatch checks

**Step 2: Run the focused tests**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: failures only where Julia validation is still incomplete.

**Step 3: Fill validation gaps**

Adjust normalization and error paths until the translated cases pass.

**Step 4: Commit**

```bash
git add src/QuanticsTransform.jl src/QuanticsTransform test/quanticstransform/constructors.jl
git commit -m "test: translate quantics constructor validation from rust"
```

### Task 5: Translate Rust numeric equivalence tests for quantics operators

**Files:**
- Modify: `test/quanticstransform/numeric_equivalence.jl`
- Use as references:
  - `../tensor4all-rs/crates/tensor4all-capi/src/quanticstransform/tests/mod.rs`
  - `../tensor4all-rs/crates/tensor4all-quanticstransform/tests/integration_test.rs`

**Step 1: Add dense-reference helpers**

In the Julia tests, add helpers that:

- enumerate basis states for small `r`
- build dense reference matrices for shift, flip, phase rotation, cumsum, and affine transforms
- compare complex results with tolerances

**Step 2: Translate the first wave**

Port numerical checks for:

- shift
- flip
- phase rotation
- cumsum

**Step 3: Translate the second wave**

Port numerical checks for:

- Fourier forward / inverse
- affine operator
- affine pullback operator
- binary operators where the existing backend path is already exposed

**Step 4: Re-run tests**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: all translated quantics numeric tests pass.

**Step 5: Commit**

```bash
git add test/quanticstransform/numeric_equivalence.jl
git commit -m "test: translate quantics numeric equivalence from rust"
```

### Task 6: Replace `TensorNetworks` skeleton tests with semantic tests

**Files:**
- Modify: `test/tensornetworks/skeleton_surface.jl`
- Create: `test/tensornetworks/helpers.jl`
- Create: `test/tensornetworks/apply.jl` if not already created
- Modify: `test/runtests.jl`

**Step 1: Write failing helper tests**

Cover:

- `findsite` and `findsites` by explicit site indices and tags
- `findallsiteinds_by_tag` and `findallsites_by_tag`
- `replace_siteinds!`, `replace_siteinds`, and `replace_siteinds_part!`
- `rearrange_siteinds`
- `makesitediagonal` and `extractdiagonal`
- `matchsiteinds`

Use tiny deterministic tensor trains with explicit site and link indices.

**Step 2: Run the helper tests**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: failures because current tests and code still assume stubs.

**Step 3: Implement the helper behavior**

Implement the helper functions in `TensorNetworks` using Julia-owned chain logic.
Do not widen the C API for behavior that can be expressed by index manipulation
over `Tensor` objects.

**Step 4: Remove the throw-centric test expectations**

Delete or rewrite checks that only assert `SkeletonNotImplemented`.

**Step 5: Re-run tests**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: `TensorNetworks` helper tests pass with real behavior.

**Step 6: Commit**

```bash
git add src/TensorNetworks.jl src/TensorNetworks test/tensornetworks test/runtests.jl
git commit -m "feat: implement tensornetwork helper semantics"
```

### Task 7: Keep HDF5 and interop coverage aligned with the new execution story

**Files:**
- Modify: `test/extensions/hdf5_roundtrip.jl`
- Modify: `test/extensions/hdf5_itensors_interop.jl`
- Modify: `docs/src/api.md` if HDF5 behavior descriptions changed during implementation

**Step 1: Review current HDF5 coverage against the Rust suite**

Use:

- `../tensor4all-rs/crates/tensor4all-hdf5/tests/test_hdf5.rs`

Map only the Julia-visible contract:

- roundtrip data
- index metadata
- column-major storage expectations
- MPS `llim` / `rlim` preservation

**Step 2: Add any missing Julia-visible checks**

Do not duplicate backend-internal schema unit tests unless they affect Julia
behavior directly.

**Step 3: Run the HDF5-focused suite**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: HDF5 roundtrip and ITensorMPS interop tests pass.

**Step 4: Commit**

```bash
git add test/extensions/hdf5_roundtrip.jl test/extensions/hdf5_itensors_interop.jl docs/src/api.md
git commit -m "test: align julia hdf5 coverage with rust-visible behavior"
```

### Task 8: Final docs and verification sweep

**Files:**
- Modify as needed: `README.md`, `AGENTS.md`, `docs/src/*.md`, `docs/design/*.md`

**Step 1: Reconcile docs with the implemented behavior**

Ensure docs no longer claim:

- placeholder-only `QuanticsTransform`
- throw-only `TensorNetworks` helper surface
- POC phase language for the active branch

**Step 2: Run full verification**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
julia --startup-file=no --project=docs docs/make.jl
```

Expected: both commands pass.

**Step 3: Inspect the final diff**

Run:

```bash
git status --short
git diff --stat
```

Expected: only intended implementation, test, and docs changes remain.

**Step 4: Commit**

```bash
git add AGENTS.md README.md docs src test
git commit -m "feat: execute julia-native quantics and tensornetwork redesign"
```
