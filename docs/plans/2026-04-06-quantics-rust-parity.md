# Quantics Rust Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose the pure Rust quantics surface through `tensor4all-capi` and `Tensor4all.jl`, so downstream code can use grids and multivariable quantics transforms without reviving the removed `TensorCI` wrapper.

**Architecture:** Keep Rust as the source of truth for quantics functionality. Fill the missing FFI layer in `tensor4all-capi`, then add thin Julia bindings in `C_API.jl` and ergonomic wrappers in `QuanticsGrids.jl` and `QuanticsTransform.jl`. Use `SimpleTT.SimpleTensorTrain` as the canonical Julia TT object and leave external TT conversions in extension modules.

**Tech Stack:** Rust, `tensor4all-rs`, `tensor4all-capi`, Julia `ccall`, `Tensor4all.jl` (`C_API.jl`, `QuanticsGrids.jl`, `QuanticsTransform.jl`, `SimpleTT.jl`)

**Spec:** `docs/specs/2026-04-06-quantics-rust-parity-design.md`

---

### Task 1: Tighten the QuanticsGrids Julia surface

**Files:**
- Modify: `/home/shinaoka/tensor4all/Tensor4all.jl/src/QuanticsGrids.jl`
- Modify: `/home/shinaoka/tensor4all/Tensor4all.jl/src/Tensor4all.jl`
- Test: `/home/shinaoka/tensor4all/Tensor4all.jl/test/test_quanticsgrids.jl`

**Step 1: Write the failing test**

Add tests that require:

- `using Tensor4all` to access `DiscretizedGrid` and `InherentDiscreteGrid`
- `unfolding=:grouped` to be accepted when Rust/C API support it

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. -e 'using Pkg; Pkg.test(; test_args=[\"test_quanticsgrids\"])'`

Expected: failures showing missing top-level exports or unsupported `:grouped`.

**Step 3: Write minimal implementation**

- extend `_unfolding_to_cint` in `QuanticsGrids.jl` to include `:grouped`
- export selected grid types/functions from `Tensor4all.jl`
- keep the submodule intact for namespaced use

**Step 4: Run test to verify it passes**

Run: `julia --startup-file=no --project=. -e 'using Pkg; Pkg.test(; test_args=[\"test_quanticsgrids\"])'`

Expected: pass.

**Step 5: Commit**

```bash
git add src/QuanticsGrids.jl src/Tensor4all.jl test/test_quanticsgrids.jl
git commit -m "feat: polish QuanticsGrids Julia surface"
```

### Task 2: Expose multivariable quantics transform constructors in the C API

**Files:**
- Modify: `/home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-capi/src/quanticstransform.rs`
- Modify: `/home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-capi/src/types.rs`
- Test: `/home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-capi/src/quanticstransform/tests/mod.rs`

**Step 1: Write the failing Rust tests**

Add tests that attempt to construct:

- `shift_operator_multivar`
- `flip_operator_multivar`
- `phase_rotation_operator_multivar`
- `affine_operator`
- `binaryop_operator`

with mixed boundary conditions and asymmetric input/output dimensions where applicable.

**Step 2: Run tests to verify they fail**

Run: `cargo test -p tensor4all-capi quanticstransform -- --nocapture`

Expected: compile or symbol failures because the C API does not expose these constructors yet.

**Step 3: Write minimal implementation**

- add FFI-safe boundary-condition conversions in `types.rs` if needed
- expose the missing constructor entry points in `quanticstransform.rs`
- keep function signatures close to the Rust backend surface

**Step 4: Run tests to verify they pass**

Run: `cargo test -p tensor4all-capi quanticstransform -- --nocapture`

Expected: pass.

**Step 5: Commit**

```bash
git add crates/tensor4all-capi/src/types.rs crates/tensor4all-capi/src/quanticstransform.rs crates/tensor4all-capi/src/quanticstransform/tests/mod.rs
git commit -m "feat: expose multivariable quantics transforms in C API"
```

### Task 3: Add Julia C bindings for the new quantics transform constructors

**Files:**
- Modify: `/home/shinaoka/tensor4all/Tensor4all.jl/src/C_API.jl`
- Test: `/home/shinaoka/tensor4all/Tensor4all.jl/test/test_quanticstransform.jl`

**Step 1: Write the failing Julia test**

Add tests that call the new C-API wrappers from Julia and fail if any symbol or argument conversion is missing.

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. -e 'using Pkg; Pkg.test(; test_args=[\"test_quanticstransform\"])'`

Expected: `UndefVarError`, `MethodError`, or symbol lookup failures.

**Step 3: Write minimal implementation**

Add `ccall` wrappers in `C_API.jl` for:

- boundary-condition enum conversion
- multivariable shift/flip/phase constructors
- affine operator constructor
- binaryop operator constructor

**Step 4: Run test to verify it passes**

Run: `julia --startup-file=no --project=. -e 'using Pkg; Pkg.test(; test_args=[\"test_quanticstransform\"])'`

Expected: pass.

**Step 5: Commit**

```bash
git add src/C_API.jl test/test_quanticstransform.jl
git commit -m "feat: add Julia bindings for quantics transform C API"
```

### Task 4: Build the Julia QuanticsTransform wrapper surface

**Files:**
- Modify: `/home/shinaoka/tensor4all/Tensor4all.jl/src/QuanticsTransform.jl`
- Modify: `/home/shinaoka/tensor4all/Tensor4all.jl/src/Tensor4all.jl`
- Test: `/home/shinaoka/tensor4all/Tensor4all.jl/test/test_quanticstransform.jl`

**Step 1: Write the failing wrapper tests**

Cover:

- `BoundaryCondition` exposure
- multivariable operator constructors
- mixed boundary conditions
- a simple affine embedding case that changes logical variable count

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. -e 'using Pkg; Pkg.test(; test_args=[\"test_quanticstransform\"])'`

Expected: failures in wrapper construction or type dispatch.

**Step 3: Write minimal implementation**

- extend `QuanticsTransform.jl` with thin wrappers over the new `C_API.jl` bindings
- keep the operator object model consistent with the existing `apply` path
- export the new constructors without adding a fake `TensorCI` layer

**Step 4: Run test to verify it passes**

Run: `julia --startup-file=no --project=. -e 'using Pkg; Pkg.test(; test_args=[\"test_quanticstransform\"])'`

Expected: pass.

**Step 5: Commit**

```bash
git add src/QuanticsTransform.jl src/Tensor4all.jl test/test_quanticstransform.jl
git commit -m "feat: add Julia wrappers for multivariable quantics transforms"
```

### Task 5: Add a ReFrequenTT-shaped regression test

**Files:**
- Modify: `/home/shinaoka/tensor4all/Tensor4all.jl/test/test_quanticstransform.jl`
- Optionally modify: `/home/shinaoka/tensor4all/Tensor4all.jl/test/runtests.jl`

**Step 1: Write the failing regression test**

Add a small, local test that exercises the two patterns that matter most for `ReFrequenTT`:

- same-dimension affine remap with mixed BC
- dimension-embedding remap such as `(omega, q) -> (nu-nup, k-kp)`

The test only needs to verify operator construction and action on a small TT, not full physics correctness.

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. -e 'using Pkg; Pkg.test(; test_args=[\"test_quanticstransform\"])'`

Expected: failure until the wrapper surface matches the needed semantics.

**Step 3: Write minimal implementation**

- refine argument conventions
- fix BC ordering if needed
- add helper utilities only if tests show repetition

**Step 4: Run test to verify it passes**

Run: `julia --startup-file=no --project=. -e 'using Pkg; Pkg.test(; test_args=[\"test_quanticstransform\"])'`

Expected: pass.

**Step 5: Commit**

```bash
git add test/test_quanticstransform.jl test/runtests.jl
git commit -m "test: cover ReFrequenTT-style quantics remaps"
```

### Task 6: Re-evaluate whether a TensorCI shim is still needed

**Files:**
- Modify: `/home/shinaoka/tensor4all/Tensor4all.jl/docs/specs/2026-04-06-quantics-rust-parity-design.md`
- Modify: `/home/shinaoka/tensor4all/Tensor4all.jl/docs/plans/2026-04-06-quantics-rust-parity.md`

**Step 1: Review the resulting public surface**

Inspect whether the new grid/transform API already gives downstream users a direct migration path.

**Step 2: Decide based on evidence**

If the answer is yes, leave `TensorCI` removed.

If the answer is no, write down the smallest possible shim and the exact gap it closes.

**Step 3: Update docs**

Record the outcome in the design/plan docs instead of guessing up front.

**Step 4: Commit**

```bash
git add docs/specs/2026-04-06-quantics-rust-parity-design.md docs/plans/2026-04-06-quantics-rust-parity.md
git commit -m "docs: record TensorCI shim decision"
```
