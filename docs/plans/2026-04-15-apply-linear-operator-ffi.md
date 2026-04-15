# Apply Linear Operator FFI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `TensorNetworks.apply(::LinearOperator, ::TensorTrain)` through `tensor4all-rs` C FFI for chain `TensorTrain`s, with partial-operator support and small-size numerical tests.

**Architecture:** Keep Julia in charge of public semantics and validation. Use a one-shot C-API kernel that accepts an operator MPO, a state TT lowered as chain `TreeTN`, and explicit mapping arrays for the operator-covered sites. Leave partial identity extension and contraction strategy to Rust `apply_linear_operator`.

**Tech Stack:** Julia, Rust, `tensor4all-capi`, `ccall`, `TreeTN`, `LinearOperator`, Documenter-compatible docstrings.

---

### Task 1: Restore a loadable Julia module

**Files:**
- Modify: `src/TensorNetworks.jl`
- Test: `julia --startup-file=no --project=. -e 'using Tensor4all'`

**Step 1: Write the failing test**

Use the existing load failure caused by missing `backend/ffi.jl` and `backend/apply.jl`.

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. -e 'using Tensor4all'`
Expected: FAIL with missing `src/TensorNetworks/backend/ffi.jl`

**Step 3: Write minimal implementation**

Create the missing backend files or remove the invalid includes so the package loads again.

**Step 4: Run test to verify it passes**

Run: `julia --startup-file=no --project=. -e 'using Tensor4all'`
Expected: PASS

### Task 2: Add failing Julia apply tests

**Files:**
- Modify: `test/runtests.jl`
- Modify: `test/tensornetworks/apply.jl`

**Step 1: Write the failing test**

Cover:
- full-chain identity apply
- partial 1-site non-identity apply on a 2-site state
- dense numerical comparison on tiny systems

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. test/tensornetworks/apply.jl`
Expected: FAIL because `TensorNetworks.apply` still throws `SkeletonNotImplemented`

**Step 3: Write minimal implementation**

None yet. Keep the test red.

**Step 4: Run test to verify it still fails correctly**

Run: `julia --startup-file=no --project=. test/tensornetworks/apply.jl`
Expected: FAIL only because `apply` is unimplemented

### Task 3: Add the Rust C-API apply wrapper

**Files:**
- Modify: `../tensor4all-rs/crates/tensor4all-capi/src/treetn.rs`
- Modify: `../tensor4all-rs/crates/tensor4all-capi/src/treetn/tests/mod.rs`
- Modify: `../tensor4all-rs/crates/tensor4all-capi/include/tensor4all_capi.h`

**Step 1: Write the failing test**

Add Rust-side C-API tests for:
- 1-site identity operator apply
- partial 1-site non-identity apply on a 2-site chain state
- out-of-range mapped position error

**Step 2: Run test to verify it fails**

Run: `cargo test -p tensor4all-capi test_treetn_apply_operator_chain --lib`
Expected: FAIL because `t4a_treetn_apply_operator_chain` does not exist yet

**Step 3: Write minimal implementation**

Expose one-shot `t4a_treetn_apply_operator_chain(...)` that:
- accepts operator/state `t4a_treetn`
- accepts `mapped_positions`
- accepts internal input/output indices and desired true output indices
- builds Rust `LinearOperator`
- calls `apply_linear_operator`

**Step 4: Run test to verify it passes**

Run:
- `cargo test -p tensor4all-capi test_treetn_apply_operator_chain --lib`
- `cargo test -p tensor4all-capi --lib`

Expected: PASS

### Task 4: Build Julia FFI glue

**Files:**
- Create: `src/TensorNetworks/backend/ffi.jl`
- Create: `src/TensorNetworks/backend/apply.jl`
- Modify: `src/TensorNetworks.jl`
- Modify: `src/TensorNetworks/deferred.jl`

**Step 1: Write the failing test**

Reuse the red Julia apply tests from Task 2.

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. test/tensornetworks/apply.jl`
Expected: FAIL because the Julia FFI bridge does not exist yet

**Step 3: Write minimal implementation**

Implement:
- low-level `ccall` wrappers and error propagation
- lowering `Index`/`Tensor`/`TensorTrain` to C handles
- chain-only validation
- partial mapping extraction from `LinearOperator`
- result reconstruction back into Julia `TensorTrain`
- `apply(::LinearOperator, ::TensorTrain; kwargs...)`

**Step 4: Run test to verify it passes**

Run: `julia --startup-file=no --project=. test/tensornetworks/apply.jl`
Expected: PASS

### Task 5: Reconcile deferred helpers and current branch changes

**Files:**
- Modify: `src/TensorNetworks/transforms.jl`
- Modify: `test/tensornetworks/transform_helpers.jl`

**Step 1: Write the failing test**

Use the new expectation that `rearrange_siteinds` throws `SkeletonNotImplemented`.

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. test/tensornetworks/transform_helpers.jl`
Expected: FAIL if dense fallback is still present

**Step 3: Write minimal implementation**

Keep `rearrange_siteinds` deferred with an internal TODO tied to upstream `tensor4all-rs` work.

**Step 4: Run test to verify it passes**

Run: `julia --startup-file=no --project=. test/tensornetworks/transform_helpers.jl`
Expected: PASS

### Task 6: Full verification

**Files:**
- Modify if needed after failures: `src/...`, `test/...`, `docs/...`

**Step 1: Run targeted verification**

Run:
- `julia --startup-file=no --project=. -e 'using Tensor4all'`
- `julia --startup-file=no --project=. test/tensornetworks/apply.jl`
- `julia --startup-file=no --project=. test/tensornetworks/transform_helpers.jl`

Expected: PASS

**Step 2: Run full verification**

Run:
- `julia --startup-file=no --project=. deps/build.jl`
- `julia --startup-file=no --project=. test/runtests.jl`
- `julia --startup-file=no --project=docs docs/make.jl`
- `git diff --check`

Expected: PASS

### Task 7: Publish Rust and Julia branches

**Files:**
- Commit only intended files in each repo

**Step 1: Commit Rust changes**

Run: commit the `tensor4all-rs` C-API patch with a focused message.

**Step 2: Push Rust branch and open PR**

Run: push the branch and open a draft PR for `tensor4all-rs`.

**Step 3: Commit Julia changes**

Run: commit the `Tensor4all.jl` apply + deferred-helper updates with a focused message.

**Step 4: Push Julia branch and open PR**

Run: push the branch and open a draft PR for `Tensor4all.jl`.
