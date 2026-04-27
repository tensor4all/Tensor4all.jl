# Index ID C API Roundtrip Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore explicit index identity roundtripping across the C API so Tensor4all.jl can use the latest local `tensor4all-rs` branch without an operation-local index pool.

**Architecture:** Keep `t4a_index_new` as the fresh-identity constructor and add language-binding-oriented `t4a_index_new_with_id` plus `t4a_index_id`. Tensor4all.jl continues to own Julia `Index.id` and passes that stable `UInt64` through the C API when constructing backend indices.

**Tech Stack:** Rust `tensor4all-capi`, Julia FFI wrappers, Julia `Test`, Rust `cargo test`.

---

### Task 1: Add Rust C API Tests For Explicit Index IDs

**Files:**
- Modify: `../tensor4all-rs/crates/tensor4all-capi/src/index/tests/mod.rs`

**Step 1: Write failing tests**

Add tests covering:
- `t4a_index_new_with_id` creates two separate handles that compare equal when `dim/id/tags/plev` match.
- Same `id` but different `tags` or `plev` does not compare equal.
- `t4a_index_id` returns the explicit `uint64_t` id.

**Step 2: Run tests to verify RED**

Run:

```bash
cargo test -p tensor4all-capi index --lib
```

Expected: compile failure because `t4a_index_new_with_id` and `t4a_index_id` do not exist.

### Task 2: Implement Rust C API Functions

**Files:**
- Modify: `../tensor4all-rs/crates/tensor4all-capi/src/index.rs`
- Modify: `../tensor4all-rs/crates/tensor4all-capi/include/tensor4all_capi.h`

**Step 1: Implement minimal Rust functions**

- Import `DynId`.
- Add helper to build an index from explicit `DynId(id)`, `dim`, tags, and `plev`.
- Export:
  - `t4a_index_new_with_id(size_t dim, uint64_t id, const char *tags_csv, int64_t plev, struct t4a_index **out)`
  - `t4a_index_id(const struct t4a_index *ptr, uint64_t *out_id)`

**Step 2: Run Rust tests**

Run:

```bash
cargo test -p tensor4all-capi index --lib
```

Expected: PASS.

### Task 3: Rebuild Local Rust Backend For Julia

**Files:**
- Generated: `deps/libtensor4all_capi.so`

**Step 1: Build from the current local Rust branch**

Run from `Tensor4all.jl`:

```bash
TENSOR4ALL_RS_PATH=../tensor4all-rs julia --startup-file=no --project=. deps/build.jl
```

Expected: release build succeeds and installs `deps/libtensor4all_capi.so`.

### Task 4: Verify Julia FFI Against Explicit Index IDs

**Files:**
- Existing: `src/TensorNetworks/backend/capi.jl`
- Existing tests: `test/core/tensor.jl`, `test/core/tensor_contract.jl`, `test/tensornetworks/*`

**Step 1: Run the failing Julia test first**

Run:

```bash
TENSOR4ALL_RS_PATH=../tensor4all-rs T4A_SKIP_HDF5_TESTS=1 julia --startup-file=no --project=. test/runtests.jl
```

Expected before Rust fix: missing `t4a_index_new_with_id`; expected after Rust fix: progress past that failure.

**Step 2: Make only necessary Julia changes**

If Rust API signatures match the existing Julia wrapper, no Julia source change is required. If the header-level C signature differs from the existing wrapper, update only `src/TensorNetworks/backend/capi.jl`.

**Step 3: Run focused Julia tests**

Run:

```bash
TENSOR4ALL_RS_PATH=../tensor4all-rs T4A_SKIP_HDF5_TESTS=1 julia --startup-file=no --project=. test/core/tensor.jl
TENSOR4ALL_RS_PATH=../tensor4all-rs T4A_SKIP_HDF5_TESTS=1 julia --startup-file=no --project=. test/core/tensor_contract.jl
```

Expected: PASS.

### Task 5: Full Verification

**Files:**
- No extra source changes unless failures identify a real regression.

**Step 1: Run Rust verification**

Run in `../tensor4all-rs`:

```bash
cargo test -p tensor4all-capi
```

Expected: PASS.

**Step 2: Run Julia verification**

Run in `Tensor4all.jl`:

```bash
TENSOR4ALL_RS_PATH=../tensor4all-rs T4A_SKIP_HDF5_TESTS=1 julia --startup-file=no --project=. test/runtests.jl
```

Expected: PASS, or report exact remaining failures.

