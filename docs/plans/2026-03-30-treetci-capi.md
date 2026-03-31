# TreeTCI C API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add C API bindings for tensor4all-treetci to the tensor4all-capi crate, enabling Julia/Python to call TreeTCI via FFI.

**Architecture:** Stateful API exposing `SimpleTreeTci` as an opaque handle with graph construction, pivot management, sweep execution, state inspection, materialization, and a high-level convenience function. Single batch callback unifies point and batch evaluation.

**Tech Stack:** Rust, C FFI, tensor4all-treetci, tensor4all-capi patterns (opaque handles, catch_unwind, status codes)

**Reference:** Design spec at `Tensor4all.jl/docs/specs/2026-03-30-treetci-capi-and-julia-wrapper-design.md`

**Working directory:** `/home/shinaoka/tensor4all/tensor4all-rs`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `crates/tensor4all-capi/Cargo.toml` | Add `tensor4all-treetci` dependency |
| Modify | `crates/tensor4all-capi/src/lib.rs` | Add `mod treetci; pub use treetci::*;` |
| Modify | `crates/tensor4all-capi/src/types.rs` | Add `t4a_treetci_graph`, `t4a_treetci_f64`, `t4a_treetci_proposer_kind` |
| Create | `crates/tensor4all-capi/src/treetci.rs` | All TreeTCI C API functions |

---

## Task 1: Scaffold — Cargo.toml, types, lib.rs, empty module

**Files:**
- Modify: `crates/tensor4all-capi/Cargo.toml`
- Modify: `crates/tensor4all-capi/src/types.rs`
- Modify: `crates/tensor4all-capi/src/lib.rs`
- Create: `crates/tensor4all-capi/src/treetci.rs`

- [ ] **Step 1: Add tensor4all-treetci dependency to Cargo.toml**

In `crates/tensor4all-capi/Cargo.toml`, add to `[dependencies]`:

```toml
tensor4all-treetci = { path = "../tensor4all-treetci" }
```

- [ ] **Step 2: Add opaque types and enum to types.rs**

At the end of `crates/tensor4all-capi/src/types.rs`, add:

```rust
// ============================================================================
// TreeTCI types
// ============================================================================

use tensor4all_treetci::{TreeTciGraph, SimpleTreeTci};
use std::ffi::c_void;

/// Opaque tree graph type for TreeTCI
#[repr(C)]
pub struct t4a_treetci_graph {
    pub(crate) _private: *const c_void,
}

impl t4a_treetci_graph {
    pub(crate) fn new(inner: TreeTciGraph) -> Self {
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const c_void,
        }
    }

    pub(crate) fn inner(&self) -> &TreeTciGraph {
        unsafe { &*(self._private as *const TreeTciGraph) }
    }
}

impl Clone for t4a_treetci_graph {
    fn clone(&self) -> Self {
        Self::new(self.inner().clone())
    }
}

impl Drop for t4a_treetci_graph {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                let _ = Box::from_raw(self._private as *mut TreeTciGraph);
            }
        }
    }
}

unsafe impl Send for t4a_treetci_graph {}
unsafe impl Sync for t4a_treetci_graph {}

/// Opaque TreeTCI state (f64)
#[repr(C)]
pub struct t4a_treetci_f64 {
    pub(crate) _private: *const c_void,
}

impl t4a_treetci_f64 {
    pub(crate) fn new(inner: SimpleTreeTci<f64>) -> Self {
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const c_void,
        }
    }

    pub(crate) fn inner(&self) -> &SimpleTreeTci<f64> {
        unsafe { &*(self._private as *const SimpleTreeTci<f64>) }
    }

    pub(crate) fn inner_mut(&mut self) -> &mut SimpleTreeTci<f64> {
        unsafe { &mut *(self._private as *mut SimpleTreeTci<f64>) }
    }
}

impl Drop for t4a_treetci_f64 {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                let _ = Box::from_raw(self._private as *mut SimpleTreeTci<f64>);
            }
        }
    }
}

// No Clone — same as t4a_tci2_f64
unsafe impl Send for t4a_treetci_f64 {}
unsafe impl Sync for t4a_treetci_f64 {}

/// Proposer kind selection for TreeTCI
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_treetci_proposer_kind {
    /// DefaultProposer: neighbor-product (matches TreeTCI.jl)
    Default = 0,
    /// SimpleProposer: random with seed
    Simple = 1,
    /// TruncatedDefaultProposer: truncated random subset of default candidates
    TruncatedDefault = 2,
}
```

Note: `c_void` is likely already imported at the top of types.rs. If not, add `use std::ffi::c_void;`. Similarly, check if `TreeTciGraph` and `SimpleTreeTci` imports conflict with existing imports — they should not since they're from a new crate.

- [ ] **Step 3: Add mod and re-export to lib.rs**

In `crates/tensor4all-capi/src/lib.rs`, add alongside the existing module declarations:

```rust
mod treetci;
```

And alongside the existing `pub use` statements:

```rust
pub use treetci::*;
```

- [ ] **Step 4: Create empty treetci.rs**

Create `crates/tensor4all-capi/src/treetci.rs`:

```rust
//! C API for TreeTCI (tree-structured tensor cross interpolation)

use crate::types::{t4a_treetci_f64, t4a_treetci_graph, t4a_treetci_proposer_kind};
use crate::{err_status, set_last_error, StatusCode, T4A_INTERNAL_ERROR, T4A_NULL_POINTER, T4A_SUCCESS};
use crate::t4a_treetn;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tensor4all_treetci::{
    DefaultProposer, GlobalIndexBatch, SimpleProposer, SimpleTreeTci, TreeTciEdge,
    TreeTciGraph, TreeTciOptions, TruncatedDefaultProposer,
};
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo build -p tensor4all-capi --release 2>&1 | head -20`

Expected: Successful compilation (warnings OK, no errors).

- [ ] **Step 6: Commit**

```bash
git add crates/tensor4all-capi/Cargo.toml crates/tensor4all-capi/src/types.rs \
       crates/tensor4all-capi/src/lib.rs crates/tensor4all-capi/src/treetci.rs
git commit -m "feat(capi): scaffold TreeTCI C API module with opaque types"
```

---

## Task 2: Graph construction functions

**Files:**
- Modify: `crates/tensor4all-capi/src/treetci.rs`

- [ ] **Step 1: Write the failing test**

Append to `crates/tensor4all-capi/src/treetci.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: 7-site branching tree
    ///     0
    ///     |
    ///     1---2
    ///     |
    ///     3
    ///     |
    ///     4
    ///    / \
    ///   5   6
    fn sample_edges() -> Vec<libc::size_t> {
        // flat: [u0,v0, u1,v1, ...]
        vec![0, 1, 1, 2, 1, 3, 3, 4, 4, 5, 4, 6]
    }

    #[test]
    fn test_graph_new_and_query() {
        let edges = sample_edges();
        let graph = t4a_treetci_graph_new(7, edges.as_ptr(), 6);
        assert!(!graph.is_null());

        let graph_ref = unsafe { &*graph };

        let mut n_sites: libc::size_t = 0;
        let status = t4a_treetci_graph_n_sites(graph, &mut n_sites);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(n_sites, 7);

        let mut n_edges: libc::size_t = 0;
        let status = t4a_treetci_graph_n_edges(graph, &mut n_edges);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(n_edges, 6);

        t4a_treetci_graph_release(graph);
    }

    #[test]
    fn test_graph_invalid_disconnected() {
        // 4 sites but only 2 edges, disconnected
        let edges: Vec<libc::size_t> = vec![0, 1, 2, 3];
        let graph = t4a_treetci_graph_new(4, edges.as_ptr(), 2);
        assert!(graph.is_null()); // should fail validation
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tensor4all-capi test_graph_new_and_query 2>&1 | tail -10`

Expected: FAIL — functions `t4a_treetci_graph_new`, etc. not defined.

- [ ] **Step 3: Implement graph functions**

Add to `crates/tensor4all-capi/src/treetci.rs` (before the `#[cfg(test)]` block):

```rust
// ============================================================================
// Graph lifecycle
// ============================================================================

impl_opaque_type_common!(treetci_graph);

/// Create a new tree graph.
///
/// # Arguments
/// - `n_sites`: Number of sites (>= 1)
/// - `edges_flat`: Edge pairs [u0, v0, u1, v1, ...] (length = n_edges * 2)
/// - `n_edges`: Number of edges (must equal n_sites - 1 for a tree)
///
/// # Returns
/// New graph handle, or NULL on error (invalid tree structure).
///
/// # Safety
/// `edges_flat` must point to a valid buffer of `n_edges * 2` elements.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_graph_new(
    n_sites: libc::size_t,
    edges_flat: *const libc::size_t,
    n_edges: libc::size_t,
) -> *mut t4a_treetci_graph {
    if edges_flat.is_null() && n_edges > 0 {
        set_last_error("edges_flat is null");
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let edges: Vec<TreeTciEdge> = (0..n_edges)
            .map(|i| {
                let u = unsafe { *edges_flat.add(2 * i) };
                let v = unsafe { *edges_flat.add(2 * i + 1) };
                TreeTciEdge::new(u, v)
            })
            .collect();

        match TreeTciGraph::new(n_sites, &edges) {
            Ok(graph) => Box::into_raw(Box::new(t4a_treetci_graph::new(graph))),
            Err(e) => {
                set_last_error(&e.to_string());
                std::ptr::null_mut()
            }
        }
    }));

    crate::unwrap_catch_ptr(result)
}

/// Get the number of sites in the graph.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_graph_n_sites(
    graph: *const t4a_treetci_graph,
    out: *mut libc::size_t,
) -> StatusCode {
    if graph.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let g = unsafe { &*graph };
        unsafe { *out = g.inner().n_sites() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the number of edges in the graph.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_graph_n_edges(
    graph: *const t4a_treetci_graph,
    out: *mut libc::size_t,
) -> StatusCode {
    if graph.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let g = unsafe { &*graph };
        unsafe { *out = g.inner().edges().len() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo nextest run --release -p tensor4all-capi test_graph 2>&1 | tail -10`

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-capi/src/treetci.rs
git commit -m "feat(capi): add TreeTCI graph construction functions"
```

---

## Task 3: Callback type and closure helpers

**Files:**
- Modify: `crates/tensor4all-capi/src/treetci.rs`

- [ ] **Step 1: Add callback type and closure helpers**

Add to `crates/tensor4all-capi/src/treetci.rs` (after the imports, before graph functions):

```rust
// ============================================================================
// Callback type
// ============================================================================

/// Batch evaluation callback for TreeTCI.
///
/// Evaluates the target function at multiple points simultaneously.
/// When `n_points == 1`, this acts as a single-point evaluation.
///
/// # Arguments
/// * `batch_data` - Column-major (n_sites, n_points) index array.
///   Element at (site, point) is at `batch_data[site + n_sites * point]`.
/// * `n_sites` - Number of sites
/// * `n_points` - Number of evaluation points
/// * `results` - Output buffer for `n_points` f64 values
/// * `user_data` - User data pointer passed through from the calling function
///
/// # Returns
/// 0 on success, non-zero on error
pub type TreeTciBatchEvalCallback = extern "C" fn(
    batch_data: *const libc::size_t,
    n_sites: libc::size_t,
    n_points: libc::size_t,
    results: *mut libc::c_double,
    user_data: *mut c_void,
) -> i32;

// ============================================================================
// Internal helpers
// ============================================================================

/// Create a batch eval closure from the C callback.
///
/// Returns a closure compatible with `Fn(GlobalIndexBatch<'_>) -> Result<Vec<f64>>`.
fn make_batch_eval_closure(
    eval_fn: TreeTciBatchEvalCallback,
    user_data: *mut c_void,
) -> impl Fn(GlobalIndexBatch<'_>) -> anyhow::Result<Vec<f64>> {
    move |batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<f64>> {
        let mut results = vec![0.0f64; batch.n_points()];
        let status = eval_fn(
            batch.data().as_ptr(),
            batch.n_sites(),
            batch.n_points(),
            results.as_mut_ptr(),
            user_data,
        );
        if status != 0 {
            anyhow::bail!("TreeTCI batch eval callback returned error status {}", status);
        }
        Ok(results)
    }
}

/// Create a point eval closure from the C batch callback (n_points=1).
fn make_point_eval_closure(
    eval_fn: TreeTciBatchEvalCallback,
    user_data: *mut c_void,
) -> impl Fn(&[usize]) -> f64 {
    move |indices: &[usize]| -> f64 {
        let mut result: f64 = 0.0;
        let status = eval_fn(
            indices.as_ptr(),
            indices.len(),
            1,
            &mut result,
            user_data,
        );
        if status != 0 {
            f64::NAN
        } else {
            result
        }
    }
}

/// Convert proposer kind enum to a boxed proposer trait object is not needed;
/// instead we dispatch at call sites. This helper creates TreeTciOptions from
/// C API parameters.
fn make_options(tolerance: f64, max_bond_dim: libc::size_t, max_iter: libc::size_t, normalize_error: bool) -> TreeTciOptions {
    TreeTciOptions {
        tolerance,
        max_bond_dim: if max_bond_dim == 0 { usize::MAX } else { max_bond_dim },
        max_iter,
        normalize_error,
    }
}
```

Also add `anyhow` import at the top if not already present. Check existing imports — if `anyhow` is not a dependency of tensor4all-capi, add it to Cargo.toml:

```toml
anyhow.workspace = true
```

(Check if `anyhow` is in the workspace `[workspace.dependencies]` in the root `Cargo.toml` first. If not, use `anyhow = "1"` directly.)

- [ ] **Step 2: Verify it compiles**

Run: `cargo build -p tensor4all-capi --release 2>&1 | head -20`

Expected: Successful compilation.

- [ ] **Step 3: Commit**

```bash
git add crates/tensor4all-capi/src/treetci.rs crates/tensor4all-capi/Cargo.toml
git commit -m "feat(capi): add TreeTCI callback type and closure helpers"
```

---

## Task 4: State lifecycle and pivot management

**Files:**
- Modify: `crates/tensor4all-capi/src/treetci.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `treetci.rs`:

```rust
    #[test]
    fn test_state_new_and_add_pivots() {
        let edges = sample_edges();
        let graph = t4a_treetci_graph_new(7, edges.as_ptr(), 6);
        assert!(!graph.is_null());

        let local_dims: Vec<libc::size_t> = vec![2; 7];
        let state = t4a_treetci_f64_new(local_dims.as_ptr(), 7, graph);
        assert!(!state.is_null());

        // Add one pivot: all zeros (column-major, n_sites=7, n_pivots=1)
        let pivot: Vec<libc::size_t> = vec![0; 7];
        let status = t4a_treetci_f64_add_global_pivots(state, pivot.as_ptr(), 7, 1);
        assert_eq!(status, T4A_SUCCESS);

        // Add two pivots at once (column-major)
        // pivot0 = [0,0,0,0,0,0,0], pivot1 = [1,0,1,0,1,0,1]
        let pivots: Vec<libc::size_t> = vec![
            0, 0, 0, 0, 0, 0, 0, // column 0 (sites 0-6 for point 0)
            1, 0, 1, 0, 1, 0, 1, // column 1 (sites 0-6 for point 1)
        ];
        let status = t4a_treetci_f64_add_global_pivots(state, pivots.as_ptr(), 7, 2);
        assert_eq!(status, T4A_SUCCESS);

        t4a_treetci_f64_release(state);
        t4a_treetci_graph_release(graph);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tensor4all-capi test_state_new_and_add_pivots 2>&1 | tail -10`

Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement state lifecycle and pivot functions**

Add to `treetci.rs` (after graph functions, before `#[cfg(test)]`):

```rust
// ============================================================================
// State lifecycle
// ============================================================================

/// Create a new TreeTCI state.
///
/// # Arguments
/// - `local_dims`: Local dimension at each site (length = n_sites)
/// - `n_sites`: Number of sites (must match graph)
/// - `graph`: Tree graph handle (not consumed; cloned internally)
///
/// # Returns
/// New state handle, or NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_new(
    local_dims: *const libc::size_t,
    n_sites: libc::size_t,
    graph: *const t4a_treetci_graph,
) -> *mut t4a_treetci_f64 {
    if local_dims.is_null() || graph.is_null() {
        set_last_error("local_dims or graph is null");
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let dims: Vec<usize> = (0..n_sites)
            .map(|i| unsafe { *local_dims.add(i) })
            .collect();
        let g = unsafe { &*graph };
        let graph_clone = g.inner().clone();

        match SimpleTreeTci::new(dims, graph_clone) {
            Ok(state) => Box::into_raw(Box::new(t4a_treetci_f64::new(state))),
            Err(e) => {
                set_last_error(&e.to_string());
                std::ptr::null_mut()
            }
        }
    }));

    crate::unwrap_catch_ptr(result)
}

/// Release a TreeTCI state.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_release(ptr: *mut t4a_treetci_f64) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

// ============================================================================
// Pivot management
// ============================================================================

/// Add global pivots to the TreeTCI state.
///
/// Each pivot is a multi-index over all sites. The pivots are projected
/// to per-edge pivot sets internally.
///
/// # Arguments
/// - `ptr`: State handle
/// - `pivots_flat`: Column-major (n_sites, n_pivots) index array
/// - `n_sites`: Number of sites (must match state)
/// - `n_pivots`: Number of pivots
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_add_global_pivots(
    ptr: *mut t4a_treetci_f64,
    pivots_flat: *const libc::size_t,
    n_sites: libc::size_t,
    n_pivots: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }
    if pivots_flat.is_null() && n_pivots > 0 {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = unsafe { &mut *ptr };
        let state_inner = state.inner_mut();

        // Unpack column-major (n_sites, n_pivots) to Vec<Vec<usize>>
        let pivots: Vec<Vec<usize>> = (0..n_pivots)
            .map(|p| {
                (0..n_sites)
                    .map(|s| unsafe { *pivots_flat.add(s + n_sites * p) })
                    .collect()
            })
            .collect();

        match state_inner.add_global_pivots(&pivots) {
            Ok(()) => T4A_SUCCESS,
            Err(e) => err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo nextest run --release -p tensor4all-capi test_state_new_and_add_pivots 2>&1 | tail -10`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-capi/src/treetci.rs
git commit -m "feat(capi): add TreeTCI state lifecycle and pivot management"
```

---

## Task 5: Sweep function

**Files:**
- Modify: `crates/tensor4all-capi/src/treetci.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
    /// Product function: f(idx) = prod(idx[s] + 1.0)
    /// This has an exact TT representation with bond dim 1.
    extern "C" fn product_batch_eval(
        batch_data: *const libc::size_t,
        n_sites: libc::size_t,
        n_points: libc::size_t,
        results: *mut libc::c_double,
        _user_data: *mut c_void,
    ) -> i32 {
        for p in 0..n_points {
            let mut val = 1.0f64;
            for s in 0..n_sites {
                let idx = unsafe { *batch_data.add(s + n_sites * p) };
                val *= (idx as f64) + 1.0;
            }
            unsafe { *results.add(p) = val };
        }
        0
    }

    #[test]
    fn test_sweep() {
        let edges = sample_edges();
        let graph = t4a_treetci_graph_new(7, edges.as_ptr(), 6);
        let local_dims: Vec<libc::size_t> = vec![2; 7];
        let state = t4a_treetci_f64_new(local_dims.as_ptr(), 7, graph);

        // Add initial pivot
        let pivot: Vec<libc::size_t> = vec![0; 7];
        t4a_treetci_f64_add_global_pivots(state, pivot.as_ptr(), 7, 1);

        // Run one sweep
        let status = t4a_treetci_f64_sweep(
            state,
            product_batch_eval,
            std::ptr::null_mut(), // no user_data needed
            t4a_treetci_proposer_kind::Default,
            1e-12,
            0, // unlimited bond dim
        );
        assert_eq!(status, T4A_SUCCESS);

        t4a_treetci_f64_release(state);
        t4a_treetci_graph_release(graph);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tensor4all-capi test_sweep 2>&1 | tail -10`

Expected: FAIL — `t4a_treetci_f64_sweep` not defined.

- [ ] **Step 3: Implement sweep function**

Add to `treetci.rs` (after pivot management):

```rust
// ============================================================================
// Sweep execution
// ============================================================================

/// Run one optimization iteration (visit all edges once).
///
/// Internally calls `optimize_with_proposer` with `max_iter=1`.
///
/// # Arguments
/// - `ptr`: State handle (mutable)
/// - `eval_cb`: Batch evaluation callback
/// - `user_data`: User data passed to callback
/// - `proposer_kind`: Proposer selection
/// - `tolerance`: Relative tolerance for this iteration
/// - `max_bond_dim`: Maximum bond dimension (0 = unlimited)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_sweep(
    ptr: *mut t4a_treetci_f64,
    eval_cb: TreeTciBatchEvalCallback,
    user_data: *mut c_void,
    proposer_kind: t4a_treetci_proposer_kind,
    tolerance: libc::c_double,
    max_bond_dim: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = unsafe { &mut *ptr };
        let state_inner = state.inner_mut();
        let batch_eval = make_batch_eval_closure(eval_cb, user_data);
        let options = make_options(tolerance, max_bond_dim, 1, true);

        let res = match proposer_kind {
            t4a_treetci_proposer_kind::Default => {
                let proposer = DefaultProposer;
                tensor4all_treetci::optimize_with_proposer(
                    state_inner, batch_eval, &options, &proposer,
                )
            }
            t4a_treetci_proposer_kind::Simple => {
                let proposer = SimpleProposer::default();
                tensor4all_treetci::optimize_with_proposer(
                    state_inner, batch_eval, &options, &proposer,
                )
            }
            t4a_treetci_proposer_kind::TruncatedDefault => {
                let proposer = TruncatedDefaultProposer::default();
                tensor4all_treetci::optimize_with_proposer(
                    state_inner, batch_eval, &options, &proposer,
                )
            }
        };

        match res {
            Ok(_) => T4A_SUCCESS,
            Err(e) => err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo nextest run --release -p tensor4all-capi test_sweep 2>&1 | tail -10`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-capi/src/treetci.rs
git commit -m "feat(capi): add TreeTCI sweep function"
```

---

## Task 6: State inspection functions

**Files:**
- Modify: `crates/tensor4all-capi/src/treetci.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
    #[test]
    fn test_state_inspection() {
        let edges = sample_edges();
        let graph = t4a_treetci_graph_new(7, edges.as_ptr(), 6);
        let local_dims: Vec<libc::size_t> = vec![2; 7];
        let state = t4a_treetci_f64_new(local_dims.as_ptr(), 7, graph);

        let pivot: Vec<libc::size_t> = vec![0; 7];
        t4a_treetci_f64_add_global_pivots(state, pivot.as_ptr(), 7, 1);

        // Run a few sweeps
        for _ in 0..4 {
            t4a_treetci_f64_sweep(
                state,
                product_batch_eval,
                std::ptr::null_mut(),
                t4a_treetci_proposer_kind::Default,
                1e-12,
                0,
            );
        }

        // max_bond_error
        let mut error: libc::c_double = 0.0;
        let status = t4a_treetci_f64_max_bond_error(state, &mut error);
        assert_eq!(status, T4A_SUCCESS);
        assert!(error < 1e-10, "error = {}", error);

        // max_rank
        let mut rank: libc::size_t = 0;
        let status = t4a_treetci_f64_max_rank(state, &mut rank);
        assert_eq!(status, T4A_SUCCESS);
        assert!(rank >= 1);

        // max_sample_value
        let mut max_val: libc::c_double = 0.0;
        let status = t4a_treetci_f64_max_sample_value(state, &mut max_val);
        assert_eq!(status, T4A_SUCCESS);
        assert!(max_val > 0.0);

        // bond_dims: query size first
        let mut n_edges: libc::size_t = 0;
        let status = t4a_treetci_f64_bond_dims(
            state,
            std::ptr::null_mut(),
            0,
            &mut n_edges,
        );
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(n_edges, 6);

        // bond_dims: fill buffer
        let mut dims = vec![0usize; n_edges];
        let status = t4a_treetci_f64_bond_dims(
            state,
            dims.as_mut_ptr(),
            n_edges,
            &mut n_edges,
        );
        assert_eq!(status, T4A_SUCCESS);
        for &d in &dims {
            assert!(d >= 1);
        }

        t4a_treetci_f64_release(state);
        t4a_treetci_graph_release(graph);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tensor4all-capi test_state_inspection 2>&1 | tail -10`

Expected: FAIL — inspection functions not defined.

- [ ] **Step 3: Implement state inspection functions**

Add to `treetci.rs`:

```rust
// ============================================================================
// State inspection
// ============================================================================

/// Get the maximum bond error across all edges.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_max_bond_error(
    ptr: *const t4a_treetci_f64,
    out: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = unsafe { &*ptr };
        unsafe { *out = state.inner().max_bond_error() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the maximum rank (bond dimension) across all edges.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_max_rank(
    ptr: *const t4a_treetci_f64,
    out: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = unsafe { &*ptr };
        unsafe { *out = state.inner().max_rank() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the maximum observed sample value (used for normalization).
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_max_sample_value(
    ptr: *const t4a_treetci_f64,
    out: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = unsafe { &*ptr };
        unsafe { *out = state.inner().max_sample_value };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the bond dimensions (ranks) at each edge.
///
/// Uses query-then-fill: pass `out_ranks = NULL` to query `out_n_edges` only.
///
/// # Arguments
/// - `out_ranks`: Output buffer (length >= n_edges), or NULL to query size
/// - `buf_len`: Buffer capacity
/// - `out_n_edges`: Outputs the number of edges
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_bond_dims(
    ptr: *const t4a_treetci_f64,
    out_ranks: *mut libc::size_t,
    buf_len: libc::size_t,
    out_n_edges: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_n_edges.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = unsafe { &*ptr };
        let inner = state.inner();
        let edges = inner.graph.edges();
        let n_edges = edges.len();

        unsafe { *out_n_edges = n_edges };

        if out_ranks.is_null() {
            return T4A_SUCCESS;
        }

        if buf_len < n_edges {
            return err_status(
                format!("Buffer too small: need {}, got {}", n_edges, buf_len),
                crate::T4A_BUFFER_TOO_SMALL,
            );
        }

        for (i, edge) in edges.iter().enumerate() {
            // Bond dim = number of pivot rows for either side of this edge
            let (key_u, _key_v) = inner.graph.subregion_vertices(*edge).unwrap();
            let rank = inner.ijset.get(&key_u).map_or(0, |v| v.len());
            unsafe { *out_ranks.add(i) = rank };
        }

        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo nextest run --release -p tensor4all-capi test_state_inspection 2>&1 | tail -10`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-capi/src/treetci.rs
git commit -m "feat(capi): add TreeTCI state inspection functions"
```

---

## Task 7: Materialization (to_treetn)

**Files:**
- Modify: `crates/tensor4all-capi/src/treetci.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
    #[test]
    fn test_to_treetn() {
        let edges = sample_edges();
        let graph = t4a_treetci_graph_new(7, edges.as_ptr(), 6);
        let local_dims: Vec<libc::size_t> = vec![2; 7];
        let state = t4a_treetci_f64_new(local_dims.as_ptr(), 7, graph);

        let pivot: Vec<libc::size_t> = vec![0; 7];
        t4a_treetci_f64_add_global_pivots(state, pivot.as_ptr(), 7, 1);

        for _ in 0..4 {
            t4a_treetci_f64_sweep(
                state,
                product_batch_eval,
                std::ptr::null_mut(),
                t4a_treetci_proposer_kind::Default,
                1e-12,
                0,
            );
        }

        // Materialize to TreeTN
        let mut treetn_ptr: *mut t4a_treetn = std::ptr::null_mut();
        let status = t4a_treetci_f64_to_treetn(
            state,
            product_batch_eval,
            std::ptr::null_mut(),
            0, // center_site
            &mut treetn_ptr,
        );
        assert_eq!(status, T4A_SUCCESS);
        assert!(!treetn_ptr.is_null());

        // Verify TreeTN is valid by checking vertex count
        let mut n_vertices: libc::size_t = 0;
        let status = crate::t4a_treetn_num_vertices(treetn_ptr, &mut n_vertices);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(n_vertices, 7);

        crate::t4a_treetn_release(treetn_ptr);
        t4a_treetci_f64_release(state);
        t4a_treetci_graph_release(graph);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tensor4all-capi test_to_treetn 2>&1 | tail -10`

Expected: FAIL — `t4a_treetci_f64_to_treetn` not defined.

- [ ] **Step 3: Implement materialization**

Add to `treetci.rs`:

```rust
// ============================================================================
// Materialization
// ============================================================================

/// Materialize the converged TreeTCI state into a TreeTN.
///
/// Internally re-evaluates tensor values using the batch callback and
/// performs LU factorization to construct per-vertex tensors.
///
/// # Arguments
/// - `ptr`: State handle (const — state is not modified)
/// - `eval_cb`: Batch evaluation callback
/// - `user_data`: User data passed to callback
/// - `center_site`: BFS root site for materialization
/// - `out_treetn`: Output TreeTN handle pointer
///
/// # Returns
/// The result is a `t4a_treetn` handle. Release with `t4a_treetn_release`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_to_treetn(
    ptr: *const t4a_treetci_f64,
    eval_cb: TreeTciBatchEvalCallback,
    user_data: *mut c_void,
    center_site: libc::size_t,
    out_treetn: *mut *mut t4a_treetn,
) -> StatusCode {
    if ptr.is_null() || out_treetn.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = unsafe { &*ptr };
        let batch_eval = make_batch_eval_closure(eval_cb, user_data);

        match tensor4all_treetci::to_treetn(state.inner(), batch_eval, Some(center_site)) {
            Ok(treetn) => {
                unsafe { *out_treetn = Box::into_raw(Box::new(t4a_treetn::new(treetn))) };
                T4A_SUCCESS
            }
            Err(e) => err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}
```

Note: `t4a_treetn::new(treetn)` — verify that `t4a_treetn` wraps `TreeTN<TensorDynLen, usize>` (which is `DefaultTreeTN<usize>`). The inner type alias may differ. Check `types.rs` for the exact inner type and adjust if needed. If `t4a_treetn` wraps a different type (e.g. `DefaultTreeTN<usize>` which may be a type alias), ensure the `to_treetn` return type matches.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo nextest run --release -p tensor4all-capi test_to_treetn 2>&1 | tail -10`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-capi/src/treetci.rs
git commit -m "feat(capi): add TreeTCI materialization to TreeTN"
```

---

## Task 8: High-level convenience function

**Files:**
- Modify: `crates/tensor4all-capi/src/treetci.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
    #[test]
    fn test_crossinterpolate_tree_f64() {
        let edges = sample_edges();
        let graph = t4a_treetci_graph_new(7, edges.as_ptr(), 6);
        let local_dims: Vec<libc::size_t> = vec![2; 7];
        let initial_pivot: Vec<libc::size_t> = vec![0; 7];

        let max_iter: libc::size_t = 10;
        let mut out_treetn: *mut t4a_treetn = std::ptr::null_mut();
        let mut out_ranks = vec![0usize; max_iter];
        let mut out_errors = vec![0.0f64; max_iter];
        let mut out_n_iters: libc::size_t = 0;

        let status = t4a_crossinterpolate_tree_f64(
            product_batch_eval,
            std::ptr::null_mut(),
            local_dims.as_ptr(),
            7,
            graph,
            initial_pivot.as_ptr(),
            1, // n_pivots
            t4a_treetci_proposer_kind::Default,
            1e-12,  // tolerance
            0,      // max_bond_dim (unlimited)
            max_iter,
            1,      // normalize_error = true
            0,      // center_site
            &mut out_treetn,
            out_ranks.as_mut_ptr(),
            out_errors.as_mut_ptr(),
            &mut out_n_iters,
        );

        assert_eq!(status, T4A_SUCCESS);
        assert!(!out_treetn.is_null());
        assert!(out_n_iters > 0);
        assert!(out_n_iters <= max_iter);

        // Verify convergence
        let actual_iters = out_n_iters;
        let last_error = out_errors[actual_iters - 1];
        assert!(last_error < 1e-10, "last_error = {}", last_error);

        // Verify TreeTN
        let mut n_vertices: libc::size_t = 0;
        crate::t4a_treetn_num_vertices(out_treetn, &mut n_vertices);
        assert_eq!(n_vertices, 7);

        crate::t4a_treetn_release(out_treetn);
        t4a_treetci_graph_release(graph);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tensor4all-capi test_crossinterpolate_tree_f64 2>&1 | tail -10`

Expected: FAIL — function not defined.

- [ ] **Step 3: Implement high-level function**

Add to `treetci.rs`:

```rust
// ============================================================================
// High-level convenience function
// ============================================================================

/// Run TreeTCI to convergence and return a TreeTN.
///
/// Equivalent to: new → add_pivots → sweep loop → materialize.
///
/// # Arguments
/// - `eval_cb`: Batch evaluation callback
/// - `user_data`: User data passed to callback
/// - `local_dims`: Local dimension at each site (length = n_sites)
/// - `n_sites`: Number of sites
/// - `graph`: Tree graph handle
/// - `initial_pivots_flat`: Column-major (n_sites, n_pivots), or NULL for empty
/// - `n_pivots`: Number of initial pivots
/// - `proposer_kind`: Proposer selection
/// - `tolerance`: Relative tolerance
/// - `max_bond_dim`: Maximum bond dimension (0 = unlimited)
/// - `max_iter`: Maximum number of iterations
/// - `normalize_error`: Whether to normalize errors (0=false, 1=true)
/// - `center_site`: Materialization center site
/// - `out_treetn`: Output TreeTN handle
/// - `out_ranks`: Buffer for max rank per iteration (length >= max_iter), or NULL
/// - `out_errors`: Buffer for normalized error per iteration (length >= max_iter), or NULL
/// - `out_n_iters`: Output: actual number of iterations performed
#[unsafe(no_mangle)]
pub extern "C" fn t4a_crossinterpolate_tree_f64(
    eval_cb: TreeTciBatchEvalCallback,
    user_data: *mut c_void,
    local_dims: *const libc::size_t,
    n_sites: libc::size_t,
    graph: *const t4a_treetci_graph,
    initial_pivots_flat: *const libc::size_t,
    n_pivots: libc::size_t,
    proposer_kind: t4a_treetci_proposer_kind,
    tolerance: libc::c_double,
    max_bond_dim: libc::size_t,
    max_iter: libc::size_t,
    normalize_error: libc::c_int,
    center_site: libc::size_t,
    out_treetn: *mut *mut t4a_treetn,
    out_ranks: *mut libc::size_t,
    out_errors: *mut libc::c_double,
    out_n_iters: *mut libc::size_t,
) -> StatusCode {
    if local_dims.is_null() || graph.is_null() || out_treetn.is_null() || out_n_iters.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        // Parse local_dims
        let dims: Vec<usize> = (0..n_sites)
            .map(|i| unsafe { *local_dims.add(i) })
            .collect();

        // Clone graph
        let g = unsafe { &*graph };
        let graph_clone = g.inner().clone();

        // Create state
        let mut state = match SimpleTreeTci::new(dims, graph_clone) {
            Ok(s) => s,
            Err(e) => return err_status(e, T4A_INTERNAL_ERROR),
        };

        // Add initial pivots
        if !initial_pivots_flat.is_null() && n_pivots > 0 {
            let pivots: Vec<Vec<usize>> = (0..n_pivots)
                .map(|p| {
                    (0..n_sites)
                        .map(|s| unsafe { *initial_pivots_flat.add(s + n_sites * p) })
                        .collect()
                })
                .collect();
            if let Err(e) = state.add_global_pivots(&pivots) {
                return err_status(e, T4A_INTERNAL_ERROR);
            }
        }

        // Run optimization
        let batch_eval = make_batch_eval_closure(eval_cb, user_data);
        let options = make_options(
            tolerance,
            max_bond_dim,
            max_iter,
            normalize_error != 0,
        );

        let (ranks, errors) = match proposer_kind {
            t4a_treetci_proposer_kind::Default => {
                let proposer = DefaultProposer;
                match tensor4all_treetci::optimize_with_proposer(
                    &mut state, &batch_eval, &options, &proposer,
                ) {
                    Ok(r) => r,
                    Err(e) => return err_status(e, T4A_INTERNAL_ERROR),
                }
            }
            t4a_treetci_proposer_kind::Simple => {
                let proposer = SimpleProposer::default();
                match tensor4all_treetci::optimize_with_proposer(
                    &mut state, &batch_eval, &options, &proposer,
                ) {
                    Ok(r) => r,
                    Err(e) => return err_status(e, T4A_INTERNAL_ERROR),
                }
            }
            t4a_treetci_proposer_kind::TruncatedDefault => {
                let proposer = TruncatedDefaultProposer::default();
                match tensor4all_treetci::optimize_with_proposer(
                    &mut state, &batch_eval, &options, &proposer,
                ) {
                    Ok(r) => r,
                    Err(e) => return err_status(e, T4A_INTERNAL_ERROR),
                }
            }
        };

        let n_iters = ranks.len();
        unsafe { *out_n_iters = n_iters };

        // Copy ranks and errors to output buffers
        if !out_ranks.is_null() {
            for (i, &r) in ranks.iter().enumerate() {
                unsafe { *out_ranks.add(i) = r };
            }
        }
        if !out_errors.is_null() {
            for (i, &e) in errors.iter().enumerate() {
                unsafe { *out_errors.add(i) = e };
            }
        }

        // Materialize
        match tensor4all_treetci::to_treetn(&state, &batch_eval, Some(center_site)) {
            Ok(treetn) => {
                unsafe { *out_treetn = Box::into_raw(Box::new(t4a_treetn::new(treetn))) };
                T4A_SUCCESS
            }
            Err(e) => err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo nextest run --release -p tensor4all-capi test_crossinterpolate_tree_f64 2>&1 | tail -10`

Expected: PASS.

- [ ] **Step 5: Run all TreeTCI tests**

Run: `cargo nextest run --release -p tensor4all-capi treetci 2>&1 | tail -20`

Expected: All tests PASS (test_graph_new_and_query, test_graph_invalid_disconnected, test_state_new_and_add_pivots, test_sweep, test_state_inspection, test_to_treetn, test_crossinterpolate_tree_f64).

- [ ] **Step 6: Run clippy and fmt**

Run:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -20
```

Fix any warnings or formatting issues.

- [ ] **Step 7: Commit**

```bash
git add crates/tensor4all-capi/src/treetci.rs
git commit -m "feat(capi): add TreeTCI high-level convenience function"
```

---

## Task 9: Final validation — full test suite

**Files:** None (validation only)

- [ ] **Step 1: Run the full capi test suite**

Run: `cargo nextest run --release -p tensor4all-capi 2>&1 | tail -30`

Expected: All existing tests + all new TreeTCI tests PASS. No regressions.

- [ ] **Step 2: Run the full workspace test suite**

Run: `cargo nextest run --release --workspace 2>&1 | tail -30`

Expected: No regressions across the workspace.

- [ ] **Step 3: Run CI checks**

Run: `cargo xtask ci 2>&1 | tail -30`

Expected: fmt, clippy, tests, docs all pass.

- [ ] **Step 4: Update API docs**

Run: `cargo run -p api-dump --release -- . -o docs/api 2>&1 | tail -10`

Verify `docs/api/tensor4all_capi.md` now includes the TreeTCI functions.

- [ ] **Step 5: Commit docs update**

```bash
git add docs/api/
git commit -m "docs(capi): update API reference with TreeTCI functions"
```
