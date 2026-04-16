# Tensor and TensorTrain Arithmetic, Norm, and isapprox

**Issue:** #40 (TensorTrain arithmetic) + #42 (llim/rlim sync)
**Date:** 2026-04-16

## Overview

Implement arithmetic operations, norm, inner product, and approximate comparison
for both `Core.Tensor` and `TensorNetworks.TensorTrain`. The TensorTrain
operations use the Rust backend via C API. The design follows ITensorMPS.jl
semantics where applicable.

## Phase 0: Fix #42 — Canonical Region Sync

### Problem

`TensorTrain.llim`/`rlim` are not synchronized with the Rust backend's
canonical metadata after C API operations. `_treetn_from_handle` takes
caller-supplied `llim`/`rlim` kwargs, and `apply()` blindly copies the input
state's values onto the result.

### Rust C API addition

```c
StatusCode t4a_treetn_canonical_region(
    const struct t4a_treetn *treetn,
    const size_t **out_vertices,
    size_t *out_len
);
```

Returns the canonical region vertices. This is a general (non-chain-specific)
accessor that works for any tree topology.

### Julia-side changes

1. **`_treetn_from_handle`**: Remove `llim`/`rlim` kwargs. Always query
   `t4a_treetn_canonical_region` from the handle and derive `llim`/`rlim`:
   - Single vertex `{c}` in canonical region: `llim = c - 1`, `rlim = c + 1`
   - Empty or multi-vertex region: `llim = 0`, `rlim = length + 1`

2. **`setindex!(tt::TensorTrain, val, i)`**: After setting `tt.data[i]`,
   invalidate to `tt.llim = 0`, `tt.rlim = length(tt) + 1`.

3. **`apply()`**: Stop explicitly passing `state.llim`/`state.rlim` to
   `_treetn_from_handle`. The automatic query handles it.

### Testing rule

Every operation that returns or mutates a `TensorTrain` must verify
`llim`/`rlim` correctness:
- After `apply()`: check returned values match backend canonical state
- After `setindex!`: check reset to `0` / `length+1`
- After `orthogonalize` (if exposed): check they reflect the new center
- Roundtrip: orthogonalize, then apply, then verify consistency

This rule applies to all subsequent phases — every `+`, `-`, `scale`, etc. must
test `llim`/`rlim` on the result.

## Phase 1: New C API Functions in tensor4all-rs

### Functions

```c
// Add two tree tensor networks (direct sum of bond dimensions, exact)
StatusCode t4a_treetn_add(
    const struct t4a_treetn *a,
    const struct t4a_treetn *b,
    struct t4a_treetn **out
);

// Scale a tree tensor network by a complex scalar
StatusCode t4a_treetn_scale(
    const struct t4a_treetn *treetn,
    double re, double im,
    struct t4a_treetn **out
);
```

These wrap the existing Rust `TensorTrain::add()` and `TensorTrain::scale()`
methods. No new algorithm work is needed, only C API plumbing.

### Cross-repo workflow

Phase 0 (`canonical_region` getter) and Phase 1 (`add`/`scale`) can be combined
into a single tensor4all-rs PR. Merge that first, then update the pin in
Tensor4all.jl `deps/build.jl`.

## Phase 2: Tensor-level Arithmetic

### Operations on `Core.Tensor`

| Function | Behavior |
|---|---|
| `+(a, b)` | Element-wise add. Auto-permute `b` to match `a`'s index order. |
| `-(a, b)` | Element-wise subtract. Same auto-permute. |
| `*(α::Number, t)`, `*(t, α::Number)` | Scalar multiply on data array. |
| `/(t, α::Number)` | Scalar divide. |
| `-(t)` | Unary negation (`-1 * t`). |
| `norm(t)` | `LinearAlgebra.norm(t.data)` (Frobenius norm). |
| `isapprox(a, b)` | Auto-permute, then `isapprox(a.data, b.data; atol, rtol)`. |
| `contract(a, b)` | Via `t4a_tensor_contract` C API. |

### Index matching for +, -, isapprox

When two tensors have the same indices in different order (e.g., `a` has
`[i, j]` and `b` has `[j, i]`), automatically permute `b`'s data to match
`a`'s index order before the element-wise operation. This follows ITensors.jl
semantics — indices define the meaning, storage order is transparent.

If the index sets do not match, throw `ArgumentError` with an actionable
message listing the mismatched indices.

### contract via C API

- Create backend tensor handles for both inputs
- Call `t4a_tensor_contract` (automatic index matching in Rust)
- Reconstruct Julia `Tensor` from the result handle

## Phase 3: TensorTrain-level Arithmetic

### Operations on `TensorNetworks.TensorTrain`

| Function | Implementation | Result `llim`/`rlim` |
|---|---|---|
| `+(a, b)` | C API `t4a_treetn_add` | From backend canonical region |
| `-(a, b)` | `a + (-1 * b)` via scale + add | From backend canonical region |
| `-(tt)` | `-1 * tt` | From backend canonical region |
| `*(α::Number, tt)`, `*(tt, α::Number)` | C API `t4a_treetn_scale` | From backend canonical region |
| `/(tt, α::Number)` | `tt * (1/α)` | From backend canonical region |
| `dot(a, b)` / `inner(a, b)` | C API `t4a_treetn_inner` | N/A (returns scalar) |
| `norm(tt)` | C API `t4a_treetn_norm` | N/A (returns scalar) |
| `isapprox(a, b; atol, rtol)` | `norm(a - b) <= max(atol, rtol * max(norm(a), norm(b)))` | N/A (returns bool) |
| `dist(a, b)` | `sqrt(abs(dot(a,a) + dot(b,b) - 2*real(dot(a,b))))` | N/A (returns scalar) |
| `add(a, b; maxdim, rtol)` | C API add, then `SimpleTT.compress!` with `:SVD` if truncation params given | From backend canonical region |

### Design notes

- All operations returning `TensorTrain` go through `_treetn_from_handle`,
  which automatically syncs `llim`/`rlim` from the backend (Phase 0).
- `isapprox` follows ITensorMPS.jl signature:
  `isapprox(a, b; atol=0, rtol=Base.rtoldefault(...))`.
- `dist` uses the efficient 3-inner-product formula (same as ITensorMPS.jl's
  `dist`), avoiding bond dimension growth.
- Scalar multiply scales the ortho center site (using `llim`/`rlim` to identify
  it), matching ITensorMPS.jl behavior. This depends on Phase 0 for correct
  `llim`/`rlim`.
- Every operation that returns a `TensorTrain` must test `llim`/`rlim`
  correctness.

## Scope boundaries

- `SimpleTT.TensorTrain{T,N}` is not in scope. Arithmetic is added to
  `Core.Tensor` and `TensorNetworks.TensorTrain` only.
- No truncation is applied by default on `+`/`-`. The separate `add()` function
  accepts optional truncation parameters.
- No changes to `LinearOperator` or `apply()` semantics beyond the `llim`/`rlim`
  fix.

## Dependencies

- Phase 0 + Phase 1 require a tensor4all-rs PR (C API additions) merged first.
- Phase 2 and Phase 3 depend on the updated pin in `deps/build.jl`.
- Phase 3 scalar multiply depends on Phase 0 (`llim`/`rlim` correctness).
