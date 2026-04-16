# Tensor and TensorTrain Arithmetic, Norm, and isapprox

**Issue:** #40 (TensorTrain arithmetic) + #42 (llim/rlim sync)
**Date:** 2026-04-16

## Overview

Implement arithmetic operations, norm, inner product, and approximate comparison
for both `Core.Tensor` and `TensorNetworks.TensorTrain`. The TensorTrain
operations use the Rust backend via C API. The design follows ITensorMPS.jl
semantics where practical, with intentional divergences documented below.

## Intentional divergences from ITensorMPS.jl

| Behavior | ITensorMPS.jl | This spec | Rationale |
|---|---|---|---|
| `+(::MPS...)` | Compresses with `cutoff=1e-15` | Exact (no truncation) | Predictability; users compress explicitly |
| `setindex!` ortho | Widens limits to include modified site | Widens limits to include modified site | Matches ITensorMPS |
| Scalar multiply | Scales first site in `ortho_lims` range | Scales first site in `ortho_lims` range | Matches ITensorMPS |

## Phase 0: Fix #42 — Canonical Region Sync

### Problem

`TensorTrain.llim`/`rlim` are not synchronized with the Rust backend's
canonical metadata after C API operations. `_treetn_from_handle` takes
caller-supplied `llim`/`rlim` kwargs, and `apply()` blindly copies the input
state's values onto the result.

### Rust improvements required

**1. Efficient tree inner product in `tensor4all-treetn`:**

The current `TreeTN::inner` uses naive full contraction, which is too expensive
for chains. Implement a general tree contraction algorithm (leaf-to-root
environment accumulation) that works for any tree topology. This replaces the
naive implementation and benefits all downstream users (C API, itensorlike).

**2. `TreeTN::scale` method in `tensor4all-treetn`:**

Currently `scale` only exists on `tensor4all-itensorlike::TensorTrain`, not on
`TreeTN`. Add a `scale` method to `TreeTN` that scales a single node tensor
by a complex scalar.

### Rust C API additions

```c
// Get the canonical region vertices (copy-out buffer pattern)
// Vertices are returned sorted ascending.
// Call with buf=NULL to query the required length via out_len.
StatusCode t4a_treetn_canonical_region(
    const struct t4a_treetn *treetn,
    size_t *buf,
    size_t buf_len,
    size_t *out_len
);

// Add two tree tensor networks (direct sum of bond dimensions)
// Optional truncation: any nonzero truncation param enables truncation.
// cutoff is converted to rtol = sqrt(cutoff) following TruncationParams
// semantics. If both rtol and cutoff are set, rtol takes precedence.
// maxdim caps the bond dimension independently.
StatusCode t4a_treetn_add(
    const struct t4a_treetn *a,
    const struct t4a_treetn *b,
    double rtol,
    double cutoff,
    size_t maxdim,
    struct t4a_treetn **out
);

// Scale a tree tensor network by a complex scalar
StatusCode t4a_treetn_scale(
    const struct t4a_treetn *treetn,
    double re, double im,
    struct t4a_treetn **out
);
```

The `canonical_region` API uses the copy-out buffer pattern consistent with
existing C API functions (`t4a_index_tags`, `t4a_tensor_dims`,
`t4a_treetn_neighbors`, `t4a_treetn_siteinds`). The returned vertices are
sorted ascending because the Rust backend stores the region as a `HashSet`.

### Julia-side changes

1. **`_treetn_from_handle`**: Remove `llim`/`rlim` kwargs. Always query
   `t4a_treetn_canonical_region` from the handle and derive `llim`/`rlim`:
   - Single vertex `{c}` (0-based backend index): `llim = c`, `rlim = c + 2`
   - Empty or multi-vertex region: `llim = 0`, `rlim = length + 1`

2. **`setindex!(tt::TensorTrain, val, i)`**: After setting `tt.data[i]`, widen
   ortho limits to include the modified site (ITensorMPS semantics):
   `tt.llim = min(tt.llim, i - 1)`, `tt.rlim = max(tt.rlim, i + 1)`.

3. **`apply()`**: Stop explicitly passing `state.llim`/`state.rlim` to
   `_treetn_from_handle`. The automatic query handles it.

4. **Audit existing code**: Review all Julia code that constructs `TensorTrain`
   with hand-written limits (e.g., `matchsiteinds` at
   `src/TensorNetworks/matchsiteinds.jl:228,259`) for correctness.
   Rule: `matchsiteinds` and similar embedding operations that change the chain
   structure must reset to `llim = 0`, `rlim = length + 1` since the
   original canonical form does not survive structural changes.

### Testing rule

Every operation that returns or mutates a `TensorTrain` must verify
`llim`/`rlim` correctness:
- After `apply()`: check returned values match backend canonical state
- After `setindex!`: check limits widen correctly
- After `orthogonalize` (if exposed): check they reflect the new center
- Roundtrip: orthogonalize, then apply, then verify consistency

This rule applies to all subsequent phases — every `+`, `-`, `scale`, etc. must
test `llim`/`rlim` on the result.

### Cross-repo workflow

All Rust improvements (tree inner product, TreeTN::scale, C API additions for
canonical_region/add/scale) are combined into a single tensor4all-rs PR. Merge
that first, then update the pin in Tensor4all.jl `deps/build.jl`.

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

### Julia-side validation

Before calling any C API function for TT operations, validate in Julia:

- **Empty trains**: Throw `ArgumentError("TensorTrain must not be empty")`
  for `norm`, `inner`, `+`, `-`, `scale` on empty `TensorTrain(Tensor[])`.
- **Length mismatch**: Throw `DimensionMismatch` for `+`, `-`, `inner` when
  `length(a) != length(b)`.
- **Site index compatibility**: For `+`, `-`, `inner`, verify that site indices
  match at each position. Throw `ArgumentError` with the mismatched position
  and indices if they differ.
- **MPS and MPO support**: All operations support both MPS-like (1 site index
  per tensor) and MPO-like (2 site indices per tensor) chains. The two operands
  must have the same structure (both MPS-like or both MPO-like).

### Operations on `TensorNetworks.TensorTrain`

| Function | Implementation | Result `llim`/`rlim` |
|---|---|---|
| `+(a, b)` | C API `t4a_treetn_add` (exact) | From backend canonical region |
| `-(a, b)` | `a + (-1 * b)` via scale + add | From backend canonical region |
| `-(tt)` | `-1 * tt` | From backend canonical region |
| `*(α::Number, tt)`, `*(tt, α::Number)` | C API `t4a_treetn_scale` | From backend canonical region |
| `/(tt, α::Number)` | `tt * (1/α)` | From backend canonical region |
| `dot(a, b)` / `inner(a, b)` | C API `t4a_treetn_inner` | N/A (returns scalar) |
| `norm(tt)` | C API `t4a_treetn_norm` | N/A (returns scalar) |
| `isapprox(a, b; atol, rtol)` | `d = norm(a - b); isfinite(d) ? d <= max(atol, rtol * max(norm(a), norm(b))) : error(...)` | N/A (returns bool) |
| `dist(a, b)` | `sqrt(abs(dot(a,a) + dot(b,b) - 2*real(dot(a,b))))` | N/A (returns scalar) |
| `add(a, b; maxdim=0, rtol=0.0, cutoff=0.0)` | C API `t4a_treetn_add` with truncation params | From backend canonical region |

### Design notes

- All operations returning `TensorTrain` go through `_treetn_from_handle`,
  which automatically syncs `llim`/`rlim` from the backend (Phase 0).
- `isapprox` follows ITensorMPS.jl signature and semantics:
  `isapprox(a, b; atol=0, rtol=Base.rtoldefault(...))`.
  Uses `norm(a - b)` (exact subtraction, matching ITensorMPS.jl).
  Includes the `isfinite(d)` check and error on nonfinite distance.
- `dist` uses the efficient 3-inner-product formula (same as ITensorMPS.jl's
  `dist`), avoiding bond dimension growth.
- Scalar multiply scales the first site in the `ortho_lims` range (matching
  ITensorMPS.jl behavior). This depends on Phase 0 for correct `llim`/`rlim`.
- `+` is exact by default (intentional divergence from ITensorMPS.jl which
  compresses with `cutoff=1e-15`). Use `add(a, b; ...)` for truncated addition.
- Every operation that returns a `TensorTrain` must test `llim`/`rlim`
  correctness.

### Negative-path tests

In addition to `llim`/`rlim` checks, test:
- Tensor index-set mismatch between `a` and `b`
- TensorTrain length mismatch
- Empty trains
- Complex scalars
- Nonfinite `isapprox` path (e.g., divergent norm)

## Scope boundaries

- `SimpleTT.TensorTrain{T,N}` is not in scope. Arithmetic is added to
  `Core.Tensor` and `TensorNetworks.TensorTrain` only.
- No truncation is applied by default on `+`/`-`. The separate `add()` function
  accepts optional truncation parameters passed to the Rust backend.
- No changes to `LinearOperator` or `apply()` semantics beyond the `llim`/`rlim`
  fix.

## Dependencies

- Phase 0 requires a tensor4all-rs PR (tree inner product algorithm, TreeTN
  scale, C API additions) merged first.
- Phase 2 does not depend on the new Rust PR — `t4a_tensor_contract` already
  exists, and Tensor arithmetic is Julia-owned.
- Phase 3 depends on the updated pin in `deps/build.jl` (for add, scale,
  canonical_region C API).
- Phase 3 scalar multiply depends on Phase 0 (`llim`/`rlim` correctness).
