# BubbleTeaCI Migration APIs — Design Spec

**Date:** 2026-04-16
**Related:** BubbleTeaCI migration, issue #45 (persistent handle — future)

## Overview

Implement all missing Tensor4all.jl APIs required for BubbleTeaCI migration.
Architecture stays copy-based (no persistent handle). DMRG sweeps are Rust-side
only. Julia provides high-level API that delegates to Rust C API.

## Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Data ownership | Copy-based (current) | Simplicity; DMRG not in Julia |
| `orthogonalize`/`truncate` | Non-destructive (new TT) | Copy-based makes in-place pointless |
| `dag` | Pure Julia | No QN system; just `conj(data)` |
| `svd`/`qr` on Tensor | Rust C API | Consistency with backend |
| QuanticsTransform | Rust C API materialize | Existing `t4a_qtransform_*` C API |
| SimpleTT | Keep as-is | Coexists for TCI/pure-Julia use |
| BubbleTeaCI milestone | Out of scope | Separate effort |

## Rust C API Additions (tensor4all-rs PR)

Only two new functions needed. Everything else already exists.

### t4a_tensor_svd

```c
StatusCode t4a_tensor_svd(
    const struct t4a_tensor *tensor,
    const struct t4a_index *const *left_inds,
    size_t n_left,
    double rtol,
    double cutoff,
    size_t maxdim,
    struct t4a_tensor **out_u,
    struct t4a_tensor **out_s,
    struct t4a_tensor **out_v
);
```

Wraps existing `tensor4all_core::svd`. Returns U, S (diagonal), V tensors with
a new bond index. Truncation controlled by rtol/cutoff/maxdim following
`TruncationParams` semantics.

### t4a_tensor_qr

```c
StatusCode t4a_tensor_qr(
    const struct t4a_tensor *tensor,
    const struct t4a_index *const *left_inds,
    size_t n_left,
    struct t4a_tensor **out_q,
    struct t4a_tensor **out_r
);
```

Wraps existing `tensor4all_core::qr`. Returns Q, R tensors with a new bond
index.

## Julia API Additions (Tensor4all.jl PR)

### Core.Tensor

| Function | Implementation | Description |
|---|---|---|
| `dag(t::Tensor)` | Pure Julia | `Tensor(conj(t.data), inds(t))` |
| `svd(t, left_inds; rtol, cutoff, maxdim)` | C API `t4a_tensor_svd` | Returns `(U, S, V)` as Tensors |
| `qr(t, left_inds)` | C API `t4a_tensor_qr` | Returns `(Q, R)` as Tensors |
| `Array(t, inds...)` | Pure Julia | Permute data to requested index order |

`svd` and `qr` accept `left_inds` as a vector of `Index` specifying which
indices go to the left factor. The remaining indices go to the right factor.
A new bond `Index` is created for the shared dimension.

### TensorNetworks.TensorTrain

| Function | Implementation | Description |
|---|---|---|
| `dag(tt)` | Pure Julia | Apply `dag` to each site tensor |
| `orthogonalize(tt, site)` | C API `t4a_treetn_orthogonalize` | Non-destructive, returns new TT |
| `truncate(tt; rtol, cutoff, maxdim)` | C API `t4a_treetn_truncate` | Non-destructive, returns new TT |
| `linkinds(tt)` | Pure Julia | Bond indices between adjacent tensors |
| `linkinds(tt, i)` | Pure Julia | Bond index between site i and i+1 |
| `linkdims(tt)` | Pure Julia | `dim.(linkinds(tt))` |
| `siteinds(tt)` | Pure Julia | Site indices at each tensor (non-bond) |
| `siteinds(tt, i)` | Pure Julia | Site indices at tensor i |

#### linkinds algorithm

For each pair of adjacent tensors `tt[i]` and `tt[i+1]`, the link index is
`commoninds(inds(tt[i]), inds(tt[i+1]))`. Returns a vector of length
`length(tt) - 1`.

#### siteinds algorithm

For each tensor `tt[i]`, the site indices are the indices that are NOT link
indices (not shared with any neighbor). For boundary tensors (i=1, i=end),
check only the one neighbor. For interior tensors, exclude indices shared
with either neighbor.

#### orthogonalize flow

1. `_new_treetn_handle(tt)` — copy to Rust
2. `ccall(t4a_treetn_orthogonalize, handle, site-1, form)` — mutates handle
3. `_treetn_from_handle(handle)` — copy back with canonical region sync
4. Release handle

#### truncate flow

1. `_new_treetn_handle(tt)` — copy to Rust
2. `ccall(t4a_treetn_truncate, handle, rtol, cutoff, maxdim)` — mutates handle
3. `_treetn_from_handle(handle)` — copy back with canonical region sync
4. Release handle

### QuanticsTransform Materialization

Currently, operator functions return metadata-only `LinearOperator` with
`mpo=nothing`. Change them to call existing C API functions and return
`LinearOperator` with a real `TensorTrain` MPO.

**Existing C API functions to wire up:**

| Julia function | C API | Notes |
|---|---|---|
| `shift_operator(r, offset; bc)` | `t4a_qtransform_shift_materialize` | |
| `flip_operator(r; bc)` | `t4a_qtransform_flip_materialize` | |
| `affine_operator(r, a, a_den, b, b_den; bc)` | `t4a_qtransform_affine_materialize` | |
| `fourier_operator(r; forward, maxdim, tolerance)` | `t4a_qtransform_fourier_materialize` | |
| `cumsum_operator(r)` | Compose from existing operators or add C API | Check availability |
| `phase_rotation_operator(r, theta)` | Compose or add C API | Check availability |

**Multivar versions** use the same C API with `target_var` parameter.

Each function:
1. Creates a `t4a_qtt_layout` handle describing the quantics layout
2. Calls the appropriate `t4a_qtransform_*_materialize` to get a `t4a_treetn` handle
3. Converts to `TensorTrain` via `_treetn_from_handle`
4. Attaches proper input/output `Index` metadata to `LinearOperator`
5. Binds input/output spaces via `set_iospaces!`

## Error Handling

All new functions follow AGENTS.md rules:

- Validate arguments in Julia before C API calls
- `ArgumentError` for invalid arguments
- `DimensionMismatch` for shape mismatches
- Include actual vs expected values in messages
- Never discard Rust error messages from `last_error_message()`

Specific validations:
- `svd`/`qr`: `left_inds` must be a subset of `inds(t)`, non-empty, not all indices
- `orthogonalize`: site must be in `1:length(tt)`, TT must be non-empty
- `truncate`: at least one of rtol/cutoff/maxdim must be specified
- `linkinds`/`siteinds`: TT must be non-empty

## Testing

### test/core/tensor_factorize.jl

- SVD of rank-2 tensor (matrix) — verify U*S*V' reconstructs original
- SVD with truncation — verify rank reduction
- SVD of rank-3+ tensor with various left_inds choices
- QR of rank-2 tensor — verify Q*R reconstructs original
- QR of rank-3+ tensor
- `dag` — verify `conj` of data, indices unchanged
- `Array(t, inds...)` — verify permutation matches requested order
- Error paths: invalid left_inds, empty tensor

### test/tensornetworks/orthogonalize_truncate.jl

- `orthogonalize(tt, site)` — verify `llim`/`rlim` correct after
- `truncate(tt; cutoff)` — verify bond dimensions reduced
- `truncate(tt; maxdim)` — verify bond dimensions capped
- Non-destructive: original TT unchanged after orthogonalize/truncate
- Round-trip: orthogonalize → truncate → norm approximately preserved

### test/tensornetworks/queries.jl

- `linkinds(tt)` — verify correct bond indices between sites
- `linkinds(tt, i)` — single bond query
- `linkdims(tt)` — verify dimensions match
- `siteinds(tt)` — verify non-bond indices at each site
- `siteinds(tt, i)` — single site query
- Edge cases: 1-site TT (no links), 2-site TT

### test/quanticstransform/materialize.jl

Full coverage of all operator patterns supported by the C API:

**shift_operator:**
- Positive and negative offsets
- Periodic and open boundary conditions
- Apply to known state, verify against dense reference (R=2-3)

**flip_operator:**
- Periodic and open BC
- Apply to known state, verify bit reversal

**affine_operator:**
- Various a_num/a_den/b_num/b_den combinations
- Periodic and open BC
- Verify against dense affine transform

**fourier_operator:**
- Forward and inverse
- With and without maxdim truncation
- Verify against dense DFT matrix

**cumsum_operator:**
- Basic cumulative sum verification

**phase_rotation_operator:**
- Various theta values
- Verify phase factors on known states

**Multivar versions:**
- Multiple variables, each target variable
- Verify only targeted variable is transformed

Each test:
1. Materialize operator (small R=2-3)
2. Apply to known state via `TensorNetworks.apply`
3. Compare result against dense reference computation

## Dependencies

- tensor4all-rs PR (svd + qr C API) must merge first
- Then update pin in `deps/build.jl` and CI workflow
- QuanticsTransform materialization uses existing C API (no Rust changes needed)

## Scope Boundaries

- BubbleTeaCI migration itself is out of scope
- Persistent handle (issue #45) is future optimization
- SimpleTT stays as-is
- DMRG/sweep algorithms stay Rust-side only
- `combiner` is not needed (BubbleTeaCI handles via reshape)
- `directsum` on Tensor is not needed (TT-level `add` already exists)
