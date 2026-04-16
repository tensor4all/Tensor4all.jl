# Phase 1: Core Migration APIs for BubbleTeaCI

**Date:** 2026-04-16
**Related:** BubbleTeaCI migration, issue #45 (persistent handle — future)

## Overview

Implement core Tensor4all.jl APIs needed for BubbleTeaCI migration: tensor
factorization (SVD, QR), TensorTrain orthogonalization/truncation, index
queries, and QuanticsTransform operator materialization.

This is Phase 1 — it covers the foundation APIs. Additional compatibility
helpers (vararg overloads, mutating variants, etc.) may be added in follow-up
phases as BubbleTeaCI call sites are migrated.

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
| Operator space binding | Caller binds via `set_iospaces!` | AGENTS.md: no auto-binding |
| SimpleTT | Keep as-is | Coexists for TCI/pure-Julia use |
| BubbleTeaCI milestone | Out of scope | Separate effort |

## Rust C API Additions (tensor4all-rs PR)

Two new functions needed. Everything else already exists.

**Already present (PR #420 and earlier):**
- `t4a_treetn_canonical_region` — canonical region query
- `t4a_treetn_add`, `t4a_treetn_scale` — arithmetic
- `t4a_treetn_inner`, `t4a_treetn_norm` — inner product / norm
- `t4a_treetn_orthogonalize` — orthogonalization
- `t4a_treetn_truncate` — truncation with rtol/cutoff/maxdim
- `t4a_tensor_contract` — tensor contraction
- `t4a_qtransform_shift_materialize` — shift operator MPO
- `t4a_qtransform_flip_materialize` — flip operator MPO
- `t4a_qtransform_affine_materialize` — affine operator MPO
- `t4a_qtransform_fourier_materialize` — Fourier operator MPO
- `t4a_qtransform_cumsum_materialize` — cumulative sum operator MPO
- `t4a_qtransform_phase_rotation_materialize` — phase rotation operator MPO

**New (this PR):**

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
| `svd(t, left_inds::Vector{Index}; rtol, cutoff, maxdim)` | C API `t4a_tensor_svd` | Returns `(U, S, V)` as Tensors |
| `svd(t, left_inds::Index...; kwargs...)` | Vararg convenience | Delegates to vector version |
| `qr(t, left_inds::Vector{Index})` | C API `t4a_tensor_qr` | Returns `(Q, R)` as Tensors |
| `qr(t, left_inds::Index...)` | Vararg convenience | Delegates to vector version |
| `Array(t, inds...)` | Pure Julia | Permute data to requested index order |

`svd` and `qr` accept `left_inds` specifying which indices go to the left
factor. The remaining indices go to the right factor. A new bond `Index` is
created for the shared dimension.

### TensorNetworks.TensorTrain

| Function | Implementation | Description |
|---|---|---|
| `dag(tt)` | Pure Julia | Apply `dag` to each site tensor |
| `orthogonalize(tt, site; form=:unitary)` | C API `t4a_treetn_orthogonalize` | Non-destructive, returns new TT |
| `truncate(tt; rtol, cutoff, maxdim)` | C API `t4a_treetn_truncate` | Non-destructive, returns new TT |
| `linkinds(tt)` | Pure Julia | Bond indices between adjacent tensors |
| `linkinds(tt, i)` | Pure Julia | Bond index between site i and i+1 |
| `linkdims(tt)` | Pure Julia | `dim.(linkinds(tt))` |
| `siteinds(tt)` | Pure Julia | Site indices at each tensor (non-bond) |
| `siteinds(tt, i)` | Pure Julia | Site indices at tensor i |

#### linkinds algorithm

For each pair of adjacent tensors `tt[i]` and `tt[i+1]`, the link index is
`commoninds(inds(tt[i]), inds(tt[i+1]))`. Must be exactly one shared index;
throw `ArgumentError` if zero or multiple shared indices found. Returns a
vector of length `length(tt) - 1`.

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

`form` keyword maps to C API enum: `:unitary` → `T4A_CANONICAL_FORM_UNITARY`
(default), `:lu` → `T4A_CANONICAL_FORM_LU`.

#### truncate flow

1. `_new_treetn_handle(tt)` — copy to Rust
2. `ccall(t4a_treetn_truncate, handle, rtol, cutoff, maxdim)` — mutates handle
3. `_treetn_from_handle(handle)` — copy back with canonical region sync
4. Release handle

### QuanticsTransform Materialization

Currently, operator functions return metadata-only `LinearOperator` with
`mpo=nothing`. Change them to call existing C API functions and return
`LinearOperator` with a real `TensorTrain` MPO.

**C API functions to wire up (all already exist):**

| Julia function | C API | Params |
|---|---|---|
| `shift_operator` | `t4a_qtransform_shift_materialize` | `target_var`, `offset`, `bc` |
| `flip_operator` | `t4a_qtransform_flip_materialize` | `target_var`, `bc` |
| `affine_operator` | `t4a_qtransform_affine_materialize` | rational matrix/vector, `m`, `n`, `bc` |
| `fourier_operator` | `t4a_qtransform_fourier_materialize` | `target_var`, `forward`, `maxbonddim`, `tolerance` |
| `cumsum_operator` | `t4a_qtransform_cumsum_materialize` | `target_var` |
| `phase_rotation_operator` | `t4a_qtransform_phase_rotation_materialize` | `target_var`, `theta` |

**Note on operator families:**
- Single-target operators (shift, flip, phase_rotation, cumsum, fourier) take
  `target_var` for multivar support.
- `affine_operator` takes rational matrix/vector data (`a_num`, `a_den`,
  `b_num`, `b_den`, `m`, `n`) — not a simple `target_var`.
- `binaryop_operator` takes `lhs_var` and `rhs_var` — separate signature.
- `binaryop_operator` and `affine_pullback_operator` are deferred from this
  phase if their layout constraints require additional design work.

**Operator space binding contract:**

Constructors return `LinearOperator` with:
- `mpo` set to the materialized `TensorTrain`
- `input_indices` and `output_indices` set to the MPO's internal site indices
- `true_input` and `true_output` left as `nothing` (unbound)

Callers must call `set_iospaces!(op, input_space, output_space)` to bind
external state indices before calling `apply`. This follows AGENTS.md:
"Do not add TensorTrain-based automatic binding for operator I/O spaces."

**Materialization flow:**

Each function:
1. Creates a `t4a_qtt_layout` handle describing the quantics layout
2. Calls the appropriate `t4a_qtransform_*_materialize` to get a `t4a_treetn`
3. Converts to `TensorTrain` via `_treetn_from_handle`
4. Extracts input/output `Index` metadata from the MPO site indices
5. Returns `LinearOperator(; mpo=tt, input_indices=..., output_indices=...)`
6. Caller is responsible for `set_iospaces!` binding

## Error Handling

All new functions follow AGENTS.md rules:

- Validate arguments in Julia before C API calls
- `ArgumentError` for invalid arguments
- `DimensionMismatch` for shape mismatches
- Include actual vs expected values in messages
- Never discard Rust error messages from `last_error_message()`

Specific validations:
- `svd`/`qr`: `left_inds` must be a subset of `inds(t)`, non-empty, not all
  indices. Duplicate indices rejected. Rank < 2 tensors rejected.
- `orthogonalize`: site must be in `1:length(tt)`, TT must be non-empty
- `truncate`: at least one of rtol/cutoff/maxdim must be specified
- `linkinds`: TT must be non-empty. Exactly one shared index between adjacent
  tensors; throw `ArgumentError` if zero or multiple.
- `siteinds`: TT must be non-empty

## Testing

### test/core/tensor_factorize.jl

- SVD of rank-2 tensor (matrix) — verify `contract(U, contract(S, dag(V)))` reconstructs original
- SVD with truncation — verify rank reduction
- SVD of rank-3+ tensor with various left_inds choices (vector and vararg)
- QR of rank-2 tensor — verify `contract(Q, R)` reconstructs original
- QR of rank-3+ tensor
- `dag` — verify `conj` of data, indices unchanged
- `Array(t, inds...)` — verify permutation matches requested order
- Error paths: rank-0/rank-1 tensors, duplicate left_inds, non-member left_inds,
  left_inds = all indices

### test/tensornetworks/orthogonalize_truncate.jl

- `orthogonalize(tt, site)` — verify `llim`/`rlim` correct after
- `orthogonalize(tt, site; form=:lu)` — verify LU form
- `truncate(tt; cutoff)` — verify bond dimensions reduced
- `truncate(tt; maxdim)` — verify bond dimensions capped
- Non-destructive: original TT unchanged after orthogonalize/truncate
- Round-trip: orthogonalize → truncate → norm approximately preserved
- Error paths: empty TT, site out of range

### test/tensornetworks/queries.jl

- `linkinds(tt)` — verify correct bond indices between sites
- `linkinds(tt, i)` — single bond query
- `linkdims(tt)` — verify dimensions match
- `siteinds(tt)` — verify non-bond indices at each site
- `siteinds(tt, i)` — single site query
- Edge cases: 1-site TT (no links), 2-site TT
- Error paths: malformed chain (zero or multiple shared indices between
  adjacent tensors)

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
- With and without maxbonddim truncation
- Verify against dense DFT matrix

**cumsum_operator:**
- Basic cumulative sum verification

**phase_rotation_operator:**
- Various theta values
- Verify phase factors on known states

**Multivar versions:**
- Multiple variables, each target variable
- Verify only targeted variable is transformed

**Operator binding tests:**
- Materialize operator, call `set_iospaces!` with external indices,
  then `apply` to state with those indices
- Verify that apply fails without `set_iospaces!` (unbound error)

**Negative-path tests:**
- Invalid layout parameters
- Mismatched boundary conditions where not supported

Each test:
1. Materialize operator (small R=2-3)
2. Bind to external state indices via `set_iospaces!`
3. Apply to known state via `TensorNetworks.apply`
4. Compare result against dense reference computation

## Dependencies

- tensor4all-rs PR (svd + qr C API) must merge first
- Then update pin in `deps/build.jl` and CI workflow
- QuanticsTransform materialization uses existing C API (no Rust changes needed)

## Scope Boundaries

**In scope:**
- Tensor: dag, svd, qr, Array(t, inds)
- TensorTrain: dag, orthogonalize, truncate, linkinds, linkdims, siteinds
- QuanticsTransform: materialization of shift, flip, affine, fourier, cumsum,
  phase_rotation operators

**Explicitly out of scope (future phases):**
- BubbleTeaCI migration itself
- Persistent handle (issue #45)
- `binaryop_operator` and `affine_pullback_operator` materialization
- LinearSolvers compatibility helpers (`commoninds(tensor, tensor)`,
  `replaceind!`, `hasinds`, `scalar`, `unity`)
- Mutating variants (`orthogonalize!`, `truncate!`)
- `combiner` (BubbleTeaCI handles via reshape)
- `directsum` on Tensor (TT-level `add` already exists)
- DMRG/sweep algorithms (Rust-side only)
