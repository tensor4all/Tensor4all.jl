# Truncation Contract

Several `Tensor4all.TensorNetworks` operations expose three keyword
arguments that control SVD-based truncation of bond dimensions:

| Keyword | Meaning |
|---|---|
| `rtol::Real`   | Relative tolerance applied to singular values. A singular value `σ_i` is dropped when `σ_i / σ_max < rtol`. Default `0.0` disables. |
| `cutoff::Real` | Convenience knob expressed in **squared-singular-value space**. Internally converted to `rtol = sqrt(cutoff)` and applied as a relative tolerance. Default `0.0` disables. |
| `maxdim::Integer` | Hard upper bound on the bond dimension after truncation. Default `0` means no rank cap. |

`maxdim` is independent of `rtol` / `cutoff` and can be combined with either.

## Precedence between `rtol` and `cutoff`

Only one of `rtol` and `cutoff` is fed to the backend per call:

1. If `cutoff > 0`, the backend uses `sqrt(cutoff)` as the relative
   tolerance (and `rtol` is ignored even if it is also set).
2. Otherwise, if `rtol > 0`, the backend uses that value verbatim.
3. Otherwise, no rtol-based truncation is applied.

This precedence is fixed by the C API resolver `select_tol` /
`cutoff_to_rtol` in `tensor4all-rs/crates/tensor4all-capi/src/treetn.rs`
(see [tensor4all/Tensor4all.jl#48](https://github.com/tensor4all/Tensor4all.jl/issues/48)
for the discussion that pinned this contract).

**Recommendation**: pick one of the two and stick to it within a given
calculation. Setting both is silently equivalent to setting only
`cutoff`.

## Why `cutoff` is squared

`cutoff` is expressed in the same space as the **sum of squared
discarded singular values**, i.e. the squared Frobenius error norm of
the truncation. Concretely, dropping a singular value `σ_i` contributes
`σ_i²` to the truncation error norm. Setting `cutoff = ε²` is therefore
roughly the same as bounding the discarded relative norm by `ε`.

Internally the backend SVD truncation works in singular-value space, so
the wrapper converts: `cutoff` (norm² space) → `sqrt(cutoff)`
(singular-value space) → `rtol`.

This matches the convention used by ITensors / ITensorMPS where
`cutoff` is the discarded weight and the rank-revealing SVD selects
singular values whose squared sum stays above
`(1 - cutoff) * ||σ||²`.

## Functions that accept the contract

The same `(rtol, cutoff, maxdim)` keyword set appears on every
truncating entry point in the public surface:

- `TensorNetworks.truncate(tt; rtol, cutoff, maxdim, form)`
- `TensorNetworks.add(a, b; rtol, cutoff, maxdim)`
- `TensorNetworks.contract(a, b; rtol, cutoff, maxdim, ...)`
- `TensorNetworks.apply(op, state; rtol, cutoff, maxdim, ...)`
- `TensorNetworks.linsolve(op, rhs; rtol, cutoff, maxdim, ...)`
- `TensorNetworks.split_to(...; rtol, cutoff, maxdim, final_sweep)`
  (truncation only takes effect when `final_sweep = true`)
- `TensorNetworks.restructure_to(...; split_rtol, split_cutoff,
  split_maxdim, ...)` (forwards the same contract; phase-prefixed names
  separate the split / swap / final passes)
- `Tensor4all.svd(t, left_inds; rtol, cutoff, maxdim)`

The argument validation is uniform across these calls: negative values
raise `ArgumentError` early in Julia before reaching the C boundary.

## Sentinel meanings

| Value | Meaning |
|---|---|
| `rtol = 0.0` | Disable rtol-based truncation. |
| `cutoff = 0.0` | Disable cutoff-based truncation. |
| `maxdim = 0` | No upper bound on bond dimension. |

The combination `(rtol = 0, cutoff = 0, maxdim = 0)` is rejected by
`truncate` (with a clear `ArgumentError`) because it would request a
truncation operation that does nothing. Other entry points (such as
`add` or `contract`) accept the all-zero combination as "no
truncation".

## Future work

The current contract is intentionally minimal: every truncating call
takes the same three knobs and dispatches them through the same
`select_tol` / `cutoff_to_rtol` resolver. Planned future enhancements:

- A user-facing `TruncationScheme` type so callers can express richer
  policies (e.g. "absolute cutoff in singular-value space",
  "discard-weight-only", or "rtol with a separate per-bond override")
  in a single value rather than a triple of keyword arguments.
- Per-bond / per-edge truncation overrides for tree-shaped networks.
- An explicit accessor that returns the singular values discarded by
  the most recent truncation call, so that downstream packages can
  audit numerical accuracy.

These will land as additive changes; the existing keyword surface will
remain supported.
