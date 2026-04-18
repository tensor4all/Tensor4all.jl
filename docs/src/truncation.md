# Truncation Contract

Several `Tensor4all.TensorNetworks` operations expose keyword arguments
that control SVD-based truncation of bond dimensions. There are two
entry points:

- **Convenience** — the historical `rtol` / `cutoff` / `maxdim` triple.
- **Full strategy** — a `svd_policy::SvdTruncationPolicy` value that
  exposes the full backend strategy (`threshold`, `scale`, `measure`,
  `rule`) one-to-one with `tensor4all-rs`.

## Convenience kwargs

| Keyword | Meaning |
|---|---|
| `rtol::Real`   | Relative tolerance applied to singular values. A singular value `σ_i` is dropped when `σ_i / σ_max < rtol`. Default `0.0` disables. |
| `cutoff::Real` | Convenience knob expressed in **squared-singular-value space**. Internally converted to `rtol = sqrt(cutoff)` and applied as a relative tolerance. Default `0.0` disables. |
| `maxdim::Integer` | Hard upper bound on the bond dimension after truncation. Default `0` means no rank cap. |

`maxdim` is independent of `rtol` / `cutoff` and can be combined with either.

## Full strategy — `svd_policy`

```julia
using Tensor4all.TensorNetworks: SvdTruncationPolicy

pol = SvdTruncationPolicy(
    threshold = 1e-8,
    scale     = :absolute,          # :relative or :absolute
    measure   = :squared_value,     # :value or :squared_value
    rule      = :discarded_tail_sum, # :per_value or :discarded_tail_sum
)

truncate(tt; svd_policy=pol, maxdim=64)
```

Each field mirrors its counterpart in `tensor4all-rs`. Pass the `svd_policy`
kwarg on any truncating entry point listed below.

Passing both a non-`nothing` `svd_policy` and a nonzero `rtol` / `cutoff`
is rejected as ambiguous with an `ArgumentError`.

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

Every truncating entry point accepts both the convenience triple and the
full `svd_policy` kwarg:

- `TensorNetworks.truncate(tt; rtol, cutoff, maxdim, svd_policy, form)`
- `TensorNetworks.add(a, b; rtol, cutoff, maxdim, svd_policy)`
- `TensorNetworks.contract(a, b; rtol, cutoff, maxdim, svd_policy, qr_rtol, ...)`
- `TensorNetworks.apply(op, state; rtol, cutoff, maxdim, svd_policy, ...)`
- `TensorNetworks.linsolve(op, rhs; rtol, cutoff, maxdim, svd_policy, ...)`
- `TensorNetworks.split_to(...; rtol, cutoff, maxdim, svd_policy, final_sweep)`
  (truncation only takes effect when `final_sweep = true`)
- `TensorNetworks.restructure_to(...; split_rtol, split_cutoff,
  split_maxdim, split_svd_policy, final_svd_policy, ...)` (phase-prefixed
  names separate the split / swap / final passes; both `split_svd_policy`
  and `final_svd_policy` can be set independently)
- `Tensor4all.svd(t, left_inds; rtol, cutoff, maxdim, svd_policy)`

Argument validation is uniform across these calls: negative values raise
`ArgumentError` early in Julia before reaching the C boundary. The
ambiguity rule (`svd_policy` together with nonzero `rtol`/`cutoff`)
applies at every call site.

The SVD-only constraint introduced by `tensor4all-rs #429` means the
`form` kwarg on `truncate`, `linsolve`, and `split_to` rejects
`form=:lu` with an `ArgumentError`; only `form=:unitary` (the default)
reaches the backend.

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

Planned additive enhancements:

- Per-bond / per-edge truncation overrides for tree-shaped networks.
- An explicit accessor that returns the singular values discarded by
  the most recent truncation call, so that downstream packages can
  audit numerical accuracy.

These will land as additive changes; the existing keyword surface will
remain supported.
