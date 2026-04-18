# Truncation Policy

Every SVD-based truncating function in `Tensor4all.TensorNetworks` takes the
same three keyword arguments:

```julia
truncate(tt; threshold=0.0, maxdim=0, svd_policy=nothing)
add(a, b; threshold=0.0, maxdim=0, svd_policy=nothing)
contract(a, b; threshold=0.0, maxdim=0, svd_policy=nothing, ...)
apply(op, state; threshold=0.0, maxdim=0, svd_policy=nothing, ...)
linsolve(op, rhs; threshold=0.0, maxdim=0, svd_policy=nothing, ...)
split_to(tt, target; threshold=0.0, maxdim=0, svd_policy=nothing, ...)
Tensor4all.svd(t, left_inds; threshold=0.0, maxdim=0, svd_policy=nothing)
```

`restructure_to` takes the split/final pair: `split_threshold`,
`split_svd_policy`, `final_threshold`, `final_svd_policy`.

## Quick start

The default policy (`scale=:relative, measure=:value, rule=:per_value`)
is the classical "drop singular values below `threshold × σ_max`" rule.

```julia
truncate(tt; threshold=1e-8)                    # σ / σ_max ≤ 1e-8 dropped
truncate(tt; maxdim=32)                          # rank cap only
truncate(tt; threshold=1e-8, maxdim=32)         # both knobs
```

`threshold == 0` disables threshold-based truncation; `maxdim == 0` means
no rank cap. At least one of them must be nonzero for `truncate`.

## The `SvdTruncationPolicy` type

`SvdTruncationPolicy` encodes the **strategy**, independently of the
numeric `threshold`:

```julia
struct SvdTruncationPolicy
    scale::Symbol       # :relative | :absolute
    measure::Symbol     # :value    | :squared_value
    rule::Symbol        # :per_value | :discarded_tail_sum
end
```

The three axes are mutually independent; all 2 × 2 × 2 = 8 combinations
are valid.

## Decision rules (8 combinations)

Let the singular values be `σ_1 ≥ σ_2 ≥ ... ≥ σ_k ≥ 0`, and let

- `x_i = σ_i` when `measure = :value`, `x_i = σ_i²` when
  `measure = :squared_value`.

Then for `threshold = t`:

| scale | rule | Keep while |
|---|---|---|
| `:relative` | `:per_value` | `x_i / max(x) > t` |
| `:absolute` | `:per_value` | `x_i > t` |
| `:relative` | `:discarded_tail_sum` | `Σ x_i (discarded) / Σ x_i (all) ≤ t` |
| `:absolute` | `:discarded_tail_sum` | `Σ x_i (discarded) ≤ t` |

`maxdim` caps the retained rank after the policy decides.

## Setting the default policy

Three configuration patterns, in order of narrowness.

### Process-wide default

Call `set_default_svd_policy!` once, typically from a module's
`__init__` or a program's entry point. A lock makes the write safe; all
subsequent reads on any thread see the new default.

```julia
using Tensor4all.TensorNetworks:
    SvdTruncationPolicy, set_default_svd_policy!

function __init__()
    # ITensors.jl-compatible truncation: discarded-weight semantics.
    set_default_svd_policy!(SvdTruncationPolicy(
        measure = :squared_value,
        rule    = :discarded_tail_sum,
    ))
end

# Now truncate(...; threshold=ε) behaves like ITensors.truncate!(...; cutoff=ε).
truncate(tt; threshold=1e-8)
```

### Task-local scoped override

`with_svd_policy` installs a policy for the current task (and its
children) and restores the previous effective policy on exit. Backed by
`ScopedValues`, so sibling tasks do not interfere.

```julia
using Tensor4all.TensorNetworks: with_svd_policy, SvdTruncationPolicy

with_svd_policy(SvdTruncationPolicy(rule=:discarded_tail_sum)) do
    tt2 = truncate(tt; threshold=1e-10)
    tt3 = add(a, b; threshold=1e-10)
end
# The previous effective policy is back in force here.
```

Parallel tasks stay isolated:

```julia
@sync begin
    Threads.@spawn with_svd_policy(SvdTruncationPolicy(rule=:discarded_tail_sum)) do
        truncate(tt_a; threshold=1e-8)
    end
    Threads.@spawn with_svd_policy(SvdTruncationPolicy(scale=:absolute)) do
        truncate(tt_b; threshold=1e-10)
    end
end
```

### Per-call override

Pass `svd_policy=` for a one-off call; nothing global changes.

```julia
truncate(tt;
    threshold = 1e-12,
    svd_policy = SvdTruncationPolicy(
        measure = :squared_value,
        rule    = :discarded_tail_sum,
    ),
)
```

## Resolution priority

```
svd_policy kwarg     (explicit per-call)
      ↑ overrides
with_svd_policy      (task-local scope)
      ↑ overrides
set_default_svd_policy!   (process-wide)
      ↑ fallback
built-in (:relative, :value, :per_value)
```

`default_svd_policy` returns the policy that would be used right
now if no `svd_policy=` were passed:

```julia
current = default_svd_policy()
@show current.scale current.measure current.rule
```

## Preset recipes

### ITensors.jl-compatible

ITensors' `cutoff` is a discarded-weight threshold
(`Σ σ_i² (discarded) / Σ σ_i² (all) ≤ cutoff`). To reproduce it:

```julia
set_default_svd_policy!(SvdTruncationPolicy(
    measure = :squared_value,
    rule    = :discarded_tail_sum,
))
truncate(tt; threshold=1e-8)   # ≡ ITensors.truncate!(...; cutoff=1e-8)
```

### Absolute Frobenius² error bound

`Σ σ_i² (discarded) ≤ threshold` — caps the absolute Frobenius²
truncation error.

```julia
SvdTruncationPolicy(
    scale   = :absolute,
    measure = :squared_value,
    rule    = :discarded_tail_sum,
)
```

### Default (classical rtol)

`σ_i / σ_max > threshold`:

```julia
SvdTruncationPolicy()   # :relative, :value, :per_value
```

## Thread safety

- `set_default_svd_policy!` is lock-guarded; concurrent writes are serialized.
- `with_svd_policy` uses `ScopedValues` — child tasks spawned inside the
  block inherit the policy automatically; sibling tasks outside the block
  are unaffected.
- Sibling `@spawn` blocks each with their own `with_svd_policy` see their
  own scoped policy without interfering.
