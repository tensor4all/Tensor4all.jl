"""
    SvdTruncationPolicy(; scale=:relative, measure=:value, rule=:per_value)

Strategy for SVD-based bond truncation. The numeric amount is a per-call
`threshold` kwarg on every truncating function; this type only describes
how that threshold is interpreted.

# Fields

- `scale::Symbol` — `:relative` or `:absolute`. `:relative` compares the
  threshold against a singular-value-derived reference scale; `:absolute`
  compares it directly to the measured quantity.
- `measure::Symbol` — `:value` or `:squared_value`. Selects whether the
  threshold acts on singular values or on their squares.
- `rule::Symbol` — `:per_value` or `:discarded_tail_sum`. `:per_value`
  applies the threshold independently to each singular value;
  `:discarded_tail_sum` applies it to the cumulative discarded tail.

See the Truncation Policy chapter of the docs for the eight decision
rules, preset recipes, and configuration patterns (process-wide default
via [`set_default_svd_policy!`](@ref), task-local override via
[`with_svd_policy`](@ref), or per-call `svd_policy=` kwarg).
"""
struct SvdTruncationPolicy
    scale::Symbol
    measure::Symbol
    rule::Symbol

    function SvdTruncationPolicy(scale::Symbol, measure::Symbol, rule::Symbol)
        scale in (:relative, :absolute) || throw(ArgumentError(
            "SvdTruncationPolicy: unknown scale $(repr(scale)). Expected :relative or :absolute",
        ))
        measure in (:value, :squared_value) || throw(ArgumentError(
            "SvdTruncationPolicy: unknown measure $(repr(measure)). Expected :value or :squared_value",
        ))
        rule in (:per_value, :discarded_tail_sum) || throw(ArgumentError(
            "SvdTruncationPolicy: unknown rule $(repr(rule)). Expected :per_value or :discarded_tail_sum",
        ))
        return new(scale, measure, rule)
    end
end

function SvdTruncationPolicy(;
    scale::Symbol=:relative,
    measure::Symbol=:value,
    rule::Symbol=:per_value,
)
    return SvdTruncationPolicy(scale, measure, rule)
end

function _scale_code(scale::Symbol)
    scale === :relative && return _T4A_THRESHOLD_SCALE_RELATIVE
    scale === :absolute && return _T4A_THRESHOLD_SCALE_ABSOLUTE
    throw(ArgumentError("unknown scale $(repr(scale))"))
end

function _measure_code(measure::Symbol)
    measure === :value && return _T4A_SINGULAR_VALUE_MEASURE_VALUE
    measure === :squared_value && return _T4A_SINGULAR_VALUE_MEASURE_SQUARED_VALUE
    throw(ArgumentError("unknown measure $(repr(measure))"))
end

function _rule_code(rule::Symbol)
    rule === :per_value && return _T4A_TRUNCATION_RULE_PER_VALUE
    rule === :discarded_tail_sum && return _T4A_TRUNCATION_RULE_DISCARDED_TAIL_SUM
    throw(ArgumentError("unknown rule $(repr(rule))"))
end

function _to_c_policy(p::SvdTruncationPolicy, threshold::Real)
    return _SvdTruncationPolicyC(
        Float64(threshold),
        _scale_code(p.scale),
        _measure_code(p.measure),
        _rule_code(p.rule),
    )
end

# --- Default policy registry ---------------------------------------------

const _PROCESS_DEFAULT_LOCK = ReentrantLock()
const _PROCESS_DEFAULT = Ref(SvdTruncationPolicy())
const _SCOPED_OVERRIDE =
    ScopedValues.ScopedValue{Union{Nothing, SvdTruncationPolicy}}(nothing)

"""
    default_svd_policy() -> SvdTruncationPolicy

Return the currently active default `SvdTruncationPolicy`. Resolution order:

1. Task-local scoped override set by [`with_svd_policy`](@ref).
2. Process-wide default set by [`set_default_svd_policy!`](@ref).
3. Built-in default (`:relative, :value, :per_value`).

Callers of truncating functions that do not pass `svd_policy=` explicitly
receive this policy as their effective policy.
"""
function default_svd_policy()
    s = _SCOPED_OVERRIDE[]
    s !== nothing && return s
    return lock(_PROCESS_DEFAULT_LOCK) do
        _PROCESS_DEFAULT[]
    end
end

"""
    set_default_svd_policy!(p::SvdTruncationPolicy) -> Nothing

Set the process-wide default truncation policy. Intended to be called at
most once, typically from a module `__init__` or program entry point.
Lock-guarded to make concurrent writes safe. A scoped override entered
via [`with_svd_policy`](@ref) still wins over this default while active.
"""
function set_default_svd_policy!(p::SvdTruncationPolicy)
    lock(_PROCESS_DEFAULT_LOCK) do
        _PROCESS_DEFAULT[] = p
    end
    return nothing
end

"""
    with_svd_policy(f, p::SvdTruncationPolicy)

Run `f()` with `p` installed as the effective default policy in the current
task scope. The previous effective policy is restored on exit. Backed by
`ScopedValues`, so child tasks spawned inside the block inherit `p` while
sibling tasks outside the block are unaffected.

```julia
with_svd_policy(SvdTruncationPolicy(rule=:discarded_tail_sum)) do
    tt2 = truncate(tt; threshold=1e-10)
    tt3 = add(a, b; threshold=1e-10)
end
```
"""
function with_svd_policy(f, p::SvdTruncationPolicy)
    return ScopedValues.with(f, _SCOPED_OVERRIDE => p)
end

# --- Call-site resolver --------------------------------------------------

"""
    _resolve_svd_policy(; threshold, svd_policy)

Internal helper: resolve the effective FFI SVD truncation policy from the
per-call kwargs.

- Returns `nothing` when `threshold == 0` (caller passes `C_NULL` to the
  backend, meaning no SVD-based truncation).
- Otherwise returns a `_SvdTruncationPolicyC` built from `threshold` plus
  either the explicit `svd_policy` (if not `nothing`) or
  [`default_svd_policy`](@ref).
- Rejects negative `threshold` with `ArgumentError`.
"""
function _resolve_svd_policy(;
    threshold::Real,
    svd_policy::Union{Nothing, SvdTruncationPolicy},
)
    threshold >= 0 || throw(ArgumentError(
        "threshold must be nonnegative, got $threshold",
    ))
    threshold == 0 && return nothing
    policy = svd_policy === nothing ? default_svd_policy() : svd_policy
    return _to_c_policy(policy, Float64(threshold))
end
