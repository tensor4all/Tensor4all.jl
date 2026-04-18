"""
    SvdTruncationPolicy(; threshold=0.0, scale=:relative,
                         measure=:value, rule=:per_value)

Full SVD truncation policy, mirroring `tensor4all_core::SvdTruncationPolicy`
and `t4a_svd_truncation_policy` in the C API.

# Fields

- `threshold::Float64` — numeric threshold used together with `scale`, `measure`
  and `rule`.
- `scale::Symbol` — one of `:relative` or `:absolute`. `:relative` compares the
  threshold against a singular-value-derived reference scale; `:absolute`
  compares it directly to the measured quantity.
- `measure::Symbol` — one of `:value` or `:squared_value`. Selects whether the
  threshold acts on singular values or on their squares.
- `rule::Symbol` — one of `:per_value` or `:discarded_tail_sum`. `:per_value`
  applies the threshold independently to each singular value; `:discarded_tail_sum`
  applies it to the cumulative discarded tail.

Pass an `SvdTruncationPolicy` via the `svd_policy` keyword to any truncating
function (`truncate`, `add`, `contract`, `apply`, `linsolve`, `split_to`, and
`Core.svd`) to access the full backend strategy beyond the `rtol` / `cutoff`
convenience subset. Passing both `svd_policy` and a nonzero `rtol`/`cutoff`
is rejected as ambiguous.
"""
struct SvdTruncationPolicy
    threshold::Float64
    scale::Symbol
    measure::Symbol
    rule::Symbol

    function SvdTruncationPolicy(threshold::Real, scale::Symbol, measure::Symbol, rule::Symbol)
        threshold >= 0 || throw(ArgumentError(
            "SvdTruncationPolicy: threshold must be nonnegative, got $threshold",
        ))
        scale in (:relative, :absolute) || throw(ArgumentError(
            "SvdTruncationPolicy: unknown scale $(repr(scale)). Expected :relative or :absolute",
        ))
        measure in (:value, :squared_value) || throw(ArgumentError(
            "SvdTruncationPolicy: unknown measure $(repr(measure)). Expected :value or :squared_value",
        ))
        rule in (:per_value, :discarded_tail_sum) || throw(ArgumentError(
            "SvdTruncationPolicy: unknown rule $(repr(rule)). Expected :per_value or :discarded_tail_sum",
        ))
        return new(Float64(threshold), scale, measure, rule)
    end
end

function SvdTruncationPolicy(;
    threshold::Real=0.0,
    scale::Symbol=:relative,
    measure::Symbol=:value,
    rule::Symbol=:per_value,
)
    return SvdTruncationPolicy(threshold, scale, measure, rule)
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

function _to_c_policy(p::SvdTruncationPolicy)
    return _SvdTruncationPolicyC(
        p.threshold,
        _scale_code(p.scale),
        _measure_code(p.measure),
        _rule_code(p.rule),
    )
end

"""
    _resolve_svd_policy(; rtol, cutoff, svd_policy)

Internal helper: resolve the effective FFI SVD truncation policy from the
user-visible kwargs. Returns either `nothing` (caller should pass `C_NULL` to
the backend, meaning no SVD-based truncation) or a `_SvdTruncationPolicyC`
struct ready to feed into `_with_svd_policy_ptr`.

Rules:

- `svd_policy` and nonzero `rtol`/`cutoff` together → `ArgumentError`.
- `svd_policy` set → converted directly.
- Otherwise, when both `rtol == 0` and `cutoff == 0`, returns `nothing`.
- Otherwise, convenience mapping: `cutoff` takes precedence over `rtol`
  (`threshold = sqrt(cutoff)`), scale `:relative`, measure `:value`,
  rule `:per_value`. Matches the historical `rtol` / `cutoff` numerical
  semantics of the Julia wrapper.
"""
function _resolve_svd_policy(;
    rtol::Real=0.0,
    cutoff::Real=0.0,
    svd_policy::Union{Nothing, SvdTruncationPolicy}=nothing,
)
    if svd_policy !== nothing && (rtol != 0 || cutoff != 0)
        throw(ArgumentError(
            "Pass either rtol/cutoff or svd_policy, not both. " *
            "Got rtol=$rtol, cutoff=$cutoff, svd_policy=$(svd_policy).",
        ))
    end
    if svd_policy !== nothing
        return _to_c_policy(svd_policy)
    end
    if rtol == 0 && cutoff == 0
        return nothing
    end
    threshold = cutoff > 0 ? sqrt(Float64(cutoff)) : Float64(rtol)
    return _SvdTruncationPolicyC(
        threshold,
        _T4A_THRESHOLD_SCALE_RELATIVE,
        _T4A_SINGULAR_VALUE_MEASURE_VALUE,
        _T4A_TRUNCATION_RULE_PER_VALUE,
    )
end
