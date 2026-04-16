_placeholder_operator(kind::Symbol; kwargs...) = TensorNetworks.LinearOperator(metadata=(; kind, kwargs...))

function _require_positive_integer(name::AbstractString, value::Integer)
    value > 0 || throw(ArgumentError("$name must be positive, got $value"))
    return value
end

function _require_multivar_target(nvars::Integer, target::Integer)
    _require_positive_integer("nvars", nvars)
    1 <= target <= nvars || throw(
        ArgumentError("target must be between 1 and $nvars, got $target"),
    )
    return target
end

function _require_nonnegative_integer(name::AbstractString, value::Integer)
    value >= 0 || throw(ArgumentError("$name must be nonnegative, got $value"))
    return value
end

function _require_nonnegative_real(name::AbstractString, value::Real)
    value >= 0 || throw(ArgumentError("$name must be nonnegative, got $value"))
    return value
end

function _require_nonzero_integer(name::AbstractString, value)
    value == 0 && throw(ArgumentError("$name must be nonzero, got $value"))
    return value
end

"""
    shift_operator(r, offset; bc=:periodic)

Construct a quantics shift operator materialized as a `TensorNetworks.LinearOperator`.
"""
function shift_operator(r::Integer, offset::Integer; bc=:periodic)
    layout_handle = _new_univariate_layout(r)
    try
        return _materialize_shift(layout_handle, 0, offset, bc, "materializing shift operator")
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    shift_operator_multivar(r, offset, nvars, target; bc=:periodic)

Construct a multivariable quantics shift operator targeting the `target`th variable.
"""
function shift_operator_multivar(
    r::Integer,
    offset::Integer,
    nvars::Integer,
    target::Integer;
    bc=:periodic,
)
    _require_multivar_target(nvars, target)
    layout_handle = _new_multivar_layout(r, nvars)
    try
        return _materialize_shift(
            layout_handle,
            target - 1,
            offset,
            bc,
            "materializing multivar shift operator",
        )
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    flip_operator(r; bc=:periodic)

Construct a quantics flip operator materialized as a `TensorNetworks.LinearOperator`.
"""
function flip_operator(r::Integer; bc=:periodic)
    layout_handle = _new_univariate_layout(r)
    try
        return _materialize_flip(layout_handle, 0, bc, "materializing flip operator")
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    flip_operator_multivar(r, nvars, target; bc=:periodic)

Construct a multivariable quantics flip operator targeting the `target`th variable.
"""
function flip_operator_multivar(r::Integer, nvars::Integer, target::Integer; bc=:periodic)
    _require_multivar_target(nvars, target)
    layout_handle = _new_multivar_layout(r, nvars)
    try
        return _materialize_flip(
            layout_handle,
            target - 1,
            bc,
            "materializing multivar flip operator",
        )
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    phase_rotation_operator(r, theta)

Construct a quantics phase-rotation operator materialized as a `TensorNetworks.LinearOperator`.
"""
function phase_rotation_operator(r::Integer, theta)
    layout_handle = _new_univariate_layout(r)
    try
        return _materialize_phase_rotation(
            layout_handle,
            0,
            theta,
            "materializing phase rotation operator",
        )
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    phase_rotation_operator_multivar(r, theta, nvars, target)

Construct a multivariable phase-rotation operator targeting the `target`th variable.
"""
function phase_rotation_operator_multivar(r::Integer, theta, nvars::Integer, target::Integer)
    _require_multivar_target(nvars, target)
    layout_handle = _new_multivar_layout(r, nvars)
    try
        return _materialize_phase_rotation(
            layout_handle,
            target - 1,
            theta,
            "materializing multivar phase rotation operator",
        )
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    cumsum_operator(r)

Construct a quantics cumulative-sum operator materialized as a `TensorNetworks.LinearOperator`.
"""
function cumsum_operator(r::Integer)
    layout_handle = _new_univariate_layout(r)
    try
        return _materialize_cumsum(layout_handle, 0, "materializing cumsum operator")
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    fourier_operator(r; forward=true, maxbonddim=typemax(Int), tolerance=0.0)

Construct a quantics Fourier operator materialized as a `TensorNetworks.LinearOperator`.
"""
function fourier_operator(
    r::Integer;
    forward::Bool=true,
    maxbonddim::Int=typemax(Int),
    tolerance::Float64=0.0,
)
    layout_handle = _new_univariate_layout(r)
    try
        return _materialize_fourier(
            layout_handle,
            0,
            forward,
            maxbonddim,
            tolerance,
            "materializing Fourier operator",
        )
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    affine_operator(r, a_num, a_den, b_num, b_den; bc=:periodic)

Construct a quantics affine operator materialized as a `TensorNetworks.LinearOperator`.
"""
function affine_operator(r::Integer, a_num, a_den, b_num, b_den; bc=:periodic)
    layout_handle = _new_univariate_layout(r)
    try
        return _materialize_affine(
            layout_handle,
            a_num,
            a_den,
            b_num,
            b_den,
            bc,
            "materializing affine operator",
        )
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    affine_pullback_operator(r, params; bc=:periodic)

Construct a metadata-only quantics affine-pullback operator placeholder.
"""
# Deferred for the Phase 1 materialization pass.
affine_pullback_operator(r::Integer, params; bc=:periodic) =
    _placeholder_operator(:affine_pullback; r, params, bc)

"""
    binaryop_operator(r, a1, b1, a2, b2; bc1=:periodic, bc2=:periodic)

Construct a metadata-only quantics binary-operator placeholder.
"""
# Deferred for the Phase 1 materialization pass.
binaryop_operator(r::Integer, a1, b1, a2, b2; bc1=:periodic, bc2=:periodic) =
    _placeholder_operator(:binaryop; r, a1, b1, a2, b2, bc1, bc2)
