"""
    QuanticsTransform

Quantics-specific constructors that return `TensorNetworks.LinearOperator`
placeholders.
"""
module QuanticsTransform

import ..TensorNetworks

export shift_operator, shift_operator_multivar
export flip_operator, flip_operator_multivar
export phase_rotation_operator, phase_rotation_operator_multivar
export cumsum_operator, fourier_operator
export affine_operator, affine_pullback_operator, binaryop_operator

_placeholder_operator(kind::Symbol; kwargs...) = TensorNetworks.LinearOperator(metadata=(; kind, kwargs...))

"""
    shift_operator(r, offset; bc=:periodic)

Construct a placeholder quantics shift operator.
"""
shift_operator(r::Integer, offset::Integer; bc=:periodic) =
    _placeholder_operator(:shift; r, offset, bc)

"""
    shift_operator_multivar(r, offset, nvars, target; bc=:periodic)

Construct a placeholder multivariate quantics shift operator.
"""
shift_operator_multivar(r::Integer, offset::Integer, nvars::Integer, target::Integer; bc=:periodic) =
    _placeholder_operator(:shift_multivar; r, offset, nvars, target, bc)

"""
    flip_operator(r; bc=:periodic)

Construct a placeholder quantics flip operator.
"""
flip_operator(r::Integer; bc=:periodic) =
    _placeholder_operator(:flip; r, bc)

"""
    flip_operator_multivar(r, nvars, target; bc=:periodic)

Construct a placeholder multivariate quantics flip operator.
"""
flip_operator_multivar(r::Integer, nvars::Integer, target::Integer; bc=:periodic) =
    _placeholder_operator(:flip_multivar; r, nvars, target, bc)

"""
    phase_rotation_operator(r, theta)

Construct a placeholder quantics phase-rotation operator.
"""
phase_rotation_operator(r::Integer, theta) =
    _placeholder_operator(:phase_rotation; r, theta)

"""
    phase_rotation_operator_multivar(r, theta, nvars, target)

Construct a placeholder multivariate quantics phase-rotation operator.
"""
phase_rotation_operator_multivar(r::Integer, theta, nvars::Integer, target::Integer) =
    _placeholder_operator(:phase_rotation_multivar; r, theta, nvars, target)

"""
    cumsum_operator(r)

Construct a placeholder quantics cumulative-sum operator.
"""
cumsum_operator(r::Integer) = _placeholder_operator(:cumsum; r)

"""
    fourier_operator(r; forward=true, maxbonddim=typemax(Int), tolerance=0.0)

Construct a placeholder quantics Fourier operator.
"""
fourier_operator(r::Integer; forward::Bool=true, maxbonddim::Int=typemax(Int), tolerance::Float64=0.0) =
    _placeholder_operator(:fourier; r, forward, maxbonddim, tolerance)

"""
    affine_operator(r, a_num, a_den, b_num, b_den; bc=:periodic)

Construct a placeholder quantics affine operator.
"""
affine_operator(r::Integer, a_num, a_den, b_num, b_den; bc=:periodic) =
    _placeholder_operator(:affine; r, a_num, a_den, b_num, b_den, bc)

"""
    affine_pullback_operator(r, params; bc=:periodic)

Construct a placeholder quantics affine pullback operator.
"""
affine_pullback_operator(r::Integer, params; bc=:periodic) =
    _placeholder_operator(:affine_pullback; r, params, bc)

"""
    binaryop_operator(r, a1, b1, a2, b2; bc1=:periodic, bc2=:periodic)

Construct a placeholder binary quantics operator.
"""
binaryop_operator(r::Integer, a1, b1, a2, b2; bc1=:periodic, bc2=:periodic) =
    _placeholder_operator(:binaryop; r, a1, b1, a2, b2, bc1, bc2)

end
