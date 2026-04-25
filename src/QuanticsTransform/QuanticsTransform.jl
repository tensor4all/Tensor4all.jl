module QuanticsTransform

import ..TensorNetworks
using ..Tensor4all: Index, Tensor, dim, id, inds

export shift_operator, shift_operator_multivar
export flip_operator, flip_operator_multivar
export phase_rotation_operator, phase_rotation_operator_multivar
export cumsum_operator, fourier_operator
export affine_operator, affine_operator_multivar,
    affine_pullback_operator, affine_pullback_operator_multivar
export binaryop_operator, binaryop_operator_multivar
export unfuse_quantics_operator

include("capi_helpers.jl")
include("operators.jl")
include("unfuse.jl")

end
