"""
    QuanticsTransform(kind, parameters)

Metadata-only descriptor for a planned quantics transform.

The descriptor is reviewable during the skeleton phase, while transform
materialization remains backend-backed and therefore stub-only.
"""
struct QuanticsTransform
    kind::Symbol
    parameters::NamedTuple
end

"""
    affine_transform(; matrix, shift)

Create an affine transform descriptor.
"""
affine_transform(; matrix, shift) = QuanticsTransform(:affine, (; matrix, shift))

"""
    shift_transform(; offsets)

Create a shift transform descriptor.
"""
shift_transform(; offsets) = QuanticsTransform(:shift, (; offsets))

"""
    flip_transform(; variables)

Create a flip transform descriptor.
"""
flip_transform(; variables) = QuanticsTransform(:flip, (; variables))

"""
    phase_rotation_transform(; phase)

Create a phase-rotation transform descriptor.
"""
phase_rotation_transform(; phase) = QuanticsTransform(:phase_rotation, (; phase))

"""
    cumsum_transform(; variable)

Create a cumulative-sum transform descriptor.
"""
cumsum_transform(; variable) = QuanticsTransform(:cumsum, (; variable))

"""
    fourier_transform(; variables)

Create a Fourier transform descriptor.
"""
fourier_transform(; variables) = QuanticsTransform(:fourier, (; variables))

"""
    binaryop_transform(; op, variables)

Create a binary-operation transform descriptor.
"""
binaryop_transform(; op, variables) = QuanticsTransform(:binaryop, (; op, variables))

"""
    materialize_transform(transform)

Placeholder for backend-backed transform materialization.
"""
materialize_transform(::QuanticsTransform) =
    throw(SkeletonNotImplemented(:materialize_transform, :quantics))
