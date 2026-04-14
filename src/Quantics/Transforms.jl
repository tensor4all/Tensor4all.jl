"""
    QuanticsTransformDescriptor(kind, parameters)

Metadata-only descriptor for a planned quantics transform.

The descriptor is reviewable during the skeleton phase, while transform
materialization remains backend-backed and therefore stub-only.
"""
struct QuanticsTransformDescriptor
    kind::Symbol
    parameters::NamedTuple
end

"""
    affine_transform(; matrix, shift)

Create an affine transform descriptor.
"""
affine_transform(; matrix, shift) = QuanticsTransformDescriptor(:affine, (; matrix, shift))

"""
    shift_transform(; offsets)

Create a shift transform descriptor.
"""
shift_transform(; offsets) = QuanticsTransformDescriptor(:shift, (; offsets))

"""
    flip_transform(; variables)

Create a flip transform descriptor.
"""
flip_transform(; variables) = QuanticsTransformDescriptor(:flip, (; variables))

"""
    phase_rotation_transform(; phase)

Create a phase-rotation transform descriptor.
"""
phase_rotation_transform(; phase) = QuanticsTransformDescriptor(:phase_rotation, (; phase))

"""
    cumsum_transform(; variable)

Create a cumulative-sum transform descriptor.
"""
cumsum_transform(; variable) = QuanticsTransformDescriptor(:cumsum, (; variable))

"""
    fourier_transform(; variables)

Create a Fourier transform descriptor.
"""
fourier_transform(; variables) = QuanticsTransformDescriptor(:fourier, (; variables))

"""
    binaryop_transform(; op, variables)

Create a binary-operation transform descriptor.
"""
binaryop_transform(; op, variables) = QuanticsTransformDescriptor(:binaryop, (; op, variables))

"""
    materialize_transform(transform)

Placeholder for backend-backed transform materialization.
"""
materialize_transform(::QuanticsTransformDescriptor) =
    throw(SkeletonNotImplemented(:materialize_transform, :quantics))
