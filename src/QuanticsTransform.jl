"""
    QuanticsTransform

Quantics transformation operators for tree tensor networks (MPS).

Provides operators for shift, flip, phase rotation, cumulative sum, and
Fourier transform in quantics representation, wrapping the Rust
`tensor4all-quanticstransform` crate via C API.

# Usage
```julia
using Tensor4all.QuanticsTransform

op = shift_operator(4, 1)
result = apply(op, mps)
```
"""
module QuanticsTransform

using ..C_API
using ..TreeTN: TreeTensorNetwork

export LinearOperator
export AffineParams
export shift_operator, flip_operator, phase_rotation_operator, cumsum_operator, fourier_operator
export shift_operator_multivar, flip_operator_multivar, phase_rotation_operator_multivar
export affine_operator, affine_pullback_operator, binaryop_operator
export apply
export set_input_space!, set_output_space!, set_iospaces!
export BoundaryCondition, Periodic, Open

# ============================================================================
# Boundary condition enum
# ============================================================================

"""
    BoundaryCondition

Boundary condition for quantics operators.

- `Periodic` (0): Periodic boundary conditions
- `Open` (1): Open boundary conditions
"""
@enum BoundaryCondition begin
    Periodic = 0
    Open = 1
end

_bc_cint(bc::BoundaryCondition) = Cint(Int(bc))

function _bc_array_cint(bc::AbstractVector{<:BoundaryCondition})
    return Cint[Int(b) for b in bc]
end

# ============================================================================
# LinearOperator type
# ============================================================================

"""
    LinearOperator

A linear operator that can be applied to a tree tensor network (MPS).

Wraps a Rust `LinearOperator` from the quantics transform crate.
Created via operator construction functions like `shift_operator`, `flip_operator`, etc.
"""
mutable struct LinearOperator
    ptr::Ptr{Cvoid}

    function LinearOperator(ptr::Ptr{Cvoid})
        ptr == C_NULL && error("Failed to create LinearOperator: null pointer")
        op = new(ptr)
        finalizer(op) do obj
            C_API.t4a_linop_release(obj.ptr)
        end
        return op
    end
end

_as_rational64(value::Rational) = Rational{Int64}(Int64(numerator(value)), Int64(denominator(value)))
_as_rational64(value::Integer) = Rational{Int64}(Int64(value), 1)

"""
    AffineParams(a, b)

Affine pullback parameters representing `f(y) = g(A*y + b)`.

- `a`: source-dimension by output-dimension affine matrix
- `b`: source-dimension shift vector
"""
struct AffineParams
    a::Matrix{Rational{Int64}}
    b::Vector{Rational{Int64}}

    function AffineParams(a::AbstractMatrix, b::AbstractVector)
        size(a, 1) == length(b) || error("Affine matrix row count must match shift length")
        a_rat = _as_rational64.(a)
        b_rat = _as_rational64.(b)
        return new(reshape(a_rat, size(a)), collect(b_rat))
    end
end

source_ndims(params::AffineParams) = size(params.a, 1)
output_ndims(params::AffineParams) = size(params.a, 2)

# ============================================================================
# Operator construction functions
# ============================================================================

"""
    shift_operator(r::Integer, offset::Integer; bc=Periodic) -> LinearOperator

Create a shift operator: f(x) = g(x + offset) mod 2^r.

# Arguments
- `r`: Number of quantics bits
- `offset`: Shift offset (can be negative)
- `bc`: Boundary condition (`Periodic` or `Open`)
"""
function shift_operator(r::Integer, offset::Integer; bc::BoundaryCondition=Periodic)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qtransform_shift(Csize_t(r), Int64(offset), _bc_cint(bc), out)
    C_API.check_status(status)
    return LinearOperator(out[])
end

"""
    shift_operator_multivar(r::Integer, offset::Integer, nvariables::Integer, target_var::Integer; bc=Periodic) -> LinearOperator

Create a shift operator acting on one variable in a multi-variable quantics system.
"""
function shift_operator_multivar(r::Integer, offset::Integer, nvariables::Integer,
                                 target_var::Integer; bc::BoundaryCondition=Periodic)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qtransform_shift_multivar(
        Csize_t(r), Int64(offset), _bc_cint(bc), Csize_t(nvariables), Csize_t(target_var), out)
    C_API.check_status(status)
    return LinearOperator(out[])
end

"""
    flip_operator(r::Integer; bc=Periodic) -> LinearOperator

Create a flip operator: f(x) = g(2^r - x).

# Arguments
- `r`: Number of quantics bits
- `bc`: Boundary condition (`Periodic` or `Open`)
"""
function flip_operator(r::Integer; bc::BoundaryCondition=Periodic)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qtransform_flip(Csize_t(r), _bc_cint(bc), out)
    C_API.check_status(status)
    return LinearOperator(out[])
end

"""
    flip_operator_multivar(r::Integer, nvariables::Integer, target_var::Integer; bc=Periodic) -> LinearOperator

Create a flip operator acting on one variable in a multi-variable quantics system.
"""
function flip_operator_multivar(r::Integer, nvariables::Integer, target_var::Integer;
                                bc::BoundaryCondition=Periodic)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qtransform_flip_multivar(
        Csize_t(r), _bc_cint(bc), Csize_t(nvariables), Csize_t(target_var), out)
    C_API.check_status(status)
    return LinearOperator(out[])
end

"""
    phase_rotation_operator(r::Integer, theta::Real) -> LinearOperator

Create a phase rotation operator: f(x) = exp(i*theta*x) * g(x).

# Arguments
- `r`: Number of quantics bits
- `theta`: Phase angle
"""
function phase_rotation_operator(r::Integer, theta::Real)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qtransform_phase_rotation(Csize_t(r), Cdouble(theta), out)
    C_API.check_status(status)
    return LinearOperator(out[])
end

"""
    phase_rotation_operator_multivar(r::Integer, theta::Real, nvariables::Integer, target_var::Integer) -> LinearOperator

Create a phase rotation operator acting on one variable in a multi-variable quantics system.
"""
function phase_rotation_operator_multivar(r::Integer, theta::Real, nvariables::Integer,
                                          target_var::Integer)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qtransform_phase_rotation_multivar(
        Csize_t(r), Cdouble(theta), Csize_t(nvariables), Csize_t(target_var), out)
    C_API.check_status(status)
    return LinearOperator(out[])
end

"""
    affine_operator(r::Integer, a_num::AbstractMatrix{<:Integer}, a_den::AbstractMatrix{<:Integer},
                    b_num::AbstractVector{<:Integer}, b_den::AbstractVector{<:Integer}; bc) -> LinearOperator

Create an affine transform with rational coefficients on quantized coordinates.
`bc` must contain one boundary condition per output variable.
"""
function affine_operator(r::Integer,
                         a_num::AbstractMatrix{<:Integer},
                         a_den::AbstractMatrix{<:Integer},
                         b_num::AbstractVector{<:Integer},
                         b_den::AbstractVector{<:Integer};
                         bc::AbstractVector{<:BoundaryCondition})
    size(a_num) == size(a_den) || error("a_num and a_den must have the same size")
    m, n = size(a_num)
    length(b_num) == m || error("b_num length must match the number of output variables")
    length(b_den) == m || error("b_den length must match the number of output variables")
    length(bc) == m || error("bc length must match the number of output variables")

    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qtransform_affine(
        Csize_t(r),
        Int64.(vec(a_num)),
        Int64.(vec(a_den)),
        Int64.(collect(b_num)),
        Int64.(collect(b_den)),
        Csize_t(m),
        Csize_t(n),
        _bc_array_cint(bc),
        out,
    )
    C_API.check_status(status)
    return LinearOperator(out[])
end

"""
    affine_pullback_operator(r::Integer, params::AffineParams;
                             bc=fill(Periodic, source_ndims(params))) -> LinearOperator

Create an affine pullback operator implementing `f(y) = g(A*y + b)`.

The input state has `source_ndims(params)` variables and the output state has
`output_ndims(params)` variables. Boundary conditions apply to the transformed
source coordinates `A*y + b`.
"""
function affine_pullback_operator(
    r::Integer,
    params::AffineParams;
    bc::AbstractVector{<:BoundaryCondition}=fill(Periodic, source_ndims(params)),
)
    length(bc) == source_ndims(params) ||
        error("Boundary condition length must match source dimension")

    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qtransform_affine_pullback(
        Csize_t(r),
        Csize_t(source_ndims(params)),
        Csize_t(output_ndims(params)),
        Int64[numerator(value) for value in vec(params.a)],
        Int64[denominator(value) for value in vec(params.a)],
        Int64[numerator(value) for value in params.b],
        Int64[denominator(value) for value in params.b],
        _bc_array_cint(bc),
        out,
    )
    C_API.check_status(status)
    return LinearOperator(out[])
end

"""
    binaryop_operator(r::Integer, a1::Integer, b1::Integer, a2::Integer, b2::Integer;
                      bc1=Periodic, bc2=Periodic) -> LinearOperator

Create a two-output binary operator corresponding to
`(a1*x + b1*y, a2*x + b2*y)`.
"""
function binaryop_operator(r::Integer, a1::Integer, b1::Integer, a2::Integer, b2::Integer;
                           bc1::BoundaryCondition=Periodic, bc2::BoundaryCondition=Periodic)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qtransform_binaryop(
        Csize_t(r), Int8(a1), Int8(b1), Int8(a2), Int8(b2), _bc_cint(bc1), _bc_cint(bc2), out)
    C_API.check_status(status)
    return LinearOperator(out[])
end

"""
    cumsum_operator(r::Integer) -> LinearOperator

Create a cumulative sum operator: y_i = sum_{j<i} x_j.

# Arguments
- `r`: Number of quantics bits
"""
function cumsum_operator(r::Integer)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qtransform_cumsum(Csize_t(r), out)
    C_API.check_status(status)
    return LinearOperator(out[])
end

"""
    fourier_operator(r::Integer; forward=true, maxbonddim=0, tolerance=0.0) -> LinearOperator

Create a Quantics Fourier Transform operator.

# Arguments
- `r`: Number of quantics bits
- `forward`: `true` for forward FT, `false` for inverse
- `maxbonddim`: Maximum bond dimension (0 = default of 12)
- `tolerance`: Tolerance for FT construction (0.0 = default of 1e-14)
"""
function fourier_operator(r::Integer; forward::Bool=true, maxbonddim::Integer=0, tolerance::Real=0.0)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qtransform_fourier(
        Csize_t(r), Cint(forward ? 1 : 0),
        Csize_t(maxbonddim), Cdouble(tolerance), out
    )
    C_API.check_status(status)
    return LinearOperator(out[])
end

# ============================================================================
# Operator application
# ============================================================================

"""
    set_input_space!(op::LinearOperator, state::TreeTensorNetwork) -> LinearOperator

Reset the operator's true input site indices to match `state`.
"""
function set_input_space!(op::LinearOperator, state::TreeTensorNetwork)
    status = C_API.t4a_linop_set_input_space(op.ptr, state.handle)
    C_API.check_status(status)
    return op
end

"""
    set_output_space!(op::LinearOperator, state::TreeTensorNetwork) -> LinearOperator

Reset the operator's true output site indices to match `state`.
"""
function set_output_space!(op::LinearOperator, state::TreeTensorNetwork)
    status = C_API.t4a_linop_set_output_space(op.ptr, state.handle)
    C_API.check_status(status)
    return op
end

"""
    set_iospaces!(op::LinearOperator, input_state::TreeTensorNetwork, output_state::TreeTensorNetwork=input_state) -> LinearOperator

Reset the operator's true input and output site indices to match the given states.
"""
function set_iospaces!(op::LinearOperator, input_state::TreeTensorNetwork,
                       output_state::TreeTensorNetwork=input_state)
    set_input_space!(op, input_state)
    set_output_space!(op, output_state)
    return op
end

"""
    apply(op::LinearOperator, state::TreeTensorNetwork; method=:naive, rtol=0.0, maxdim=0) -> TreeTensorNetwork

Apply a linear operator to a tree tensor network (MPS).

# Arguments
- `op`: Linear operator
- `state`: Input MPS/TreeTN
- `method`: Contraction method - `:naive`, `:zipup`, or `:fit`
- `rtol`: Relative tolerance (0.0 = default)
- `maxdim`: Maximum bond dimension (0 = unlimited)
"""
function apply(op::LinearOperator, state::TreeTensorNetwork;
               method::Symbol=:naive, rtol::Real=0.0, maxdim::Integer=0)
    method_int = if method == :naive
        Cint(0)
    elseif method == :zipup
        Cint(1)
    elseif method == :fit
        Cint(2)
    else
        error("Unknown method: $method. Use :naive, :zipup, or :fit")
    end

    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_linop_apply(
        op.ptr, state.handle, method_int,
        Cdouble(rtol), Csize_t(maxdim), out
    )
    C_API.check_status(status)

    # Get the number of vertices from the result
    n_out = Ref{Csize_t}(0)
    status = C_API.t4a_treetn_num_vertices(out[], n_out)
    C_API.check_status(status)
    n = Int(n_out[])

    # Result TreeTN has usize node names (0, 1, ..., n-1)
    node_names = collect(0:n-1)
    node_map = Dict{Int, Int}(name => name for name in node_names)
    return TreeTensorNetwork{Int}(out[], node_map, node_names)
end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, op::LinearOperator)
    print(io, "LinearOperator()")
end

# ============================================================================
# Copy / Clone
# ============================================================================

function Base.copy(op::LinearOperator)
    ptr = C_API.t4a_linop_clone(op.ptr)
    return LinearOperator(ptr)
end

function Base.deepcopy(op::LinearOperator)
    ptr = C_API.t4a_linop_clone(op.ptr)
    return LinearOperator(ptr)
end

end # module QuanticsTransform
