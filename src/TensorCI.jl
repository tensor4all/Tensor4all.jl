"""
    TensorCI

High-level Julia wrappers for TensorCI2 from tensor4all-tensorci.

This provides tensor cross interpolation algorithms for approximating
high-dimensional functions as tensor trains.
"""
module TensorCI

using ..C_API
using ..SimpleTT: SimpleTensorTrain

export TensorCI2, crossinterpolate2, crossinterpolate2_tci

const _TensorCIScalar = Union{Float64, ComplexF64}

_suffix(::Type{Float64}) = "f64"
_suffix(::Type{ComplexF64}) = "c64"
_api(::Type{T}, name::Symbol) where {T<:_TensorCIScalar} =
    getfield(C_API, Symbol("t4a_tci2_", _suffix(T), "_", name))
_cross_api(::Type{T}) where {T<:_TensorCIScalar} =
    getfield(C_API, Symbol("t4a_crossinterpolate2_", _suffix(T)))

function _infer_scalar_type(f, local_dims::Vector{<:Integer}, initial_pivots::Vector{Vector{Int}})
    sample_indices = isempty(initial_pivots) ? zeros(Int, length(local_dims)) : initial_pivots[1]
    sample_value = f(sample_indices...)
    if sample_value isa Real
        return Float64
    elseif sample_value isa Complex
        return ComplexF64
    end
    error("TensorCI callback must return a real or complex scalar, got $(typeof(sample_value))")
end

# ============================================================================
# Callback infrastructure for passing Julia functions to Rust
# ============================================================================

"""
    _trampoline(indices_ptr::Ptr{Int64}, n_indices::Csize_t,
                result_ptr::Ptr{Float64}, user_data::Ptr{Cvoid})::Cint

Internal trampoline function for f64 callbacks. Converts C-style call to a Julia closure.
"""
function _trampoline(
    indices_ptr::Ptr{Int64},
    n_indices::Csize_t,
    result_ptr::Ptr{Float64},
    user_data::Ptr{Cvoid}
)::Cint
    try
        f_ref = unsafe_pointer_to_objref(user_data)::Ref{Any}
        f = f_ref[]
        indices = unsafe_wrap(Array, indices_ptr, Int(n_indices))
        val = Float64(f(indices...))
        unsafe_store!(result_ptr, val)
        return Cint(0)
    catch err
        @error "Error in TCI callback" exception = (err, catch_backtrace())
        return Cint(-1)
    end
end

"""
    _trampoline_c64(indices_ptr::Ptr{Int64}, n_indices::Csize_t,
                    result_re::Ptr{Float64}, result_im::Ptr{Float64},
                    user_data::Ptr{Cvoid})::Cint

Internal trampoline function for c64 callbacks. Converts C-style call to a Julia closure.
"""
function _trampoline_c64(
    indices_ptr::Ptr{Int64},
    n_indices::Csize_t,
    result_re::Ptr{Float64},
    result_im::Ptr{Float64},
    user_data::Ptr{Cvoid}
)::Cint
    try
        f_ref = unsafe_pointer_to_objref(user_data)::Ref{Any}
        f = f_ref[]
        indices = unsafe_wrap(Array, indices_ptr, Int(n_indices))
        val = ComplexF64(f(indices...))
        unsafe_store!(result_re, real(val))
        unsafe_store!(result_im, imag(val))
        return Cint(0)
    catch err
        @error "Error in complex TCI callback" exception = (err, catch_backtrace())
        return Cint(-1)
    end
end

const _TRAMPOLINE_PTR_REF = Ref{Ptr{Cvoid}}(C_NULL)
const _TRAMPOLINE_C64_PTR_REF = Ref{Ptr{Cvoid}}(C_NULL)

function _get_trampoline_ptr(::Type{Float64})
    if _TRAMPOLINE_PTR_REF[] == C_NULL
        _TRAMPOLINE_PTR_REF[] = @cfunction(
            _trampoline,
            Cint,
            (Ptr{Int64}, Csize_t, Ptr{Float64}, Ptr{Cvoid})
        )
    end
    return _TRAMPOLINE_PTR_REF[]
end

function _get_trampoline_ptr(::Type{ComplexF64})
    if _TRAMPOLINE_C64_PTR_REF[] == C_NULL
        _TRAMPOLINE_C64_PTR_REF[] = @cfunction(
            _trampoline_c64,
            Cint,
            (Ptr{Int64}, Csize_t, Ptr{Float64}, Ptr{Float64}, Ptr{Cvoid})
        )
    end
    return _TRAMPOLINE_C64_PTR_REF[]
end

# ============================================================================
# TensorCI2 type
# ============================================================================

"""
    TensorCI2{T<:Union{Float64, ComplexF64}}

A TCI (Tensor Cross Interpolation) object for the 2-site algorithm.

This wraps the Rust `TensorCI2{T}` and provides access to the interpolated
tensor train representation of a function.
"""
mutable struct TensorCI2{T<:_TensorCIScalar}
    ptr::Ptr{Cvoid}
    local_dims::Vector{Int}

    function TensorCI2{T}(ptr::Ptr{Cvoid}, local_dims::Vector{Int}) where {T<:_TensorCIScalar}
        ptr == C_NULL && error("Failed to create TensorCI2: null pointer")
        tci = new{T}(ptr, copy(local_dims))
        finalizer(tci) do obj
            _api(T, :release)(obj.ptr)
        end
        return tci
    end
end

"""
    TensorCI2(local_dims::Vector{<:Integer})

Create a new empty Float64 TensorCI2 object with the given local dimensions.
"""
TensorCI2(local_dims::Vector{<:Integer}) = TensorCI2{Float64}(local_dims)

function TensorCI2{T}(local_dims::Vector{<:Integer}) where {T<:_TensorCIScalar}
    dims = Csize_t.(local_dims)
    ptr = _api(T, :new)(dims)
    return TensorCI2{T}(ptr, Int.(local_dims))
end

"""
    length(tci::TensorCI2) -> Int

Get the number of sites.
"""
function Base.length(tci::TensorCI2{T}) where T
    out_len = Ref{Csize_t}(0)
    status = _api(T, :len)(tci.ptr, out_len)
    C_API.check_status(status)
    return Int(out_len[])
end

"""
    rank(tci::TensorCI2) -> Int

Get the current maximum bond dimension (rank).
"""
function rank(tci::TensorCI2{T}) where T
    out_rank = Ref{Csize_t}(0)
    status = _api(T, :rank)(tci.ptr, out_rank)
    C_API.check_status(status)
    return Int(out_rank[])
end

"""
    link_dims(tci::TensorCI2) -> Vector{Int}

Get the link (bond) dimensions.
"""
function link_dims(tci::TensorCI2{T}) where T
    n = length(tci)
    n <= 1 && return Int[]
    dims = Vector{Csize_t}(undef, n - 1)
    status = _api(T, :link_dims)(tci.ptr, dims)
    C_API.check_status(status)
    return Int.(dims)
end

"""
    max_sample_value(tci::TensorCI2) -> Float64

Get the maximum sample value encountered during interpolation.
"""
function max_sample_value(tci::TensorCI2{T}) where T
    out_value = Ref{Cdouble}(0.0)
    status = _api(T, :max_sample_value)(tci.ptr, out_value)
    C_API.check_status(status)
    return out_value[]
end

"""
    max_bond_error(tci::TensorCI2) -> Float64

Get the maximum bond error from the last sweep.
"""
function max_bond_error(tci::TensorCI2{T}) where T
    out_value = Ref{Cdouble}(0.0)
    status = _api(T, :max_bond_error)(tci.ptr, out_value)
    C_API.check_status(status)
    return out_value[]
end

"""
    add_global_pivots!(tci::TensorCI2, pivots::Vector{Vector{Int}})

Add global pivots to the TCI. Each pivot is a vector of indices (0-based).
"""
function add_global_pivots!(tci::TensorCI2{T}, pivots::Vector{Vector{Int}}) where T
    isempty(pivots) && return

    n_sites = length(tci)
    n_pivots = length(pivots)
    flat_pivots = Csize_t[]
    for pivot in pivots
        length(pivot) == n_sites || error("Pivot length must match number of sites")
        append!(flat_pivots, Csize_t.(pivot))
    end

    status = _api(T, :add_global_pivots)(tci.ptr, flat_pivots, n_pivots, n_sites)
    C_API.check_status(status)
end

"""
    to_tensor_train(tci::TensorCI2) -> SimpleTensorTrain

Convert the TCI to a SimpleTensorTrain.
"""
function to_tensor_train(tci::TensorCI2{T}) where T
    ptr = _api(T, :to_tensor_train)(tci.ptr)
    ptr == C_NULL && error("Failed to convert TCI to TensorTrain")
    return SimpleTensorTrain{T}(ptr)
end

"""
    show(io::IO, tci::TensorCI2)

Display TCI information.
"""
function Base.show(io::IO, tci::TensorCI2{T}) where T
    n = length(tci)
    r = rank(tci)
    print(io, "TensorCI2{$T}(sites=$n, rank=$r)")
end

function Base.show(io::IO, ::MIME"text/plain", tci::TensorCI2{T}) where T
    println(io, "TensorCI2{$T}")
    println(io, "  Sites: $(length(tci))")
    println(io, "  Local dims: $(tci.local_dims)")
    println(io, "  Link dims: $(link_dims(tci))")
    println(io, "  Max rank: $(rank(tci))")
    println(io, "  Max bond error: $(max_bond_error(tci))")
end

# ============================================================================
# High-level crossinterpolate2 function
# ============================================================================

"""
    crossinterpolate2_tci([T], f, local_dims; kwargs...) -> (TensorCI2{T}, Float64)

Perform cross interpolation of `f` and return the stateful `TensorCI2`.
When `T` is omitted, the wrapper infers `Float64` or `ComplexF64` from
the first pivot evaluation.
"""
function crossinterpolate2_tci(
    ::Type{T},
    f,
    local_dims::Vector{<:Integer};
    initial_pivots::Vector{Vector{Int}} = [zeros(Int, length(local_dims))],
    tolerance::Float64 = 1e-8,
    max_bonddim::Int = 0,
    max_iter::Int = 20
) where {T<:_TensorCIScalar}
    n_sites = length(local_dims)
    dims = Csize_t.(local_dims)

    flat_pivots = Csize_t[]
    for pivot in initial_pivots
        length(pivot) == n_sites || error("Initial pivot length must match number of sites")
        append!(flat_pivots, Csize_t.(pivot))
    end
    n_initial_pivots = length(initial_pivots)

    f_ref = Ref{Any}(f)
    out_tci = Ref{Ptr{Cvoid}}(C_NULL)
    out_final_error = Ref{Cdouble}(0.0)

    GC.@preserve f_ref begin
        user_data = pointer_from_objref(f_ref)
        status = _cross_api(T)(
            dims,
            flat_pivots,
            n_initial_pivots,
            _get_trampoline_ptr(T),
            user_data,
            tolerance,
            max_bonddim,
            max_iter,
            out_tci,
            out_final_error
        )
        C_API.check_status(status)
    end

    tci = TensorCI2{T}(out_tci[], Int.(local_dims))
    return tci, out_final_error[]
end

function crossinterpolate2_tci(
    f,
    local_dims::Vector{<:Integer};
    initial_pivots::Vector{Vector{Int}} = [zeros(Int, length(local_dims))],
    tolerance::Float64 = 1e-8,
    max_bonddim::Int = 0,
    max_iter::Int = 20
)
    T = _infer_scalar_type(f, local_dims, initial_pivots)
    return crossinterpolate2_tci(
        T,
        f,
        local_dims;
        initial_pivots=initial_pivots,
        tolerance=tolerance,
        max_bonddim=max_bonddim,
        max_iter=max_iter,
    )
end

"""
    crossinterpolate2([T], f, local_dims; kwargs...) -> (SimpleTensorTrain{T}, Float64)

High-level wrapper that returns a `SimpleTensorTrain`. For advanced use, see
[`crossinterpolate2_tci`](@ref), which also returns the underlying `TensorCI2`.
"""
function crossinterpolate2(
    ::Type{T},
    f,
    local_dims::Vector{<:Integer};
    initial_pivots::Vector{Vector{Int}} = [zeros(Int, length(local_dims))],
    tolerance::Float64 = 1e-8,
    max_bonddim::Int = 0,
    max_iter::Int = 20
) where {T<:_TensorCIScalar}
    tci, err = crossinterpolate2_tci(
        T,
        f,
        local_dims;
        initial_pivots=initial_pivots,
        tolerance=tolerance,
        max_bonddim=max_bonddim,
        max_iter=max_iter,
    )
    return to_tensor_train(tci), err
end

function crossinterpolate2(
    f,
    local_dims::Vector{<:Integer};
    initial_pivots::Vector{Vector{Int}} = [zeros(Int, length(local_dims))],
    tolerance::Float64 = 1e-8,
    max_bonddim::Int = 0,
    max_iter::Int = 20
)
    tci, err = crossinterpolate2_tci(
        f,
        local_dims;
        initial_pivots=initial_pivots,
        tolerance=tolerance,
        max_bonddim=max_bonddim,
        max_iter=max_iter,
    )
    return to_tensor_train(tci), err
end

# ============================================================================
# Low-level sweep operations
# ============================================================================

"""
    sweep2site!(tci::TensorCI2, f; forward=true, tolerance=1e-8,
                max_bonddim=0, pivot_search=:full, strictly_nested=true)

Perform one 2-site sweep. `f` is called with 0-based indices.
"""
function sweep2site!(
    tci::TensorCI2{T},
    f;
    forward::Bool = true,
    tolerance::Float64 = 1e-8,
    max_bonddim::Int = 0,
    pivot_search::Symbol = :full,
    strictly_nested::Bool = true,
) where T
    ps = pivot_search == :full ? Cint(0) : Cint(1)
    f_ref = Ref{Any}(f)
    GC.@preserve f_ref begin
        user_data = pointer_from_objref(f_ref)
        status = _api(T, :sweep2site)(
            tci.ptr,
            _get_trampoline_ptr(T),
            user_data,
            Cint(forward),
            tolerance,
            Csize_t(max_bonddim),
            ps,
            Cint(strictly_nested)
        )
        C_API.check_status(status)
    end
end

"""
    sweep1site!(tci::TensorCI2, f; forward=true, rel_tol=1e-14,
                abs_tol=0.0, max_bonddim=0, update_tensors=true)

Perform one 1-site sweep for cleanup/canonicalization.
"""
function sweep1site!(
    tci::TensorCI2{T},
    f;
    forward::Bool = true,
    rel_tol::Float64 = 1e-14,
    abs_tol::Float64 = 0.0,
    max_bonddim::Int = 0,
    update_tensors::Bool = true,
) where T
    f_ref = Ref{Any}(f)
    GC.@preserve f_ref begin
        user_data = pointer_from_objref(f_ref)
        status = _api(T, :sweep1site)(
            tci.ptr,
            _get_trampoline_ptr(T),
            user_data,
            Cint(forward),
            rel_tol,
            abs_tol,
            Csize_t(max_bonddim),
            Cint(update_tensors)
        )
        C_API.check_status(status)
    end
end

"""
    fill_site_tensors!(tci::TensorCI2, f)

Fill all site tensors from function evaluations.
"""
function fill_site_tensors!(tci::TensorCI2{T}, f) where T
    f_ref = Ref{Any}(f)
    GC.@preserve f_ref begin
        user_data = pointer_from_objref(f_ref)
        status = _api(T, :fill_site_tensors)(tci.ptr, _get_trampoline_ptr(T), user_data)
        C_API.check_status(status)
    end
end

"""
    make_canonical!(tci::TensorCI2, f; rel_tol=1e-14, abs_tol=0.0, max_bonddim=0)

Make TCI canonical via 3 one-site sweeps.
"""
function make_canonical!(
    tci::TensorCI2{T},
    f;
    rel_tol::Float64 = 1e-14,
    abs_tol::Float64 = 0.0,
    max_bonddim::Int = 0,
) where T
    f_ref = Ref{Any}(f)
    GC.@preserve f_ref begin
        user_data = pointer_from_objref(f_ref)
        status = _api(T, :make_canonical)(
            tci.ptr,
            _get_trampoline_ptr(T),
            user_data,
            rel_tol,
            abs_tol,
            Csize_t(max_bonddim)
        )
        C_API.check_status(status)
    end
end

"""
    pivot_error(tci::TensorCI2) -> Float64

Get the maximum bond error from the last sweep.
"""
function pivot_error(tci::TensorCI2{T}) where T
    out = Ref{Cdouble}(0.0)
    status = _api(T, :pivot_error)(tci.ptr, out)
    C_API.check_status(status)
    return out[]
end

# ============================================================================
# I-set / J-set access
# ============================================================================

"""
    get_i_set(tci::TensorCI2, site::Integer) -> Vector{Vector{Int}}

Get the I-set at the given site (0-indexed).
"""
function get_i_set(tci::TensorCI2{T}, site::Integer) where T
    n_indices = Ref{Csize_t}(0)
    index_len = Ref{Csize_t}(0)
    status = _api(T, :i_set_size)(tci.ptr, Csize_t(site), n_indices, index_len)
    C_API.check_status(status)

    n = Int(n_indices[])
    il = Int(index_len[])
    n == 0 && return Vector{Int}[]

    buf = Vector{Csize_t}(undef, n * il)
    out_n = Ref{Csize_t}(0)
    out_il = Ref{Csize_t}(0)
    status = _api(T, :get_i_set)(tci.ptr, Csize_t(site), buf, Csize_t(n * il), out_n, out_il)
    C_API.check_status(status)

    return [Int.(buf[(i - 1) * il + 1:i * il]) for i in 1:n]
end

"""
    get_j_set(tci::TensorCI2, site::Integer) -> Vector{Vector{Int}}

Get the J-set at the given site (0-indexed).
"""
function get_j_set(tci::TensorCI2{T}, site::Integer) where T
    n_indices = Ref{Csize_t}(0)
    index_len = Ref{Csize_t}(0)
    status = _api(T, :j_set_size)(tci.ptr, Csize_t(site), n_indices, index_len)
    C_API.check_status(status)

    n = Int(n_indices[])
    il = Int(index_len[])
    n == 0 && return Vector{Int}[]

    buf = Vector{Csize_t}(undef, n * il)
    out_n = Ref{Csize_t}(0)
    out_il = Ref{Csize_t}(0)
    status = _api(T, :get_j_set)(tci.ptr, Csize_t(site), buf, Csize_t(n * il), out_n, out_il)
    C_API.check_status(status)

    return [Int.(buf[(i - 1) * il + 1:i * il]) for i in 1:n]
end

end # module TensorCI
