"""
    QuanticsTCI

High-level Julia wrappers for Quantics Tensor Cross Interpolation.

This module combines TCI with quantics grid representations to efficiently
interpolate functions on continuous or discrete domains.

Supports both `Float64` and `ComplexF64` element types.
"""
module QuanticsTCI

using ..C_API
using ..QuanticsGrids: DiscretizedGrid, InherentDiscreteGrid, localdimensions,
    _unfolding_to_cint
using ..SimpleTT: SimpleTensorTrain
import ..linkdims, ..evaluate, ..maxbonderror, ..maxrank

export QuanticsTensorCI2
export quanticscrossinterpolate
export evaluate, integral, to_tensor_train
export linkdims, maxbonderror, maxrank

# ============================================================================
# Type dispatch helpers
# ============================================================================

const _QtciScalar = Union{Float64, ComplexF64}

_suffix(::Type{Float64}) = "f64"
_suffix(::Type{ComplexF64}) = "c64"

_qtci_api(::Type{T}, name::Symbol) where {T<:_QtciScalar} =
    getfield(C_API, Symbol("t4a_qtci_", _suffix(T), "_", name))

# ============================================================================
# Callback infrastructure for passing Julia functions to Rust
# ============================================================================

# For continuous domain (f64 coordinates) returning Float64
function _trampoline_f64_f64(
    coords_ptr::Ptr{Float64},
    ndims::Csize_t,
    result_ptr::Ptr{Float64},
    user_data::Ptr{Cvoid},
)::Cint
    try
        f_ref = unsafe_pointer_to_objref(user_data)::Ref{Any}
        f = f_ref[]
        coords = unsafe_wrap(Array, coords_ptr, Int(ndims))
        val = Float64(f(coords...))
        unsafe_store!(result_ptr, val)
        return Cint(0)
    catch e
        @error "Error in QTCI f64 callback" exception = (e, catch_backtrace())
        return Cint(-1)
    end
end

# For continuous domain (f64 coordinates) returning ComplexF64
# Result buffer has 2 doubles: [re, im]
function _trampoline_f64_c64(
    coords_ptr::Ptr{Float64},
    ndims::Csize_t,
    result_ptr::Ptr{Float64},
    user_data::Ptr{Cvoid},
)::Cint
    try
        f_ref = unsafe_pointer_to_objref(user_data)::Ref{Any}
        f = f_ref[]
        coords = unsafe_wrap(Array, coords_ptr, Int(ndims))
        val = ComplexF64(f(coords...))
        unsafe_store!(result_ptr, real(val), 1)
        unsafe_store!(result_ptr, imag(val), 2)
        return Cint(0)
    catch e
        @error "Error in QTCI c64 callback" exception = (e, catch_backtrace())
        return Cint(-1)
    end
end

# For discrete domain (i64 indices, 1-indexed) returning Float64
function _trampoline_i64_f64(
    indices_ptr::Ptr{Int64},
    ndims::Csize_t,
    result_ptr::Ptr{Float64},
    user_data::Ptr{Cvoid},
)::Cint
    try
        f_ref = unsafe_pointer_to_objref(user_data)::Ref{Any}
        f = f_ref[]
        indices = unsafe_wrap(Array, indices_ptr, Int(ndims))
        val = Float64(f(indices...))
        unsafe_store!(result_ptr, val)
        return Cint(0)
    catch e
        @error "Error in QTCI discrete f64 callback" exception = (e, catch_backtrace())
        return Cint(-1)
    end
end

# For discrete domain (i64 indices, 1-indexed) returning ComplexF64
function _trampoline_i64_c64(
    indices_ptr::Ptr{Int64},
    ndims::Csize_t,
    result_ptr::Ptr{Float64},
    user_data::Ptr{Cvoid},
)::Cint
    try
        f_ref = unsafe_pointer_to_objref(user_data)::Ref{Any}
        f = f_ref[]
        indices = unsafe_wrap(Array, indices_ptr, Int(ndims))
        val = ComplexF64(f(indices...))
        unsafe_store!(result_ptr, real(val), 1)
        unsafe_store!(result_ptr, imag(val), 2)
        return Cint(0)
    catch e
        @error "Error in QTCI discrete c64 callback" exception = (e, catch_backtrace())
        return Cint(-1)
    end
end

# Create C function pointers lazily to avoid precompilation issues
const _TRAMPOLINE_F64_F64_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const _TRAMPOLINE_F64_C64_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const _TRAMPOLINE_I64_F64_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const _TRAMPOLINE_I64_C64_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _get_trampoline_f64(::Type{Float64})
    if _TRAMPOLINE_F64_F64_PTR[] == C_NULL
        _TRAMPOLINE_F64_F64_PTR[] = @cfunction(
            _trampoline_f64_f64,
            Cint,
            (Ptr{Float64}, Csize_t, Ptr{Float64}, Ptr{Cvoid})
        )
    end
    return _TRAMPOLINE_F64_F64_PTR[]
end

function _get_trampoline_f64(::Type{ComplexF64})
    if _TRAMPOLINE_F64_C64_PTR[] == C_NULL
        _TRAMPOLINE_F64_C64_PTR[] = @cfunction(
            _trampoline_f64_c64,
            Cint,
            (Ptr{Float64}, Csize_t, Ptr{Float64}, Ptr{Cvoid})
        )
    end
    return _TRAMPOLINE_F64_C64_PTR[]
end

function _get_trampoline_i64(::Type{Float64})
    if _TRAMPOLINE_I64_F64_PTR[] == C_NULL
        _TRAMPOLINE_I64_F64_PTR[] = @cfunction(
            _trampoline_i64_f64,
            Cint,
            (Ptr{Int64}, Csize_t, Ptr{Float64}, Ptr{Cvoid})
        )
    end
    return _TRAMPOLINE_I64_F64_PTR[]
end

function _get_trampoline_i64(::Type{ComplexF64})
    if _TRAMPOLINE_I64_C64_PTR[] == C_NULL
        _TRAMPOLINE_I64_C64_PTR[] = @cfunction(
            _trampoline_i64_c64,
            Cint,
            (Ptr{Int64}, Csize_t, Ptr{Float64}, Ptr{Cvoid})
        )
    end
    return _TRAMPOLINE_I64_C64_PTR[]
end

# ============================================================================
# QtciOptions helper
# ============================================================================

"""
Build a QtciOptions handle from keyword arguments.
Caller must call `C_API.t4a_qtci_options_release(opts)` after use.
"""
function _build_options(;
    tolerance::Float64 = 1e-8,
    maxbonddim::Int = 0,
    maxiter::Int = 200,
    nrandominitpivot::Int = 5,
    verbosity::Int = 0,
    unfoldingscheme::Symbol = :interleaved,
    nsearchglobalpivot::Int = 5,
    nsearch::Int = 100,
    normalizeerror::Bool = true,
)
    opts = C_API.t4a_qtci_options_default()
    opts == C_NULL && error("Failed to create QtciOptions")

    C_API.check_status(C_API.t4a_qtci_options_set_tolerance(opts, tolerance))
    C_API.check_status(C_API.t4a_qtci_options_set_maxbonddim(opts, maxbonddim))
    C_API.check_status(C_API.t4a_qtci_options_set_maxiter(opts, maxiter))
    C_API.check_status(C_API.t4a_qtci_options_set_nrandominitpivot(opts, nrandominitpivot))
    C_API.check_status(C_API.t4a_qtci_options_set_verbosity(opts, verbosity))

    scheme_c = _unfolding_to_cint(unfoldingscheme)
    C_API.check_status(C_API.t4a_qtci_options_set_unfoldingscheme(opts, scheme_c))

    C_API.check_status(C_API.t4a_qtci_options_set_nsearchglobalpivot(opts, nsearchglobalpivot))
    C_API.check_status(C_API.t4a_qtci_options_set_nsearch(opts, nsearch))
    C_API.check_status(C_API.t4a_qtci_options_set_normalize_error(opts, normalizeerror ? 1 : 0))

    return opts
end

# ============================================================================
# QuanticsTensorCI2 type
# ============================================================================

"""
    QuanticsTensorCI2{V}

A Quantics TCI (Tensor Cross Interpolation) object.

`V` can be `Float64` or `ComplexF64`, determining which C API variant is used.

# Methods

- `linkdims(qtci)` - Get the link (bond) dimensions
- `evaluate(qtci, indices)` - Evaluate at grid indices
- `sum(qtci)` - Compute the factorized sum over all grid points
- `integral(qtci)` - Compute the integral over the continuous domain
- `to_tensor_train(qtci)` - Convert to SimpleTensorTrain{V}
- `maxbonderror(qtci)` - Get the maximum bond error
- `maxrank(qtci)` - Get the maximum rank (bond dimension)

# Callable Interface

The object can be called directly with indices:
```julia
qtci(1, 2)  # Equivalent to evaluate(qtci, [1, 2])
```
"""
mutable struct QuanticsTensorCI2{V<:_QtciScalar}
    ptr::Ptr{Cvoid}

    function QuanticsTensorCI2{V}(ptr::Ptr{Cvoid}) where {V<:_QtciScalar}
        ptr == C_NULL && error("Failed to create QuanticsTensorCI2: null pointer")
        qtci = new{V}(ptr)
        finalizer(qtci) do obj
            _qtci_api(V, :release)(obj.ptr)
        end
        return qtci
    end
end

# ============================================================================
# Accessors
# ============================================================================

"""
    linkdims(qtci::QuanticsTensorCI2) -> Vector{Int}

Get the link (bond) dimensions.
"""
function linkdims(qtci::QuanticsTensorCI2{V}) where {V}
    # Use a generous buffer
    buf = Vector{Csize_t}(undef, 1024)
    status = _qtci_api(V, :link_dims)(qtci.ptr, buf, Csize_t(length(buf)))
    C_API.check_status(status)
    # Find actual length by looking for trailing zeros
    result = Int.(buf)
    last_nonzero = findlast(x -> x > 0, result)
    return isnothing(last_nonzero) ? Int[] : result[1:last_nonzero]
end

"""
    maxbonderror(qtci::QuanticsTensorCI2) -> Float64

Get the maximum bond error from the QuanticsTCI.
"""
function maxbonderror(qtci::QuanticsTensorCI2{V}) where {V}
    out_value = Ref{Cdouble}(0.0)
    status = _qtci_api(V, :max_bond_error)(qtci.ptr, out_value)
    C_API.check_status(status)
    return out_value[]
end

"""
    maxrank(qtci::QuanticsTensorCI2) -> Int

Get the maximum rank (bond dimension) from the QuanticsTCI.
"""
function maxrank(qtci::QuanticsTensorCI2{V}) where {V}
    out_rank = Ref{Csize_t}(0)
    status = _qtci_api(V, :max_rank)(qtci.ptr, out_rank)
    C_API.check_status(status)
    return Int(out_rank[])
end

# ============================================================================
# Operations: Float64
# ============================================================================

"""
    evaluate(qtci::QuanticsTensorCI2{Float64}, indices::Vector{<:Integer}) -> Float64

Evaluate the QTCI at the given grid indices.
"""
function evaluate(qtci::QuanticsTensorCI2{Float64}, indices::Vector{<:Integer})
    idx = Int64.(indices)
    out_value = Ref{Cdouble}(0.0)
    status = C_API.t4a_qtci_f64_evaluate(qtci.ptr, idx, Csize_t(length(idx)), out_value)
    C_API.check_status(status)
    return out_value[]
end

"""
    Base.sum(qtci::QuanticsTensorCI2{Float64}) -> Float64

Compute the factorized sum over all grid points.
"""
function Base.sum(qtci::QuanticsTensorCI2{Float64})
    out_value = Ref{Cdouble}(0.0)
    status = C_API.t4a_qtci_f64_sum(qtci.ptr, out_value)
    C_API.check_status(status)
    return out_value[]
end

"""
    integral(qtci::QuanticsTensorCI2{Float64}) -> Float64

Compute the integral over the continuous domain.
"""
function integral(qtci::QuanticsTensorCI2{Float64})
    out_value = Ref{Cdouble}(0.0)
    status = C_API.t4a_qtci_f64_integral(qtci.ptr, out_value)
    C_API.check_status(status)
    return out_value[]
end

"""
    to_tensor_train(qtci::QuanticsTensorCI2{Float64}) -> SimpleTensorTrain{Float64}

Convert the QTCI to a SimpleTensorTrain.
"""
function to_tensor_train(qtci::QuanticsTensorCI2{Float64})
    ptr = C_API.t4a_qtci_f64_to_tensor_train(qtci.ptr)
    ptr == C_NULL && error("Failed to convert QTCI to TensorTrain")
    return SimpleTensorTrain{Float64}(ptr)
end

# ============================================================================
# Operations: ComplexF64
# ============================================================================

"""
    evaluate(qtci::QuanticsTensorCI2{ComplexF64}, indices::Vector{<:Integer}) -> ComplexF64

Evaluate the QTCI at the given grid indices.
"""
function evaluate(qtci::QuanticsTensorCI2{ComplexF64}, indices::Vector{<:Integer})
    idx = Int64.(indices)
    out_re = Ref{Cdouble}(0.0)
    out_im = Ref{Cdouble}(0.0)
    status = C_API.t4a_qtci_c64_evaluate(qtci.ptr, idx, Csize_t(length(idx)), out_re, out_im)
    C_API.check_status(status)
    return ComplexF64(out_re[], out_im[])
end

"""
    Base.sum(qtci::QuanticsTensorCI2{ComplexF64}) -> ComplexF64

Compute the factorized sum over all grid points.
"""
function Base.sum(qtci::QuanticsTensorCI2{ComplexF64})
    out_re = Ref{Cdouble}(0.0)
    out_im = Ref{Cdouble}(0.0)
    status = C_API.t4a_qtci_c64_sum(qtci.ptr, out_re, out_im)
    C_API.check_status(status)
    return ComplexF64(out_re[], out_im[])
end

"""
    integral(qtci::QuanticsTensorCI2{ComplexF64}) -> ComplexF64

Compute the integral over the continuous domain.
"""
function integral(qtci::QuanticsTensorCI2{ComplexF64})
    out_re = Ref{Cdouble}(0.0)
    out_im = Ref{Cdouble}(0.0)
    status = C_API.t4a_qtci_c64_integral(qtci.ptr, out_re, out_im)
    C_API.check_status(status)
    return ComplexF64(out_re[], out_im[])
end

"""
    to_tensor_train(qtci::QuanticsTensorCI2{ComplexF64}) -> SimpleTensorTrain{ComplexF64}

Convert the QTCI to a SimpleTensorTrain.
"""
function to_tensor_train(qtci::QuanticsTensorCI2{ComplexF64})
    ptr = C_API.t4a_qtci_c64_to_tensor_train(qtci.ptr)
    ptr == C_NULL && error("Failed to convert QTCI to TensorTrain")
    return SimpleTensorTrain{ComplexF64}(ptr)
end

# ============================================================================
# Callable interface
# ============================================================================

function (qtci::QuanticsTensorCI2)(indices::Integer...)
    evaluate(qtci, collect(Int64, indices))
end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, qtci::QuanticsTensorCI2{V}) where {V}
    r = maxrank(qtci)
    print(io, "QuanticsTensorCI2{$V}(maxrank=$r)")
end

# ============================================================================
# High-level interpolation functions
# ============================================================================

"""
    quanticscrossinterpolate(::Type{V}, f, grid::DiscretizedGrid; kwargs...) -> (QuanticsTensorCI2{V}, Vector{Int}, Vector{Float64})

Perform quantics cross interpolation on a continuous domain.

# Arguments
- `V`: Element type (`Float64` or `ComplexF64`)
- `f`: Function that takes Float64 coordinates and returns `V`
- `grid`: DiscretizedGrid describing the domain

# Keyword Arguments
- `tolerance::Float64 = 1e-8`: Convergence tolerance
- `maxbonddim::Int = 0`: Maximum bond dimension (0 = unlimited)
- `maxiter::Int = 200`: Maximum iterations
- `initialpivots::Union{Nothing, Vector{Vector{Int}}} = nothing`: Initial pivots (1-indexed)
- `nrandominitpivot::Int = 5`: Number of random initial pivots
- `verbosity::Int = 0`: Verbosity level
- `unfoldingscheme::Symbol = :interleaved`: `:interleaved` or `:fused`
- `nsearchglobalpivot::Int = 5`: Number of global pivot searches
- `nsearch::Int = 100`: Number of searches
- `normalizeerror::Bool = true`: Whether to normalize errors

# Returns
- `qtci::QuanticsTensorCI2{V}`: The QTCI object
- `ranks::Vector{Int}`: Per-iteration maximum ranks
- `errors::Vector{Float64}`: Per-iteration errors

# Example
```julia
using Tensor4all.QuanticsGrids
using Tensor4all.QuanticsTCI

grid = DiscretizedGrid(1, 10, [0.0], [1.0])
qtci, ranks, errors = quanticscrossinterpolate(Float64, x -> sin(x), grid)
integral(qtci)  # Should be close to 1 - cos(1) ~ 0.4597
```
"""
function quanticscrossinterpolate(
    ::Type{V},
    f,
    grid::DiscretizedGrid;
    tolerance::Float64 = 1e-8,
    maxbonddim::Int = 0,
    maxiter::Int = 200,
    initialpivots::Union{Nothing, Vector{Vector{Int}}} = nothing,
    nrandominitpivot::Int = 5,
    verbosity::Int = 0,
    unfoldingscheme::Symbol = :interleaved,
    nsearchglobalpivot::Int = 5,
    nsearch::Int = 100,
    normalizeerror::Bool = true,
) where {V<:_QtciScalar}
    # Build options
    opts = _build_options(;
        tolerance, maxbonddim, maxiter, nrandominitpivot,
        verbosity, unfoldingscheme, nsearchglobalpivot, nsearch, normalizeerror,
    )

    try
        # Prepare initial pivots (convert 1-indexed to C API format if needed)
        if initialpivots !== nothing && !isempty(initialpivots)
            n_sites = length(initialpivots[1])
            n_pivots = length(initialpivots)
            flat_pivots = Vector{Int64}(undef, n_sites * n_pivots)
            for j in 1:n_pivots
                for i in 1:n_sites
                    flat_pivots[i + n_sites * (j - 1)] = Int64(initialpivots[j][i])
                end
            end
            pivots_ptr = flat_pivots
            pivots_n = Csize_t(n_pivots)
        else
            pivots_ptr = C_NULL
            pivots_n = Csize_t(0)
        end

        # Prepare output buffers
        out_qtci = Ref{Ptr{Cvoid}}(C_NULL)
        out_ranks = Vector{Csize_t}(undef, maxiter)
        out_errors = Vector{Cdouble}(undef, maxiter)
        out_n_iters = Ref{Csize_t}(0)

        f_ref = Ref{Any}(f)
        GC.@preserve f_ref pivots_ptr begin
            user_data = pointer_from_objref(f_ref)
            trampoline_ptr = _get_trampoline_f64(V)

            crossinterp_fn = if V === Float64
                C_API.t4a_quanticscrossinterpolate_f64
            else
                C_API.t4a_quanticscrossinterpolate_c64
            end

            status = crossinterp_fn(
                grid.ptr,
                trampoline_ptr,
                user_data,
                opts,
                tolerance,
                Csize_t(maxbonddim),
                Csize_t(maxiter),
                pivots_ptr,
                pivots_n,
                out_qtci,
                out_ranks,
                out_errors,
                out_n_iters,
            )

            C_API.check_status(status)
        end

        n_iters = Int(out_n_iters[])
        ranks = Vector{Int}(out_ranks[1:n_iters])
        errors = Vector{Float64}(out_errors[1:n_iters])
        qtci = QuanticsTensorCI2{V}(out_qtci[])

        return qtci, ranks, errors
    finally
        C_API.t4a_qtci_options_release(opts)
    end
end

"""
    quanticscrossinterpolate(::Type{V}, f, size::NTuple{N,Int}; kwargs...) -> (QuanticsTensorCI2{V}, Vector{Int}, Vector{Float64})

Perform quantics cross interpolation on a discrete integer domain.

# Arguments
- `V`: Element type (`Float64` or `ComplexF64`)
- `f`: Function that takes 1-indexed integer indices and returns `V`
- `size`: Grid sizes per dimension (must be powers of 2)

# Keyword Arguments
Same as the `DiscretizedGrid` variant, plus:
- `unfoldingscheme::Symbol = :interleaved`: `:interleaved` or `:fused`

# Example
```julia
qtci, ranks, errors = quanticscrossinterpolate(Float64, (i, j) -> Float64(i + j), (8, 8))
qtci(3, 4)  # Should be close to 7.0
```
"""
function quanticscrossinterpolate(
    ::Type{V},
    f,
    size::NTuple{N,Int};
    tolerance::Float64 = 1e-8,
    maxbonddim::Int = 0,
    maxiter::Int = 200,
    initialpivots::Union{Nothing, Vector{Vector{Int}}} = nothing,
    nrandominitpivot::Int = 5,
    verbosity::Int = 0,
    unfoldingscheme::Symbol = :interleaved,
    nsearchglobalpivot::Int = 5,
    nsearch::Int = 100,
    normalizeerror::Bool = true,
) where {V<:_QtciScalar, N}
    # Build options
    opts = _build_options(;
        tolerance, maxbonddim, maxiter, nrandominitpivot,
        verbosity, unfoldingscheme, nsearchglobalpivot, nsearch, normalizeerror,
    )

    try
        sizes_c = Csize_t.(collect(size))
        scheme_c = _unfolding_to_cint(unfoldingscheme)

        # Prepare initial pivots
        if initialpivots !== nothing && !isempty(initialpivots)
            n_sites_pivot = length(initialpivots[1])
            n_pivots = length(initialpivots)
            flat_pivots = Vector{Int64}(undef, n_sites_pivot * n_pivots)
            for j in 1:n_pivots
                for i in 1:n_sites_pivot
                    flat_pivots[i + n_sites_pivot * (j - 1)] = Int64(initialpivots[j][i])
                end
            end
            pivots_ptr = flat_pivots
            pivots_n = Csize_t(n_pivots)
        else
            pivots_ptr = C_NULL
            pivots_n = Csize_t(0)
        end

        # Prepare output buffers
        out_qtci = Ref{Ptr{Cvoid}}(C_NULL)
        out_ranks = Vector{Csize_t}(undef, maxiter)
        out_errors = Vector{Cdouble}(undef, maxiter)
        out_n_iters = Ref{Csize_t}(0)

        f_ref = Ref{Any}(f)
        GC.@preserve f_ref pivots_ptr begin
            user_data = pointer_from_objref(f_ref)
            trampoline_ptr = _get_trampoline_i64(V)

            crossinterp_fn = if V === Float64
                C_API.t4a_quanticscrossinterpolate_discrete_f64
            else
                C_API.t4a_quanticscrossinterpolate_discrete_c64
            end

            status = crossinterp_fn(
                sizes_c,
                Csize_t(N),
                trampoline_ptr,
                user_data,
                opts,
                tolerance,
                Csize_t(maxbonddim),
                Csize_t(maxiter),
                scheme_c,
                pivots_ptr,
                pivots_n,
                out_qtci,
                out_ranks,
                out_errors,
                out_n_iters,
            )

            C_API.check_status(status)
        end

        n_iters = Int(out_n_iters[])
        ranks = Vector{Int}(out_ranks[1:n_iters])
        errors = Vector{Float64}(out_errors[1:n_iters])
        qtci = QuanticsTensorCI2{V}(out_qtci[])

        return qtci, ranks, errors
    finally
        C_API.t4a_qtci_options_release(opts)
    end
end

"""
    quanticscrossinterpolate(F::Array{V}; kwargs...) -> (QuanticsTensorCI2{V}, Vector{Int}, Vector{Float64})

Perform quantics cross interpolation from a dense array.

The array is wrapped as a function and the size tuple variant is called.
Grid sizes are taken from `size(F)`.

# Example
```julia
F = [Float64(i + j) for i in 1:8, j in 1:8]
qtci, ranks, errors = quanticscrossinterpolate(F)
```
"""
function quanticscrossinterpolate(
    F::Array{V};
    kwargs...
) where {V<:_QtciScalar}
    sz = Base.size(F)
    f = (indices...) -> F[indices...]
    return quanticscrossinterpolate(V, f, sz; kwargs...)
end

end # module QuanticsTCI
