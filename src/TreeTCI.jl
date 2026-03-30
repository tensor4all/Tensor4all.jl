"""
    TreeTCI

Tree-structured tensor cross interpolation via tensor4all-rs.

Provides `TreeTciGraph` for defining tree topologies and `SimpleTreeTci`
for running TCI on arbitrary tree structures. Results are returned as
`TreeTN.TreeTensorNetwork`.

# Usage
```julia
using Tensor4all.TreeTCI

graph = TreeTciGraph(4, [(0, 1), (1, 2), (2, 3)])
f(batch) = [sum(Float64, batch[:, j]) for j in 1:size(batch, 2)]
tci, ranks, errors = crossinterpolate2(f, [3, 3, 3, 3], graph)
ttn = to_treetn(tci, f)
```
"""
module TreeTCI

using ..C_API
import ..TreeTN: TreeTensorNetwork

export TreeTciGraph, SimpleTreeTci
export crossinterpolate2

const _TreeTciScalar = Union{Float64, ComplexF64}

_suffix(::Type{Float64}) = "f64"
_suffix(::Type{ComplexF64}) = "c64"
_sym_for(::Type{T}, name::Symbol) where {T<:_TreeTciScalar} =
    C_API._sym(Symbol("t4a_treetci_", _suffix(T), "_", name))
_cross_sym_for(::Type{T}) where {T<:_TreeTciScalar} =
    C_API._sym(Symbol("t4a_crossinterpolate_tree_", _suffix(T)))

function _infer_scalar_type(f, local_dims::Vector{<:Integer}, initial_pivots::Vector{Vector{Int}})
    sample_indices = isempty(initial_pivots) ? zeros(Int, length(local_dims)) : initial_pivots[1]
    sample_values = f(reshape(sample_indices, :, 1))
    length(sample_values) == 1 ||
        error("TreeTCI batch callback must return exactly one value for a single-point batch")
    sample_value = sample_values[1]
    if sample_value isa Real
        return Float64
    elseif sample_value isa Complex
        return ComplexF64
    end
    error("TreeTCI batch callback must return real or complex values, got $(typeof(sample_value))")
end

# ============================================================================
# TreeTciGraph
# ============================================================================

"""
    TreeTciGraph(n_sites, edges)

Define a tree graph structure for TreeTCI.

# Arguments
- `n_sites::Int`: Number of sites
- `edges::Vector{Tuple{Int,Int}}`: Edge list (0-based site indices)

# Examples
```julia
# Linear chain: 0-1-2-3
graph = TreeTciGraph(4, [(0, 1), (1, 2), (2, 3)])

# Star graph: 0 at center
graph = TreeTciGraph(4, [(0, 1), (0, 2), (0, 3)])

# 7-site branching tree
graph = TreeTciGraph(7, [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (4, 6)])
```
"""
mutable struct TreeTciGraph
    ptr::Ptr{Cvoid}
    n_sites::Int

    function TreeTciGraph(n_sites::Int, edges::Vector{Tuple{Int, Int}})
        n_edges = length(edges)
        edges_flat = Vector{Csize_t}(undef, 2 * n_edges)
        for (i, (u, v)) in enumerate(edges)
            edges_flat[2 * i - 1] = Csize_t(u)
            edges_flat[2 * i] = Csize_t(v)
        end

        ptr = ccall(
            C_API._sym(:t4a_treetci_graph_new),
            Ptr{Cvoid},
            (Csize_t, Ptr{Csize_t}, Csize_t),
            Csize_t(n_sites),
            n_edges == 0 ? C_NULL : edges_flat,
            Csize_t(n_edges),
        )
        if ptr == C_NULL
            error("Failed to create TreeTciGraph: $(C_API.last_error_message())")
        end

        graph = new(ptr, n_sites)
        finalizer(graph) do obj
            if obj.ptr != C_NULL
                ccall(C_API._sym(:t4a_treetci_graph_release), Cvoid, (Ptr{Cvoid},), obj.ptr)
                obj.ptr = C_NULL
            end
        end
        return graph
    end
end

# ============================================================================
# Batch Eval Trampoline
# ============================================================================

"""
Internal trampoline for f64 batch callbacks.

The user function signature is: `f(batch::Matrix{Csize_t}) -> Vector{Float64}`
where `batch` is column-major `(n_sites, n_points)` with 0-based indices.
"""
function _treetci_batch_trampoline(
    batch_data::Ptr{Csize_t},
    n_sites::Csize_t,
    n_points::Csize_t,
    results::Ptr{Cdouble},
    user_data::Ptr{Cvoid},
)::Cint
    try
        f_ref = unsafe_pointer_to_objref(user_data)::Ref{Any}
        f = f_ref[]
        batch = unsafe_wrap(Array, batch_data, (Int(n_sites), Int(n_points)))
        vals = f(batch)
        length(vals) == Int(n_points) ||
            error("Batch callback returned $(length(vals)) values for $(Int(n_points)) points")
        for i in 1:Int(n_points)
            unsafe_store!(results, Float64(vals[i]), i)
        end
        return Cint(0)
    catch err
        @error "TreeTCI batch eval callback error" exception = (err, catch_backtrace())
        return Cint(-1)
    end
end

"""
Internal trampoline for c64 batch callbacks.

The user function signature is: `f(batch::Matrix{Csize_t}) -> Vector{ComplexF64}`
where `batch` is column-major `(n_sites, n_points)` with 0-based indices.
Results are written as interleaved doubles.
"""
function _treetci_batch_trampoline_c64(
    batch_data::Ptr{Csize_t},
    n_sites::Csize_t,
    n_points::Csize_t,
    results::Ptr{Cdouble},
    user_data::Ptr{Cvoid},
)::Cint
    try
        f_ref = unsafe_pointer_to_objref(user_data)::Ref{Any}
        f = f_ref[]
        batch = unsafe_wrap(Array, batch_data, (Int(n_sites), Int(n_points)))
        vals = ComplexF64.(f(batch))
        length(vals) == Int(n_points) ||
            error("Batch callback returned $(length(vals)) values for $(Int(n_points)) points")
        interleaved = reinterpret(Float64, vals)
        for i in eachindex(interleaved)
            unsafe_store!(results, interleaved[i], i)
        end
        return Cint(0)
    catch err
        @error "TreeTCI complex batch eval callback error" exception = (err, catch_backtrace())
        return Cint(-1)
    end
end

const _BATCH_TRAMPOLINE_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const _BATCH_TRAMPOLINE_C64_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _get_batch_trampoline(::Type{Float64})
    if _BATCH_TRAMPOLINE_PTR[] == C_NULL
        _BATCH_TRAMPOLINE_PTR[] = @cfunction(
            _treetci_batch_trampoline,
            Cint,
            (Ptr{Csize_t}, Csize_t, Csize_t, Ptr{Cdouble}, Ptr{Cvoid}),
        )
    end
    return _BATCH_TRAMPOLINE_PTR[]
end

function _get_batch_trampoline(::Type{ComplexF64})
    if _BATCH_TRAMPOLINE_C64_PTR[] == C_NULL
        _BATCH_TRAMPOLINE_C64_PTR[] = @cfunction(
            _treetci_batch_trampoline_c64,
            Cint,
            (Ptr{Csize_t}, Csize_t, Csize_t, Ptr{Cdouble}, Ptr{Cvoid}),
        )
    end
    return _BATCH_TRAMPOLINE_C64_PTR[]
end

# ============================================================================
# Proposer helpers
# ============================================================================

const _PROPOSER_DEFAULT = Cint(0)
const _PROPOSER_SIMPLE = Cint(1)
const _PROPOSER_TRUNCATED_DEFAULT = Cint(2)

function _proposer_to_cint(proposer::Symbol)::Cint
    if proposer === :default
        return _PROPOSER_DEFAULT
    elseif proposer === :simple
        return _PROPOSER_SIMPLE
    elseif proposer === :truncated_default
        return _PROPOSER_TRUNCATED_DEFAULT
    end
    error("Unknown proposer: $proposer. Use :default, :simple, or :truncated_default")
end

# ============================================================================
# SimpleTreeTci
# ============================================================================

"""
    SimpleTreeTci{T<:Union{Float64, ComplexF64}}(local_dims, graph)

Stateful TreeTCI object for tree-structured tensor cross interpolation.

# Arguments
- `local_dims::Vector{Int}`: Local dimension at each site (length = graph.n_sites)
- `graph::TreeTciGraph`: Tree graph structure

# Lifecycle
```julia
tci = SimpleTreeTci([2, 2, 2, 2], graph)
add_global_pivots!(tci, [zeros(Int, 4)])
for _ in 1:20
    sweep!(tci, f; tolerance=1e-8)
    max_bond_error(tci) < 1e-8 && break
end
ttn = to_treetn(tci, f)
```
"""
mutable struct SimpleTreeTci{T<:_TreeTciScalar}
    ptr::Ptr{Cvoid}
    graph::TreeTciGraph
    local_dims::Vector{Int}

    function SimpleTreeTci{T}(local_dims::Vector{<:Integer}, graph::TreeTciGraph) where {T<:_TreeTciScalar}
        dims_int = Int.(local_dims)
        length(dims_int) == graph.n_sites ||
            error("local_dims length ($(length(dims_int))) != graph.n_sites ($(graph.n_sites))")

        dims_csize = Csize_t.(dims_int)
        ptr = ccall(
            _sym_for(T, :new),
            Ptr{Cvoid},
            (Ptr{Csize_t}, Csize_t, Ptr{Cvoid}),
            dims_csize,
            Csize_t(length(dims_csize)),
            graph.ptr,
        )
        if ptr == C_NULL
            error("Failed to create SimpleTreeTci: $(C_API.last_error_message())")
        end

        tci = new{T}(ptr, graph, dims_int)
        finalizer(tci) do obj
            if obj.ptr != C_NULL
                ccall(_sym_for(T, :release), Cvoid, (Ptr{Cvoid},), obj.ptr)
                obj.ptr = C_NULL
            end
        end
        return tci
    end
end

SimpleTreeTci(local_dims::Vector{<:Integer}, graph::TreeTciGraph) =
    SimpleTreeTci{Float64}(local_dims, graph)

# ============================================================================
# Pivot management
# ============================================================================

"""
    add_global_pivots!(tci, pivots)

Add global pivots. Each pivot is a full multi-index over all sites (0-based).

# Arguments
- `tci::SimpleTreeTci`
- `pivots::Vector{Vector{Int}}`: Each element has length `n_sites`, 0-based indices
"""
function add_global_pivots!(tci::SimpleTreeTci{T}, pivots::Vector{Vector{Int}}) where {T}
    n_sites = length(tci.local_dims)
    n_pivots = length(pivots)
    n_pivots == 0 && return tci

    pivots_flat = Vector{Csize_t}(undef, n_sites * n_pivots)
    for j in 1:n_pivots
        pivot = pivots[j]
        length(pivot) == n_sites ||
            error("Pivot $j has length $(length(pivot)), expected $n_sites")
        for i in 1:n_sites
            pivots_flat[i + n_sites * (j - 1)] = Csize_t(pivot[i])
        end
    end

    C_API.check_status(ccall(
        _sym_for(T, :add_global_pivots),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Csize_t),
        tci.ptr,
        pivots_flat,
        Csize_t(n_sites),
        Csize_t(n_pivots),
    ))
    return tci
end

# ============================================================================
# Sweep
# ============================================================================

"""
    sweep!(tci, f; proposer=:default, tolerance=1e-8, max_bond_dim=0)

Run one optimization iteration (visit all edges once).

# Arguments
- `tci::SimpleTreeTci`
- `f`: Batch evaluation function `f(batch::Matrix{Csize_t}) -> Vector{T}`
  where `batch` is column-major `(n_sites, n_points)` with 0-based indices
- `proposer`: `:default`, `:simple`, or `:truncated_default`
- `tolerance`: Relative tolerance
- `max_bond_dim`: Maximum bond dimension (0 = unlimited)
"""
function sweep!(
    tci::SimpleTreeTci{T},
    f;
    proposer::Symbol = :default,
    tolerance::Float64 = 1e-8,
    max_bond_dim::Int = 0,
) where {T}
    f_ref = Ref{Any}(f)
    GC.@preserve f_ref begin
        C_API.check_status(ccall(
            _sym_for(T, :sweep),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cint, Cdouble, Csize_t),
            tci.ptr,
            _get_batch_trampoline(T),
            pointer_from_objref(f_ref),
            _proposer_to_cint(proposer),
            tolerance,
            Csize_t(max_bond_dim),
        ))
    end
    return tci
end

# ============================================================================
# State inspection
# ============================================================================

"""Maximum bond error across all edges."""
function max_bond_error(tci::SimpleTreeTci{T}) where {T}
    out = Ref{Cdouble}(0.0)
    C_API.check_status(ccall(
        _sym_for(T, :max_bond_error),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        tci.ptr,
        out,
    ))
    return out[]
end

"""Maximum rank (bond dimension) across all edges."""
function max_rank(tci::SimpleTreeTci{T}) where {T}
    out = Ref{Csize_t}(0)
    C_API.check_status(ccall(
        _sym_for(T, :max_rank),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        tci.ptr,
        out,
    ))
    return Int(out[])
end

"""Maximum observed sample value (for normalization)."""
function max_sample_value(tci::SimpleTreeTci{T}) where {T}
    out = Ref{Cdouble}(0.0)
    C_API.check_status(ccall(
        _sym_for(T, :max_sample_value),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        tci.ptr,
        out,
    ))
    return out[]
end

"""Bond dimensions (ranks) at each edge."""
function bond_dims(tci::SimpleTreeTci{T}) where {T}
    n_edges_ref = Ref{Csize_t}(0)
    C_API.check_status(ccall(
        _sym_for(T, :bond_dims),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
        tci.ptr,
        Ptr{Csize_t}(C_NULL),
        Csize_t(0),
        n_edges_ref,
    ))

    n_edges = Int(n_edges_ref[])
    buf = Vector{Csize_t}(undef, n_edges)
    C_API.check_status(ccall(
        _sym_for(T, :bond_dims),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
        tci.ptr,
        buf,
        Csize_t(n_edges),
        n_edges_ref,
    ))
    return Int.(buf)
end

# ============================================================================
# Materialization
# ============================================================================

function _wrap_treetn(handle::Ptr{Cvoid}, n_sites::Int)
    node_names = collect(1:n_sites)
    node_map = Dict{Int, Int}(i => i - 1 for i in node_names)
    return TreeTensorNetwork{Int}(handle, node_map, node_names)
end

"""
    to_treetn(tci, f; center_site=0)

Convert converged TreeTCI state to a TreeTensorNetwork.

# Arguments
- `tci::SimpleTreeTci`: Converged state
- `f`: Batch evaluation function (same as `sweep!`)
- `center_site`: BFS root site for materialization (0-based)
"""
function to_treetn(tci::SimpleTreeTci{T}, f; center_site::Int = 0) where {T}
    f_ref = Ref{Any}(f)
    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve f_ref begin
        C_API.check_status(ccall(
            _sym_for(T, :to_treetn),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Ptr{Ptr{Cvoid}}),
            tci.ptr,
            _get_batch_trampoline(T),
            pointer_from_objref(f_ref),
            Csize_t(center_site),
            out_ptr,
        ))
    end
    return _wrap_treetn(out_ptr[], length(tci.local_dims))
end

# ============================================================================
# High-level convenience function
# ============================================================================

"""
    crossinterpolate2([T], f, localdims, graph; kwargs...) -> (tci, ranks, errors)

Run TreeTCI to convergence on a tree graph.

The sweep loop runs in Julia, printing convergence info when `verbosity > 0`.
This matches the API style of `TensorCrossInterpolation.crossinterpolate2`.

# Arguments
- `T`: Scalar type (`Float64` or `ComplexF64`). Inferred if omitted.
- `f`: Batch evaluation function — `f(batch) -> Vector{T}` where `batch` is a
  `Matrix{Csize_t}` of shape `(n_sites, n_points)` in column-major layout.
  `batch[i, j]` is the **0-based** local index at site `i` for evaluation point `j`.
  The function must return a `Vector` of length `n_points`.
- `localdims::Union{Vector{Int}, NTuple{N,Int}}`: Local dimensions at each site
- `graph::TreeTciGraph`: Tree graph structure

# Keyword Arguments
- `initialpivots::Vector{Vector{Int}} = [zeros(Int, n)]`: Initial pivots (0-based)
- `tolerance::Float64 = 1e-8`: Target tolerance
- `maxbonddim::Int = typemax(Int)`: Maximum bond dimension
- `maxiter::Int = 20`: Maximum sweeps
- `verbosity::Int = 0`: 0=silent, 1=summary per loginterval, 2=bond dims and timing
- `loginterval::Int = 10`: Print every N iterations (when verbosity >= 1)
- `normalizeerror::Bool = true`: Normalize error by max sample value
- `proposer::Symbol = :default`: `:default`, `:simple`, or `:truncated_default`
- `center_site::Int = 0`: Materialization center site (0-based)

# Returns
- `tci::SimpleTreeTci{T}`: The converged TCI state (call `to_treetn(tci, f)` to materialize)
- `ranks::Vector{Int}`: Max rank per iteration
- `errors::Vector{Float64}`: (Normalized) error per iteration

# Example
```julia
using Tensor4all.TreeTCI

# Define a star graph: site 0 connected to sites 1,2,3,4
graph = TreeTciGraph(5, [(0,1), (0,2), (0,3), (0,4)])

# Batch evaluation function (0-based indices)
function f(batch)
    n_sites, n_pts = size(batch)
    [prod(batch[i, j] + 1.0 for i in 1:n_sites) for j in 1:n_pts]
end

# Run TCI
tci, ranks, errors = crossinterpolate2(f, fill(3, 5), graph;
    tolerance=1e-10, verbosity=1)

# Materialize to TreeTensorNetwork
ttn = to_treetn(tci, f)
```
"""
function crossinterpolate2(
    ::Type{T},
    f,
    localdims::Union{Vector{<:Integer}, NTuple{N,Integer}},
    graph::TreeTciGraph;
    initialpivots::Vector{Vector{Int}} = [zeros(Int, graph.n_sites)],
    tolerance::Float64 = 1e-8,
    maxbonddim::Int = typemax(Int),
    maxiter::Int = 20,
    verbosity::Int = 0,
    loginterval::Int = 10,
    normalizeerror::Bool = true,
    proposer::Symbol = :default,
    center_site::Int = 0,
) where {T<:_TreeTciScalar, N}
    dims_int = Int.(collect(localdims))
    n_sites = length(dims_int)
    n_sites == graph.n_sites ||
        error("localdims length ($n_sites) != graph.n_sites ($(graph.n_sites))")

    bd = maxbonddim == typemax(Int) ? 0 : maxbonddim

    # Create state and add initial pivots
    tci = SimpleTreeTci{T}(dims_int, graph)
    add_global_pivots!(tci, initialpivots)

    ranks = Int[]
    errors = Float64[]
    t_start = time()

    # Sweep loop in Julia
    for iter in 1:maxiter
        t_sweep_start = time()
        sweep!(tci, f; proposer=proposer, tolerance=tolerance, max_bond_dim=bd)
        t_sweep = time() - t_sweep_start

        r = max_rank(tci)
        err = max_bond_error(tci)
        msv = max_sample_value(tci)
        normalized_err = (normalizeerror && msv > 0) ? err / msv : err

        push!(ranks, r)
        push!(errors, normalized_err)

        should_log = iter % loginterval == 0 || iter == 1 || normalized_err < tolerance
        if verbosity >= 1 && should_log
            @info "TreeTCI" iteration=iter rank=r error=normalized_err maxsamplevalue=msv
        end
        if verbosity >= 2 && should_log
            bd_vec = bond_dims(tci)
            elapsed = time() - t_start
            @info "TreeTCI detail" iteration=iter bonddims=bd_vec sweep_sec=round(t_sweep; digits=3) elapsed_sec=round(elapsed; digits=3)
        end

        if normalized_err < tolerance
            break
        end
    end

    return tci, ranks, errors
end

function crossinterpolate2(
    f,
    localdims::Union{Vector{<:Integer}, NTuple{N,Integer}},
    graph::TreeTciGraph;
    kwargs...,
) where {N}
    pivots = get(kwargs, :initialpivots, Vector{Int}[])
    T = _infer_scalar_type(f, collect(localdims), pivots)
    return crossinterpolate2(T, f, localdims, graph; kwargs...)
end

end # module TreeTCI
