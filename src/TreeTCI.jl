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
ttn, ranks, errors = crossinterpolate_tree(f, [3, 3, 3, 3], graph)
```
"""
module TreeTCI

using ..C_API
import ..TreeTN: TreeTensorNetwork

export TreeTciGraph, SimpleTreeTci
export crossinterpolate_tree
export add_global_pivots!, sweep!
export max_bond_error, max_rank, max_sample_value, bond_dims
export to_treetn

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
Internal trampoline for C batch callback.

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

const _BATCH_TRAMPOLINE_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _get_batch_trampoline()
    if _BATCH_TRAMPOLINE_PTR[] == C_NULL
        _BATCH_TRAMPOLINE_PTR[] = @cfunction(
            _treetci_batch_trampoline,
            Cint,
            (Ptr{Csize_t}, Csize_t, Csize_t, Ptr{Cdouble}, Ptr{Cvoid}),
        )
    end
    return _BATCH_TRAMPOLINE_PTR[]
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
    SimpleTreeTci(local_dims, graph)

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
mutable struct SimpleTreeTci
    ptr::Ptr{Cvoid}
    graph::TreeTciGraph
    local_dims::Vector{Int}

    function SimpleTreeTci(local_dims::Vector{Int}, graph::TreeTciGraph)
        length(local_dims) == graph.n_sites ||
            error("local_dims length ($(length(local_dims))) != graph.n_sites ($(graph.n_sites))")

        dims_csize = Csize_t.(local_dims)
        ptr = ccall(
            C_API._sym(:t4a_treetci_f64_new),
            Ptr{Cvoid},
            (Ptr{Csize_t}, Csize_t, Ptr{Cvoid}),
            dims_csize,
            Csize_t(length(dims_csize)),
            graph.ptr,
        )
        if ptr == C_NULL
            error("Failed to create SimpleTreeTci: $(C_API.last_error_message())")
        end

        tci = new(ptr, graph, copy(local_dims))
        finalizer(tci) do obj
            if obj.ptr != C_NULL
                ccall(C_API._sym(:t4a_treetci_f64_release), Cvoid, (Ptr{Cvoid},), obj.ptr)
                obj.ptr = C_NULL
            end
        end
        return tci
    end
end

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
function add_global_pivots!(tci::SimpleTreeTci, pivots::Vector{Vector{Int}})
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
        C_API._sym(:t4a_treetci_f64_add_global_pivots),
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
- `f`: Batch evaluation function `f(batch::Matrix{Csize_t}) -> Vector{Float64}`
  where `batch` is column-major `(n_sites, n_points)` with 0-based indices
- `proposer`: `:default`, `:simple`, or `:truncated_default`
- `tolerance`: Relative tolerance
- `max_bond_dim`: Maximum bond dimension (0 = unlimited)
"""
function sweep!(
    tci::SimpleTreeTci,
    f;
    proposer::Symbol = :default,
    tolerance::Float64 = 1e-8,
    max_bond_dim::Int = 0,
)
    f_ref = Ref{Any}(f)
    GC.@preserve f_ref begin
        C_API.check_status(ccall(
            C_API._sym(:t4a_treetci_f64_sweep),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cint, Cdouble, Csize_t),
            tci.ptr,
            _get_batch_trampoline(),
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
function max_bond_error(tci::SimpleTreeTci)::Float64
    out = Ref{Cdouble}(0.0)
    C_API.check_status(ccall(
        C_API._sym(:t4a_treetci_f64_max_bond_error),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        tci.ptr,
        out,
    ))
    return out[]
end

"""Maximum rank (bond dimension) across all edges."""
function max_rank(tci::SimpleTreeTci)::Int
    out = Ref{Csize_t}(0)
    C_API.check_status(ccall(
        C_API._sym(:t4a_treetci_f64_max_rank),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        tci.ptr,
        out,
    ))
    return Int(out[])
end

"""Maximum observed sample value (for normalization)."""
function max_sample_value(tci::SimpleTreeTci)::Float64
    out = Ref{Cdouble}(0.0)
    C_API.check_status(ccall(
        C_API._sym(:t4a_treetci_f64_max_sample_value),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        tci.ptr,
        out,
    ))
    return out[]
end

"""Bond dimensions (ranks) at each edge."""
function bond_dims(tci::SimpleTreeTci)::Vector{Int}
    n_edges_ref = Ref{Csize_t}(0)
    C_API.check_status(ccall(
        C_API._sym(:t4a_treetci_f64_bond_dims),
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
        C_API._sym(:t4a_treetci_f64_bond_dims),
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
function to_treetn(tci::SimpleTreeTci, f; center_site::Int = 0)
    f_ref = Ref{Any}(f)
    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve f_ref begin
        C_API.check_status(ccall(
            C_API._sym(:t4a_treetci_f64_to_treetn),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Ptr{Ptr{Cvoid}}),
            tci.ptr,
            _get_batch_trampoline(),
            pointer_from_objref(f_ref),
            Csize_t(center_site),
            out_ptr,
        ))
    end
    return _wrap_treetn(out_ptr[], length(tci.local_dims))
end

end # module TreeTCI
