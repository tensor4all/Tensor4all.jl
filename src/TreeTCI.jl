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

end # module TreeTCI
