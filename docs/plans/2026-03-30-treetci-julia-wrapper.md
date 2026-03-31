# TreeTCI Julia Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Julia ラッパーモジュール `TreeTCI` を Tensor4all.jl に追加し、tensor4all-rs の TreeTCI C API を呼び出せるようにする。

**Architecture:** `src/TreeTCI.jl` サブモジュールとして既存パターン（TensorCI, SimpleTT）に準拠。バッチコールバック trampoline で Julia 関数を C API に渡す。出力は既存の `TreeTN.TreeTensorNetwork` 型。

**Tech Stack:** Julia, ccall, Tensor4all.jl C_API module, tensor4all-capi TreeTCI functions

**Prerequisite:** tensor4all-rs の `feat/treetci-capi` ブランチがビルドされ、`libtensor4all_capi.so` が `deps/` に配置されていること。

**Working directory:** `/home/shinaoka/tensor4all/Tensor4all.jl`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/TreeTCI.jl` | TreeTCI モジュール全体 |
| Modify | `src/Tensor4all.jl` | `include("TreeTCI.jl")` 追加 |
| Create | `test/test_treetci.jl` | TreeTCI テスト |
| Modify | `test/runtests.jl` | テストファイル include 追加 |

---

## Task 1: Scaffold — モジュール骨格とメインモジュール統合

**Files:**
- Create: `src/TreeTCI.jl`
- Modify: `src/Tensor4all.jl`

- [ ] **Step 1: Create empty TreeTCI module**

Create `src/TreeTCI.jl`:

```julia
"""
    TreeTCI

Tree-structured tensor cross interpolation via tensor4all-rs.

Provides `TreeTciGraph` for defining tree topologies and `SimpleTreeTci`
for running TCI on arbitrary tree structures. Results are returned as
`TreeTN.TreeTensorNetwork`.

# Usage
```julia
using Tensor4all.TreeTCI

graph = TreeTciGraph(4, [(0,1), (1,2), (2,3)])
f(batch) = [sum(Float64, batch[:, j]) for j in 1:size(batch, 2)]
ttn, ranks, errors = crossinterpolate_tree(f, [3, 3, 3, 3], graph)
```
"""
module TreeTCI

using ..C_API
import ..TreeTN: TreeTensorNetwork

export TreeTciGraph, SimpleTreeTci
export crossinterpolate_tree

end # module TreeTCI
```

- [ ] **Step 2: Add include to Tensor4all.jl**

In `src/Tensor4all.jl`, add after the `include("QuanticsTransform.jl")` line (line 938):

```julia
# ============================================================================
# Tree-structured TCI (tree tensor cross interpolation).
# Use: using Tensor4all.TreeTCI
include("TreeTCI.jl")
```

- [ ] **Step 3: Verify it loads**

Run: `julia --startup-file=no -e 'using Pkg; Pkg.activate("."); using Tensor4all; using Tensor4all.TreeTCI; println("OK")'`

Expected: `OK` (module loads without error)

- [ ] **Step 4: Commit**

```bash
git add src/TreeTCI.jl src/Tensor4all.jl
git commit -m "feat: scaffold TreeTCI module"
```

---

## Task 2: TreeTciGraph type

**Files:**
- Modify: `src/TreeTCI.jl`
- Create: `test/test_treetci.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the failing test**

Create `test/test_treetci.jl`:

```julia
using Tensor4all.TreeTCI
using Tensor4all.TreeTN: TreeTensorNetwork
using Test

@testset "TreeTCI" begin
    @testset "TreeTciGraph" begin
        # Linear chain: 0-1-2-3
        graph = TreeTciGraph(4, [(0, 1), (1, 2), (2, 3)])
        @test graph.n_sites == 4
        @test graph.ptr != C_NULL

        # Star graph: 0 at center
        graph_star = TreeTciGraph(4, [(0, 1), (0, 2), (0, 3)])
        @test graph_star.n_sites == 4

        # Invalid: disconnected
        @test_throws ErrorException TreeTciGraph(4, [(0, 1), (2, 3)])
    end
end
```

Add to `test/runtests.jl`, inside the `@testset "Tensor4all.jl"` block:

```julia
    include("test_treetci.jl")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --startup-file=no -e 'using Pkg; Pkg.activate("."); Pkg.test()' 2>&1 | tail -20`

Expected: FAIL — `TreeTciGraph` not defined.

- [ ] **Step 3: Implement TreeTciGraph**

Add to `src/TreeTCI.jl` (inside the module, before `end`):

```julia
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
graph = TreeTciGraph(7, [(0,1), (1,2), (1,3), (3,4), (4,5), (4,6)])
```
"""
mutable struct TreeTciGraph
    ptr::Ptr{Cvoid}
    n_sites::Int

    function TreeTciGraph(n_sites::Int, edges::Vector{Tuple{Int,Int}})
        n_edges = length(edges)
        edges_flat = Vector{Csize_t}(undef, n_edges * 2)
        for (i, (u, v)) in enumerate(edges)
            edges_flat[2i - 1] = Csize_t(u)
            edges_flat[2i] = Csize_t(v)
        end
        ptr = ccall(
            C_API._sym(:t4a_treetci_graph_new),
            Ptr{Cvoid},
            (Csize_t, Ptr{Csize_t}, Csize_t),
            Csize_t(n_sites), edges_flat, Csize_t(n_edges),
        )
        if ptr == C_NULL
            error("Failed to create TreeTciGraph: $(C_API.last_error_message())")
        end
        obj = new(ptr, n_sites)
        finalizer(obj) do x
            if x.ptr != C_NULL
                ccall(C_API._sym(:t4a_treetci_graph_release), Cvoid, (Ptr{Cvoid},), x.ptr)
                x.ptr = C_NULL
            end
        end
        obj
    end
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --startup-file=no -e 'using Pkg; Pkg.activate("."); include("test/test_treetci.jl")'`

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/TreeTCI.jl test/test_treetci.jl test/runtests.jl
git commit -m "feat: add TreeTciGraph type"
```

---

## Task 3: Batch callback trampoline and proposer helpers

**Files:**
- Modify: `src/TreeTCI.jl`

- [ ] **Step 1: Add trampoline and helpers**

Add to `src/TreeTCI.jl` (after TreeTciGraph, before `end`):

```julia
# ============================================================================
# Batch Eval Trampoline
# ============================================================================

"""
Internal trampoline for C batch callback.

The user function signature is: `f(batch::Matrix{Csize_t}) -> Vector{Float64}`
where `batch` is column-major (n_sites, n_points) with 0-based indices.
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
        for i in 1:Int(n_points)
            unsafe_store!(results, Float64(vals[i]), i)
        end
        return Cint(0)
    catch e
        @error "TreeTCI batch eval callback error" exception = (e, catch_backtrace())
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
    _BATCH_TRAMPOLINE_PTR[]
end

# ============================================================================
# Proposer helpers
# ============================================================================

const _PROPOSER_DEFAULT = Cint(0)
const _PROPOSER_SIMPLE = Cint(1)
const _PROPOSER_TRUNCATED_DEFAULT = Cint(2)

function _proposer_to_cint(proposer::Symbol)::Cint
    if proposer === :default
        _PROPOSER_DEFAULT
    elseif proposer === :simple
        _PROPOSER_SIMPLE
    elseif proposer === :truncated_default
        _PROPOSER_TRUNCATED_DEFAULT
    else
        error("Unknown proposer: $proposer. Use :default, :simple, or :truncated_default")
    end
end
```

- [ ] **Step 2: Verify it compiles**

Run: `julia --startup-file=no -e 'using Pkg; Pkg.activate("."); using Tensor4all.TreeTCI; println("OK")'`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/TreeTCI.jl
git commit -m "feat: add TreeTCI batch trampoline and proposer helpers"
```

---

## Task 4: SimpleTreeTci — state, pivots, sweep, inspection

**Files:**
- Modify: `src/TreeTCI.jl`
- Modify: `test/test_treetci.jl`

- [ ] **Step 1: Write the failing test**

Add to `test/test_treetci.jl`, inside the `@testset "TreeTCI"` block:

```julia
    @testset "Stateful API" begin
        n_sites = 7
        local_dims = fill(2, n_sites)
        edges = [(0, i) for i in 1:6]  # star graph
        graph = TreeTciGraph(n_sites, edges)

        # Product function: f(idx) = prod(idx[s] + 1.0)
        function f_batch(batch)
            n_pts = size(batch, 2)
            results = Vector{Float64}(undef, n_pts)
            for j in 1:n_pts
                val = 1.0
                for i in 1:size(batch, 1)
                    val *= (batch[i, j] + 1.0)
                end
                results[j] = val
            end
            results
        end

        tci = SimpleTreeTci(local_dims, graph)
        add_global_pivots!(tci, [zeros(Int, n_sites)])

        for _ in 1:4
            sweep!(tci, f_batch; tolerance=1e-12)
        end

        @test max_bond_error(tci) < 1e-10
        @test max_rank(tci) >= 1
        @test max_sample_value(tci) > 0.0

        bd = bond_dims(tci)
        @test length(bd) == 6  # n_edges
        @test all(d -> d >= 1, bd)

        ttn = to_treetn(tci, f_batch)
        @test ttn isa TreeTensorNetwork
    end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --startup-file=no -e 'using Pkg; Pkg.activate("."); include("test/test_treetci.jl")' 2>&1 | tail -10`

Expected: FAIL — `SimpleTreeTci` not defined.

- [ ] **Step 3: Implement SimpleTreeTci and all methods**

Add to `src/TreeTCI.jl` (after proposer helpers, before `end`):

```julia
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
for i in 1:max_iter
    sweep!(tci, f; tolerance=1e-8)
    max_bond_error(tci) < 1e-8 && break
end
ttn = to_treetn(tci, f)
```
"""
mutable struct SimpleTreeTci
    ptr::Ptr{Cvoid}
    graph::TreeTciGraph      # prevent GC
    local_dims::Vector{Int}

    function SimpleTreeTci(local_dims::Vector{Int}, graph::TreeTciGraph)
        length(local_dims) == graph.n_sites ||
            error("local_dims length ($(length(local_dims))) != graph.n_sites ($(graph.n_sites))")
        dims_csize = Csize_t.(local_dims)
        ptr = ccall(
            C_API._sym(:t4a_treetci_f64_new),
            Ptr{Cvoid},
            (Ptr{Csize_t}, Csize_t, Ptr{Cvoid}),
            dims_csize, Csize_t(length(dims_csize)), graph.ptr,
        )
        if ptr == C_NULL
            error("Failed to create SimpleTreeTci: $(C_API.last_error_message())")
        end
        obj = new(ptr, graph, local_dims)
        finalizer(obj) do x
            if x.ptr != C_NULL
                ccall(C_API._sym(:t4a_treetci_f64_release), Cvoid, (Ptr{Cvoid},), x.ptr)
                x.ptr = C_NULL
            end
        end
        obj
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
- `pivots::Vector{Vector{Int}}`: Each element has length n_sites, 0-based indices
"""
function add_global_pivots!(tci::SimpleTreeTci, pivots::Vector{Vector{Int}})
    n_sites = length(tci.local_dims)
    n_pivots = length(pivots)
    n_pivots == 0 && return
    pivots_flat = Vector{Csize_t}(undef, n_sites * n_pivots)
    for j in 1:n_pivots
        length(pivots[j]) == n_sites ||
            error("Pivot $j has length $(length(pivots[j])), expected $n_sites")
        for i in 1:n_sites
            pivots_flat[i + n_sites * (j - 1)] = Csize_t(pivots[j][i])
        end
    end
    C_API.check_status(ccall(
        C_API._sym(:t4a_treetci_f64_add_global_pivots),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Csize_t),
        tci.ptr, pivots_flat, Csize_t(n_sites), Csize_t(n_pivots),
    ))
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
  - `batch` is column-major (n_sites, n_points), 0-based indices
- `proposer`: `:default`, `:simple`, or `:truncated_default`
- `tolerance`: Relative tolerance
- `max_bond_dim`: Maximum bond dimension (0 = unlimited)
"""
function sweep!(tci::SimpleTreeTci, f;
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
end

# ============================================================================
# State inspection
# ============================================================================

"""Maximum bond error across all edges."""
function max_bond_error(tci::SimpleTreeTci)::Float64
    out = Ref{Cdouble}(0.0)
    C_API.check_status(ccall(
        C_API._sym(:t4a_treetci_f64_max_bond_error),
        Cint, (Ptr{Cvoid}, Ptr{Cdouble}),
        tci.ptr, out,
    ))
    out[]
end

"""Maximum rank (bond dimension) across all edges."""
function max_rank(tci::SimpleTreeTci)::Int
    out = Ref{Csize_t}(0)
    C_API.check_status(ccall(
        C_API._sym(:t4a_treetci_f64_max_rank),
        Cint, (Ptr{Cvoid}, Ptr{Csize_t}),
        tci.ptr, out,
    ))
    Int(out[])
end

"""Maximum observed sample value (for normalization)."""
function max_sample_value(tci::SimpleTreeTci)::Float64
    out = Ref{Cdouble}(0.0)
    C_API.check_status(ccall(
        C_API._sym(:t4a_treetci_f64_max_sample_value),
        Cint, (Ptr{Cvoid}, Ptr{Cdouble}),
        tci.ptr, out,
    ))
    out[]
end

"""Bond dimensions (ranks) at each edge."""
function bond_dims(tci::SimpleTreeTci)::Vector{Int}
    n_edges_ref = Ref{Csize_t}(0)
    C_API.check_status(ccall(
        C_API._sym(:t4a_treetci_f64_bond_dims),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
        tci.ptr, C_NULL, Csize_t(0), n_edges_ref,
    ))
    n_edges = Int(n_edges_ref[])
    buf = Vector{Csize_t}(undef, n_edges)
    C_API.check_status(ccall(
        C_API._sym(:t4a_treetci_f64_bond_dims),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
        tci.ptr, buf, Csize_t(n_edges), n_edges_ref,
    ))
    Int.(buf)
end

# ============================================================================
# Materialization
# ============================================================================

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
    TreeTensorNetwork(out_ptr[])
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --startup-file=no -e 'using Pkg; Pkg.activate("."); include("test/test_treetci.jl")'`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/TreeTCI.jl test/test_treetci.jl
git commit -m "feat: add SimpleTreeTci with sweep, inspection, and materialization"
```

---

## Task 5: High-level convenience function

**Files:**
- Modify: `src/TreeTCI.jl`
- Modify: `test/test_treetci.jl`

- [ ] **Step 1: Write the failing test**

Add to `test/test_treetci.jl`, inside the `@testset "TreeTCI"` block:

```julia
    @testset "High-level API" begin
        n_sites = 4
        local_dims = fill(3, n_sites)
        graph = TreeTciGraph(n_sites, [(0, 1), (1, 2), (2, 3)])

        f_batch(batch) = [sum(Float64, batch[:, j]) for j in 1:size(batch, 2)]

        ttn, ranks, errors = crossinterpolate_tree(
            f_batch, local_dims, graph;
            initial_pivots = [zeros(Int, n_sites)],
            tolerance = 1e-10,
            max_iter = 20,
        )

        @test ttn isa TreeTensorNetwork
        @test length(ranks) > 0
        @test last(errors) < 1e-8
    end

    @testset "High-level API without initial pivots" begin
        # Tests the default zero-pivot behavior
        n_sites = 3
        local_dims = fill(2, n_sites)
        graph = TreeTciGraph(n_sites, [(0, 1), (1, 2)])

        f_batch(batch) = [prod(batch[i, j] + 1.0 for i in 1:size(batch, 1)) for j in 1:size(batch, 2)]

        ttn, ranks, errors = crossinterpolate_tree(f_batch, local_dims, graph)

        @test ttn isa TreeTensorNetwork
        @test length(ranks) > 0
    end
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `crossinterpolate_tree` not defined.

- [ ] **Step 3: Implement crossinterpolate_tree**

Add to `src/TreeTCI.jl` (after `to_treetn`, before `end # module`):

```julia
# ============================================================================
# High-level convenience function
# ============================================================================

"""
    crossinterpolate_tree(f, local_dims, graph; kwargs...) -> (ttn, ranks, errors)

Run TreeTCI to convergence and return a TreeTensorNetwork.

# Arguments
- `f`: Batch evaluation function `f(batch::Matrix{Csize_t}) -> Vector{Float64}`
- `local_dims::Vector{Int}`: Local dimension at each site
- `graph::TreeTciGraph`: Tree graph structure

# Keyword Arguments
- `initial_pivots::Vector{Vector{Int}} = []`: Initial pivots (0-based). If empty, defaults to zero pivot.
- `proposer::Symbol = :default`: `:default`, `:simple`, or `:truncated_default`
- `tolerance::Float64 = 1e-8`: Relative tolerance
- `max_bond_dim::Int = 0`: Maximum bond dimension (0 = unlimited)
- `max_iter::Int = 20`: Maximum iterations
- `normalize_error::Bool = true`: Normalize errors by max sample value
- `center_site::Int = 0`: Materialization center site (0-based)

# Returns
- `ttn::TreeTensorNetwork`
- `ranks::Vector{Int}`: Max rank per iteration
- `errors::Vector{Float64}`: Normalized error per iteration
"""
function crossinterpolate_tree(
    f, local_dims::Vector{Int}, graph::TreeTciGraph;
    initial_pivots::Vector{Vector{Int}} = Vector{Int}[],
    proposer::Symbol = :default,
    tolerance::Float64 = 1e-8,
    max_bond_dim::Int = 0,
    max_iter::Int = 20,
    normalize_error::Bool = true,
    center_site::Int = 0,
)
    n_sites = length(local_dims)
    n_pivots = length(initial_pivots)

    # Pack initial pivots column-major (n_sites, n_pivots)
    pivots_flat = if n_pivots > 0
        buf = Vector{Csize_t}(undef, n_sites * n_pivots)
        for j in 1:n_pivots
            length(initial_pivots[j]) == n_sites ||
                error("Pivot $j has length $(length(initial_pivots[j])), expected $n_sites")
            for i in 1:n_sites
                buf[i + n_sites * (j - 1)] = Csize_t(initial_pivots[j][i])
            end
        end
        buf
    else
        Csize_t[]
    end

    # Output buffers (pre-allocate max_iter)
    out_ranks = Vector{Csize_t}(undef, max_iter)
    out_errors = Vector{Cdouble}(undef, max_iter)
    out_n_iters = Ref{Csize_t}(0)
    out_treetn = Ref{Ptr{Cvoid}}(C_NULL)

    dims_csize = Csize_t.(local_dims)
    f_ref = Ref{Any}(f)

    GC.@preserve f_ref pivots_flat out_ranks out_errors dims_csize begin
        C_API.check_status(ccall(
            C_API._sym(:t4a_crossinterpolate_tree_f64),
            Cint,
            (
                Ptr{Cvoid}, Ptr{Cvoid},            # eval_cb, user_data
                Ptr{Csize_t}, Csize_t,             # local_dims, n_sites
                Ptr{Cvoid},                         # graph
                Ptr{Csize_t}, Csize_t,             # initial_pivots, n_pivots
                Cint,                               # proposer_kind
                Cdouble, Csize_t, Csize_t,         # tol, max_bond_dim, max_iter
                Cint,                               # normalize_error
                Csize_t,                            # center_site
                Ptr{Ptr{Cvoid}},                    # out_treetn
                Ptr{Csize_t}, Ptr{Cdouble}, Ptr{Csize_t}, # out_ranks, errors, n_iters
            ),
            _get_batch_trampoline(),
            pointer_from_objref(f_ref),
            dims_csize, Csize_t(n_sites),
            graph.ptr,
            n_pivots > 0 ? pivots_flat : C_NULL, Csize_t(n_pivots),
            _proposer_to_cint(proposer),
            tolerance, Csize_t(max_bond_dim), Csize_t(max_iter),
            normalize_error ? Cint(1) : Cint(0),
            Csize_t(center_site),
            out_treetn,
            out_ranks, out_errors, out_n_iters,
        ))
    end

    n_iters = Int(out_n_iters[])
    ttn = TreeTensorNetwork(out_treetn[])
    ranks = Int.(out_ranks[1:n_iters])
    errors = Float64.(out_errors[1:n_iters])
    return (ttn, ranks, errors)
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --startup-file=no -e 'using Pkg; Pkg.activate("."); include("test/test_treetci.jl")'`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/TreeTCI.jl test/test_treetci.jl
git commit -m "feat: add crossinterpolate_tree high-level API"
```

---

## Task 6: Final validation

**Files:** None (validation only)

- [ ] **Step 1: Ensure tensor4all-rs is on the correct branch**

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs && git checkout feat/treetci-capi
```

- [ ] **Step 2: Rebuild the Rust library via Pkg.build**

`deps/build.jl` が自動的に sibling の `../tensor4all-rs/` を検出して RustToolChain.jl 経由でビルドする。

```bash
cd /home/shinaoka/tensor4all/Tensor4all.jl
julia --startup-file=no -e 'using Pkg; Pkg.activate("."); Pkg.build()'
```

Expected: `Build complete! Library installed to: .../deps/libtensor4all_capi.so`

- [ ] **Step 3: Run full test suite**

```bash
julia --startup-file=no -e 'using Pkg; Pkg.activate("."); Pkg.test()'
```

Expected: All tests pass, including new TreeTCI tests. No regressions.
