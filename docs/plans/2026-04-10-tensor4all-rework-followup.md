# Tensor4all Skeleton Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `Tensor4all.jl` from the phase-0 reset into a reviewable, TreeTN-general API skeleton aligned with `docs/design/`, while keeping backend loading lazy, keeping high-level `TTFunction` logic out of this package, and avoiding any fake numerics.

**Architecture:** Build the package bottom-up. `Core` owns common errors, lazy backend loading, `Index`, and `Tensor`; `TreeTN` adds general tensor-network wrappers plus chain aliases and runtime topology predicates; `Quantics` adds grid, layout, transform, and QTCI placeholder types; `ext/` owns ITensors and HDF5 conversion stubs; docs and smoke tests grow in lockstep so every public symbol is reviewable before real backend behavior is enabled.

**Tech Stack:** Julia 1.9+, Documenter.jl, package extensions, `tensor4all-rs` C API, `ITensors.jl`, `HDF5.jl`, `Test` stdlib.

---

## Design Inputs

This plan implements the architecture described in:

- `docs/design/julia_ffi.md`
- `docs/design/julia_ffi_core.md`
- `docs/design/julia_ffi_tt.md`
- `docs/design/julia_ffi_quantics.md`
- `docs/design/julia_ffi_extensions.md`
- `docs/design/julia_ffi_roadmap.md`

This plan must remain consistent with `AGENTS.md`, especially:

- the Julia API is TreeTN-general, not chain-only
- `TensorTrain = TreeTensorNetwork{Int}` is the primary chain alias
- `MPS` and `MPO` are aliases or runtime conventions, not separate Julia types
- Julia-side composition should be preferred over adding C API calls unless the primitive is genuinely multi-language useful
- `TTFunction` stays in `BubbleTeaCI`, not in this package

## Non-Goals

- no real contractions, decompositions, or backend numerics beyond lazy loading hooks
- no eager `dlopen` during `using Tensor4all`
- no restored pre-reset module tree
- no `TTFunction`, `GriddedFunction`, or application-level QTT workflows inside this repo
- no pretending that a stubbed API is numerically complete

## Planned File Structure

### Source Files

- `src/Tensor4all.jl`
  - top-level module wiring, exports, and includes only
- `src/Core/Errors.jl`
  - `SkeletonPhaseError`, `SkeletonNotImplemented`, `BackendUnavailableError`
- `src/Core/Backend.jl`
  - lazy backend path resolution and `require_backend()`
- `src/Core/Index.jl`
  - `Index` metadata type and Julia-side helpers
- `src/Core/Tensor.jl`
  - `Tensor` metadata type, dense-array constructor checks, and stubbed contraction
- `src/TreeTN/TreeTensorNetwork.jl`
  - `TreeTensorNetwork{V}`, `TensorTrain`, `MPS`, `MPO`, topology predicates, runtime checks
- `src/Quantics/Grids.jl`
  - variable, layout, and grid types plus coordinate conversion helpers
- `src/Quantics/Transforms.jl`
  - transform descriptor types and constructor stubs
- `src/Quantics/QTCI.jl`
  - `QTCIOptions`, diagnostics, and placeholder result containers

### Extension Files

- `ext/Tensor4allITensorsExt.jl`
  - extension-only compatibility stubs for `ITensors.jl`
- `ext/Tensor4allHDF5Ext.jl`
  - extension-only compatibility stubs for `HDF5.jl`

### Test Files

- `test/runtests.jl`
  - test entrypoint, includes each layer test file
- `test/core/bootstrap.jl`
  - error types, lazy backend helper, import smoke tests
- `test/core/index.jl`
  - `Index` construction and metadata helper behavior
- `test/core/tensor.jl`
  - `Tensor` construction, metadata access, and stub behavior
- `test/ttn/tree_tensor_network.jl`
  - `TreeTensorNetwork` aliases, topology predicates, runtime checks
- `test/quantics/grids.jl`
  - quantics metadata and coordinate conversion
- `test/quantics/transforms.jl`
  - transform constructor metadata and stub behavior
- `test/extensions/itensors_ext.jl`
  - extension load check and conversion stub errors
- `test/extensions/hdf5_ext.jl`
  - extension load check and save/load stub errors

### Documentation Files

- `docs/src/index.md`
  - review-first landing page plus package status banner
- `docs/src/modules.md`
  - module overview and dependency graph for the skeleton
- `docs/src/api.md`
  - auto-generated API reference for the skeleton surface
- `docs/src/design_documents.md`
  - links to design docs and execution plan
- `docs/src/deferred_rework_plan.md`
  - short page that points to this implementation plan
- `docs/make.jl`
  - Documenter page list and API reference inclusion
- `README.md`
  - package status, design links, and skeleton-phase expectations

## Review Gates

- After Task 2: confirm `Index` naming and metadata semantics
- After Task 4: confirm `TreeTensorNetwork`, `TensorTrain`, `MPS`, and `MPO` surface
- After Task 5: confirm quantics type names and grid-layout conventions
- After Task 6: confirm extension boundary and weak dependency wiring
- After Task 7: confirm docs site accurately reflects implemented versus stubbed behavior

---

### Task 1: Rebuild the Package Scaffold Around Core Layers

**Files:**
- Create: `src/Core/Errors.jl`
- Create: `src/Core/Backend.jl`
- Modify: `src/Tensor4all.jl`
- Create: `test/core/bootstrap.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the failing bootstrap tests**

```julia
# test/core/bootstrap.jl
using Test
using Tensor4all

@testset "bootstrap errors and lazy backend helpers" begin
    @test Tensor4all.SKELETON_PHASE === true
    @test isdefined(Tensor4all, :SkeletonPhaseError)
    @test isdefined(Tensor4all, :SkeletonNotImplemented)
    @test isdefined(Tensor4all, :BackendUnavailableError)
    @test isdefined(Tensor4all, :backend_library_path)
    @test isdefined(Tensor4all, :require_backend)

    placeholder = Tensor4all.SkeletonNotImplemented(:contract, :core)
    @test sprint(showerror, placeholder) ==
        "Tensor4all skeleton phase: `contract` is planned in the `core` layer but not implemented yet."

    missing = Tensor4all.BackendUnavailableError("backend missing")
    @test sprint(showerror, missing) == "backend missing"
    @test Tensor4all.backend_library_path() isa String
end
```

```julia
# test/runtests.jl
using Test
using Tensor4all

include("core/bootstrap.jl")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: FAIL because `SkeletonNotImplemented`, `BackendUnavailableError`, `backend_library_path`, and `require_backend` do not exist yet.

- [ ] **Step 3: Implement the scaffold**

```julia
# src/Core/Errors.jl
"""
    SkeletonPhaseError(message)

Raised when code expects functionality that is intentionally deferred during the
skeleton-review phase.
"""
struct SkeletonPhaseError <: Exception
    message::String
end

Base.showerror(io::IO, err::SkeletonPhaseError) = print(io, err.message)

"""
    SkeletonNotImplemented(api, layer)

Raised by public APIs that are intentionally present for review but whose
backend behavior has not been implemented yet.
"""
struct SkeletonNotImplemented <: Exception
    api::Symbol
    layer::Symbol
end

Base.showerror(io::IO, err::SkeletonNotImplemented) = print(
    io,
    "Tensor4all skeleton phase: `",
    err.api,
    "` is planned in the `",
    err.layer,
    "` layer but not implemented yet.",
)

"""
    BackendUnavailableError(message)

Raised when a backend-backed operation is requested but the `tensor4all-rs`
shared library is not available.
"""
struct BackendUnavailableError <: Exception
    message::String
end

Base.showerror(io::IO, err::BackendUnavailableError) = print(io, err.message)
```

```julia
# src/Core/Backend.jl
const _backend_handle = Ref{Ptr{Cvoid}}(C_NULL)

backend_library_name() = "libtensor4all_capi." * Libdl.dlext

function backend_library_path()
    return get(
        ENV,
        "TENSOR4ALL_CAPI_PATH",
        normpath(joinpath(@__DIR__, "..", "..", "deps", backend_library_name())),
    )
end

function require_backend()
    path = backend_library_path()
    isfile(path) || throw(BackendUnavailableError(
        "tensor4all-rs backend unavailable at `$path`. Run `julia --startup-file=no --project=. deps/build.jl` or set `TENSOR4ALL_CAPI_PATH`.",
    ))
    if _backend_handle[] == C_NULL
        _backend_handle[] = Libdl.dlopen(path)
    end
    return _backend_handle[]
end
```

```julia
# src/Tensor4all.jl
"""
    Tensor4all

`Tensor4all.jl` is in an API-skeleton review phase.

The package is being rebuilt around the design documents in `docs/design/`.
Importing the package should succeed without a compiled Rust backend. Public
symbols may exist before their backend behavior is implemented; such calls
raise review-friendly stub exceptions rather than silently faking numerics.
"""
module Tensor4all

using Libdl

const SKELETON_PHASE = true

include("Core/Errors.jl")
include("Core/Backend.jl")

export SKELETON_PHASE
export SkeletonPhaseError, SkeletonNotImplemented, BackendUnavailableError
export backend_library_path, require_backend

end
```

- [ ] **Step 4: Run tests to verify the scaffold passes**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: PASS with a single bootstrap testset.

- [ ] **Step 5: Commit**

```bash
git add src/Tensor4all.jl src/Core/Errors.jl src/Core/Backend.jl test/runtests.jl test/core/bootstrap.jl
git commit -m "feat: add skeleton error and backend scaffold"
```

---

### Task 2: Add the `Index` Metadata Skeleton

**Files:**
- Create: `src/Core/Index.jl`
- Modify: `src/Tensor4all.jl`
- Create: `test/core/index.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the failing `Index` tests**

```julia
# test/core/index.jl
using Test
using Tensor4all

@testset "Index skeleton" begin
    i = Tensor4all.Index(4; tags=["x", "site"], plev=1)
    j = Tensor4all.sim(i)

    @test Tensor4all.dim(i) == 4
    @test Tensor4all.tags(i) == ["x", "site"]
    @test Tensor4all.plev(i) == 1
    @test Tensor4all.hastag(i, "x")
    @test Tensor4all.id(i) != Tensor4all.id(j)
    @test Tensor4all.dim(j) == Tensor4all.dim(i)

    ip = Tensor4all.prime(i, 2)
    @test Tensor4all.plev(ip) == 3
    @test Tensor4all.id(ip) == Tensor4all.id(i)
    @test Tensor4all.plev(Tensor4all.noprime(ip)) == 0
    @test Tensor4all.plev(Tensor4all.setprime(i, 7)) == 7

    xs = [i, j, ip]
    ys = [j, ip]
    @test Tensor4all.commoninds(xs, ys) == [j, ip]
    @test Tensor4all.uniqueinds(xs, ys) == [i]
end
```

```julia
# test/runtests.jl
using Test
using Tensor4all

include("core/bootstrap.jl")
include("core/index.jl")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: FAIL because `Index`, `dim`, `id`, `tags`, `plev`, `sim`, `prime`, and related helpers do not exist yet.

- [ ] **Step 3: Implement `Index`**

```julia
# src/Core/Index.jl
struct Index
    dim::Int
    id::UInt64
    tags::Vector{String}
    plev::Int
end

const _next_index_id = Ref{UInt64}(0)
next_index_id() = (_next_index_id[] += 1)

function Index(dim::Integer; tags=String[], plev::Integer=0, id::Integer=next_index_id())
    dim > 0 || throw(ArgumentError("Index dimension must be positive, got $dim"))
    plev >= 0 || throw(ArgumentError("Index prime level must be nonnegative, got $plev"))
    return Index(Int(dim), UInt64(id), collect(String.(tags)), Int(plev))
end

dim(i::Index) = i.dim
id(i::Index) = i.id
tags(i::Index) = copy(i.tags)
plev(i::Index) = i.plev
hastag(i::Index, tag::AbstractString) = String(tag) in i.tags

sim(i::Index) = Index(dim(i); tags=tags(i), plev=plev(i))
prime(i::Index, n::Integer=1) = Index(dim(i); tags=tags(i), plev=plev(i) + Int(n), id=id(i))
noprime(i::Index) = Index(dim(i); tags=tags(i), plev=0, id=id(i))
setprime(i::Index, n::Integer) = Index(dim(i); tags=tags(i), plev=Int(n), id=id(i))

Base.:(==)(a::Index, b::Index) =
    dim(a) == dim(b) && id(a) == id(b) && plev(a) == plev(b) && tags(a) == tags(b)

Base.hash(i::Index, h::UInt) = hash((dim(i), id(i), plev(i), tags(i)), h)

replaceind(xs::AbstractVector{Index}, old::Index, new::Index) =
    [x == old ? new : x for x in xs]

function replaceinds(xs::AbstractVector{Index}, replacements::Pair{Index,Index}...)
    ys = collect(xs)
    for (old, new) in replacements
        ys = replaceind(ys, old, new)
    end
    return ys
end

commoninds(xs::AbstractVector{Index}, ys::AbstractVector{Index}) = [x for x in xs if x in ys]
uniqueinds(xs::AbstractVector{Index}, ys::AbstractVector{Index}) = [x for x in xs if x ∉ ys]
```

```julia
# src/Tensor4all.jl
module Tensor4all

using Libdl

const SKELETON_PHASE = true

include("Core/Errors.jl")
include("Core/Backend.jl")
include("Core/Index.jl")

export SKELETON_PHASE
export SkeletonPhaseError, SkeletonNotImplemented, BackendUnavailableError
export backend_library_path, require_backend
export Index, dim, id, tags, plev, hastag
export sim, prime, noprime, setprime
export replaceind, replaceinds, commoninds, uniqueinds

end
```

- [ ] **Step 4: Run tests to verify `Index` passes**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: PASS with bootstrap and `Index` testsets.

- [ ] **Step 5: Commit**

```bash
git add src/Tensor4all.jl src/Core/Index.jl test/runtests.jl test/core/index.jl
git commit -m "feat: add index skeleton"
```

---

### Task 3: Add the `Tensor` Metadata Skeleton and Array-Shape Validation

**Files:**
- Create: `src/Core/Tensor.jl`
- Modify: `src/Tensor4all.jl`
- Create: `test/core/tensor.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the failing `Tensor` tests**

```julia
# test/core/tensor.jl
using Test
using Tensor4all

@testset "Tensor skeleton" begin
    i = Tensor4all.Index(2; tags=["i"])
    j = Tensor4all.Index(3; tags=["j"])
    data = reshape(collect(1.0:6.0), 2, 3)
    tensor = Tensor4all.Tensor(data, [i, j])

    @test Tensor4all.rank(tensor) == 2
    @test Tensor4all.dims(tensor) == (2, 3)
    @test Tensor4all.inds(tensor) == [i, j]
    @test Tensor4all.inds(Tensor4all.prime(tensor)) == [Tensor4all.prime(i), Tensor4all.prime(j)]

    bad = PermutedDimsArray(reshape(collect(1.0:8.0), 2, 2, 2), (2, 1, 3))
    k = Tensor4all.Index(2; tags=["k"])
    @test_throws ArgumentError Tensor4all.Tensor(bad, [i, j, k])
    @test_throws DimensionMismatch Tensor4all.Tensor(data, [i])
    @test_throws Tensor4all.SkeletonNotImplemented Tensor4all.contract(tensor, tensor)
end
```

```julia
# test/runtests.jl
using Test
using Tensor4all

include("core/bootstrap.jl")
include("core/index.jl")
include("core/tensor.jl")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: FAIL because `Tensor`, `inds`, `rank`, `dims`, and `contract` do not exist yet.

- [ ] **Step 3: Implement `Tensor`**

```julia
# src/Core/Tensor.jl
struct Tensor{T,N}
    data::Array{T,N}
    inds::Vector{Index}
end

function Tensor(data::Array{T,N}, inds::AbstractVector{Index}) where {T,N}
    length(inds) == N || throw(DimensionMismatch(
        "Tensor rank $N requires $N indices, got $(length(inds))",
    ))
    Tuple(dim.(inds)) == size(data) || throw(DimensionMismatch(
        "Tensor dimensions $(Tuple(dim.(inds))) do not match data size $(size(data))",
    ))
    return Tensor{T,N}(data, collect(inds))
end

function Tensor(data, inds::AbstractVector{Index})
    throw(ArgumentError(
        "Array must be contiguous in memory for C API. Got $(typeof(data)). Use collect(data) to make a contiguous copy.",
    ))
end

inds(T::Tensor) = copy(T.inds)
rank(T::Tensor) = length(T.inds)
dims(T::Tensor) = size(T.data)

prime(T::Tensor, n::Integer=1) = Tensor(copy(T.data), prime.(inds(T), Ref(n)))

function swapinds(T::Tensor, a::Index, b::Index)
    newinds = [idx == a ? b : idx == b ? a : idx for idx in inds(T)]
    return Tensor(copy(T.data), newinds)
end

contract(::Tensor, ::Tensor) = throw(SkeletonNotImplemented(:contract, :core))
```

```julia
# src/Tensor4all.jl
module Tensor4all

using Libdl

const SKELETON_PHASE = true

include("Core/Errors.jl")
include("Core/Backend.jl")
include("Core/Index.jl")
include("Core/Tensor.jl")

export SKELETON_PHASE
export SkeletonPhaseError, SkeletonNotImplemented, BackendUnavailableError
export backend_library_path, require_backend
export Index, dim, id, tags, plev, hastag
export sim, prime, noprime, setprime
export replaceind, replaceinds, commoninds, uniqueinds
export Tensor, inds, rank, dims, swapinds, contract

end
```

- [ ] **Step 4: Run tests to verify `Tensor` passes**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: PASS with bootstrap, `Index`, and `Tensor` testsets.

- [ ] **Step 5: Commit**

```bash
git add src/Tensor4all.jl src/Core/Tensor.jl test/runtests.jl test/core/tensor.jl
git commit -m "feat: add tensor skeleton"
```

---

### Task 4: Add the TreeTN-General Network Skeleton and Chain Aliases

**Files:**
- Create: `src/TreeTN/TreeTensorNetwork.jl`
- Modify: `src/Tensor4all.jl`
- Create: `test/ttn/tree_tensor_network.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the failing TreeTN tests**

```julia
# test/ttn/tree_tensor_network.jl
using Test
using Tensor4all

@testset "TreeTensorNetwork skeleton" begin
    s1 = Tensor4all.Index(2; tags=["x1"])
    s2 = Tensor4all.Index(2; tags=["x2"])
    s3 = Tensor4all.Index(2; tags=["x3"])
    l12 = Tensor4all.Index(3; tags=["l12"])
    l23 = Tensor4all.Index(3; tags=["l23"])

    t1 = Tensor4all.Tensor(rand(2, 3), [s1, l12])
    t2 = Tensor4all.Tensor(rand(3, 2, 3), [l12, s2, l23])
    t3 = Tensor4all.Tensor(rand(3, 2), [l23, s3])

    tt = Tensor4all.TreeTensorNetwork(
        Dict(1 => t1, 2 => t2, 3 => t3);
        adjacency=Dict(1 => [2], 2 => [1, 3], 3 => [2]),
        siteinds=Dict(1 => [s1], 2 => [s2], 3 => [s3]),
        linkinds=Dict((1, 2) => l12, (2, 3) => l23),
    )

    @test Tensor4all.TensorTrain === Tensor4all.TreeTensorNetwork{Int}
    @test Tensor4all.MPS === Tensor4all.TensorTrain
    @test Tensor4all.MPO === Tensor4all.TensorTrain
    @test Tensor4all.vertices(tt) == [1, 2, 3]
    @test Tensor4all.neighbors(tt, 2) == [1, 3]
    @test Tensor4all.siteinds(tt, 2) == [s2]
    @test Tensor4all.linkind(tt, 1, 2) == l12
    @test Tensor4all.is_chain(tt)
    @test Tensor4all.is_mps_like(tt)
    @test !Tensor4all.is_mpo_like(tt)

    @test_throws Tensor4all.SkeletonNotImplemented Tensor4all.norm(tt)
    @test_throws Tensor4all.SkeletonNotImplemented Tensor4all.to_dense(tt)
end
```

```julia
# test/runtests.jl
using Test
using Tensor4all

include("core/bootstrap.jl")
include("core/index.jl")
include("core/tensor.jl")
include("ttn/tree_tensor_network.jl")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: FAIL because `TreeTensorNetwork`, `TensorTrain`, `vertices`, `neighbors`, `siteinds`, `linkind`, and the topology predicates do not exist yet.

- [ ] **Step 3: Implement the TreeTN skeleton**

```julia
# src/TreeTN/TreeTensorNetwork.jl
struct TreeTensorNetwork{V}
    tensors::Dict{V,Tensor}
    adjacency::Dict{V,Vector{V}}
    site_index_map::Dict{V,Vector{Index}}
    link_index_map::Dict{Tuple{V,V},Index}
end

function TreeTensorNetwork(
    tensors::Dict{V,Tensor};
    adjacency::Dict{V,Vector{V}},
    siteinds::Dict{V,Vector{Index}},
    linkinds::Dict{Tuple{V,V},Index},
) where {V}
    return TreeTensorNetwork{V}(tensors, adjacency, siteinds, linkinds)
end

const TensorTrain = TreeTensorNetwork{Int}
const MPS = TensorTrain
const MPO = TensorTrain

vertices(ttn::TreeTensorNetwork) = sort(collect(keys(ttn.tensors)))
neighbors(ttn::TreeTensorNetwork{V}, v::V) where {V} = copy(get(ttn.adjacency, v, V[]))
siteinds(ttn::TreeTensorNetwork, v) = copy(ttn.site_index_map[v])
linkind(ttn::TreeTensorNetwork, a, b) = get(ttn.link_index_map, (a, b), ttn.link_index_map[(b, a)])

function is_chain(ttn::TreeTensorNetwork{Int})
    verts = vertices(ttn)
    verts == collect(1:length(verts)) || return false
    degrees = Dict(v => length(neighbors(ttn, v)) for v in verts)
    count(==(1), values(degrees)) == 2 || return length(verts) == 1
    count(==(2), values(degrees)) == max(length(verts) - 2, 0)
end

is_chain(::TreeTensorNetwork) = false
is_mps_like(ttn::TreeTensorNetwork) = all(length(siteinds(ttn, v)) == 1 for v in vertices(ttn))
is_mpo_like(ttn::TreeTensorNetwork) = all(length(siteinds(ttn, v)) == 2 for v in vertices(ttn))

function _require_chain(ttn::TreeTensorNetwork, opname::Symbol)
    is_chain(ttn) || throw(ArgumentError("`$opname` requires a chain topology with vertices 1:n"))
    return ttn
end

orthogonalize!(ttn::TreeTensorNetwork, args...) = throw(SkeletonNotImplemented(:orthogonalize!, :tt))
truncate!(ttn::TreeTensorNetwork, args...) = throw(SkeletonNotImplemented(:truncate!, :tt))
inner(a::TreeTensorNetwork, b::TreeTensorNetwork) = throw(SkeletonNotImplemented(:inner, :tt))
norm(ttn::TreeTensorNetwork) = throw(SkeletonNotImplemented(:norm, :tt))
to_dense(ttn::TreeTensorNetwork) = throw(SkeletonNotImplemented(:to_dense, :tt))
evaluate(ttn::TreeTensorNetwork, args...) = throw(SkeletonNotImplemented(:evaluate, :tt))
contract(a::TreeTensorNetwork, b::TreeTensorNetwork) = throw(SkeletonNotImplemented(:contract, :tt))
```

```julia
# src/Tensor4all.jl
module Tensor4all

using Libdl

const SKELETON_PHASE = true

include("Core/Errors.jl")
include("Core/Backend.jl")
include("Core/Index.jl")
include("Core/Tensor.jl")
include("TreeTN/TreeTensorNetwork.jl")

export SKELETON_PHASE
export SkeletonPhaseError, SkeletonNotImplemented, BackendUnavailableError
export backend_library_path, require_backend
export Index, dim, id, tags, plev, hastag
export sim, prime, noprime, setprime
export replaceind, replaceinds, commoninds, uniqueinds
export Tensor, inds, rank, dims, swapinds, contract
export TreeTensorNetwork, TensorTrain, MPS, MPO
export vertices, neighbors, siteinds, linkind
export is_chain, is_mps_like, is_mpo_like
export orthogonalize!, truncate!, inner, norm, to_dense, evaluate

end
```

- [ ] **Step 4: Run tests to verify the TreeTN layer passes**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: PASS with the TreeTN testset verifying topology metadata and stubbed operations.

- [ ] **Step 5: Commit**

```bash
git add src/Tensor4all.jl src/TreeTN/TreeTensorNetwork.jl test/runtests.jl test/ttn/tree_tensor_network.jl
git commit -m "feat: add tree tensor network skeleton"
```

---

### Task 5: Add the Quantics Metadata Skeleton

**Files:**
- Create: `src/Quantics/Grids.jl`
- Create: `src/Quantics/Transforms.jl`
- Create: `src/Quantics/QTCI.jl`
- Modify: `src/Tensor4all.jl`
- Create: `test/quantics/grids.jl`
- Create: `test/quantics/transforms.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the failing quantics tests**

```julia
# test/quantics/grids.jl
using Test
using Tensor4all

@testset "Quantics grid metadata" begin
    x = Tensor4all.GridVariable(:x, 3; lower=0.0, upper=1.0, include_endpoint=true)
    y = Tensor4all.GridVariable(:y, 2; lower=-1.0, upper=1.0, include_endpoint=false)
    layout = Tensor4all.QuanticsLayout(:interleaved, [[:x, :y]])
    grid = Tensor4all.QuanticsGrid([x, y], layout)

    coords = Tensor4all.grid_coords(grid, (x=3, y=2))
    @test keys(coords) == (:x, :y)
    @test coords.x == 3 / 7
    @test coords.y == 0.0
end
```

```julia
# test/quantics/transforms.jl
using Test
using Tensor4all

@testset "Quantics transform metadata" begin
    shift = Tensor4all.shift_transform(; offsets=(x=1,))
    affine = Tensor4all.affine_transform(; matrix=[1.0 0.0; 0.0 1.0], shift=[0.0, 0.0])
    options = Tensor4all.QTCIOptions()

    @test shift.kind == :shift
    @test affine.kind == :affine
    @test options.max_rank == 64
    @test_throws Tensor4all.SkeletonNotImplemented Tensor4all.materialize_transform(shift)
end
```

```julia
# test/runtests.jl
using Test
using Tensor4all

include("core/bootstrap.jl")
include("core/index.jl")
include("core/tensor.jl")
include("ttn/tree_tensor_network.jl")
include("quantics/grids.jl")
include("quantics/transforms.jl")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: FAIL because the quantics types and transform constructors do not exist yet.

- [ ] **Step 3: Implement the quantics skeleton**

```julia
# src/Quantics/Grids.jl
struct GridVariable
    name::Symbol
    nbits::Int
    lower::Float64
    upper::Float64
    include_endpoint::Bool
end

function GridVariable(name::Symbol, nbits::Integer; lower::Real, upper::Real, include_endpoint::Bool=true)
    nbits > 0 || throw(ArgumentError("GridVariable requires at least one bit, got $nbits"))
    lower < upper || throw(ArgumentError("GridVariable requires lower < upper, got $lower >= $upper"))
    return GridVariable(name, Int(nbits), Float64(lower), Float64(upper), include_endpoint)
end

struct QuanticsLayout
    style::Symbol
    groups::Vector{Vector{Symbol}}
end

struct QuanticsGrid
    variables::Vector{GridVariable}
    layout::QuanticsLayout
end

function grid_coords(grid::QuanticsGrid, sample::NamedTuple)
    pairs = Pair{Symbol,Float64}[]
    for var in grid.variables
        level = getproperty(sample, var.name)
        levels = 2^var.nbits - (var.include_endpoint ? 1 : 0)
        step = (var.upper - var.lower) / levels
        push!(pairs, var.name => (var.lower + step * level))
    end
    return (; pairs...)
end
```

```julia
# src/Quantics/Transforms.jl
struct QuanticsTransform
    kind::Symbol
    parameters::NamedTuple
end

affine_transform(; matrix, shift) = QuanticsTransform(:affine, (; matrix, shift))
shift_transform(; offsets) = QuanticsTransform(:shift, (; offsets))
flip_transform(; variables) = QuanticsTransform(:flip, (; variables))
phase_rotation_transform(; phase) = QuanticsTransform(:phase_rotation, (; phase))
cumsum_transform(; variable) = QuanticsTransform(:cumsum, (; variable))
fourier_transform(; variables) = QuanticsTransform(:fourier, (; variables))
binaryop_transform(; op, variables) = QuanticsTransform(:binaryop, (; op, variables))

materialize_transform(::QuanticsTransform) =
    throw(SkeletonNotImplemented(:materialize_transform, :quantics))
```

```julia
# src/Quantics/QTCI.jl
Base.@kwdef struct QTCIOptions
    tolerance::Float64 = 1.0e-8
    max_rank::Int = 64
    max_sweeps::Int = 10
end

Base.@kwdef struct QTCIDiagnostics
    converged::Bool = false
    sweeps::Int = 0
    final_error::Float64 = Inf
end

struct QTCIResultPlaceholder
    options::QTCIOptions
    diagnostics::QTCIDiagnostics
end
```

```julia
# src/Tensor4all.jl
module Tensor4all

using Libdl

const SKELETON_PHASE = true

include("Core/Errors.jl")
include("Core/Backend.jl")
include("Core/Index.jl")
include("Core/Tensor.jl")
include("TreeTN/TreeTensorNetwork.jl")
include("Quantics/Grids.jl")
include("Quantics/Transforms.jl")
include("Quantics/QTCI.jl")

export SKELETON_PHASE
export SkeletonPhaseError, SkeletonNotImplemented, BackendUnavailableError
export backend_library_path, require_backend
export Index, dim, id, tags, plev, hastag
export sim, prime, noprime, setprime
export replaceind, replaceinds, commoninds, uniqueinds
export Tensor, inds, rank, dims, swapinds, contract
export TreeTensorNetwork, TensorTrain, MPS, MPO
export vertices, neighbors, siteinds, linkind
export is_chain, is_mps_like, is_mpo_like
export orthogonalize!, truncate!, inner, norm, to_dense, evaluate
export GridVariable, QuanticsLayout, QuanticsGrid, grid_coords
export QuanticsTransform
export affine_transform, shift_transform, flip_transform
export phase_rotation_transform, cumsum_transform, fourier_transform, binaryop_transform
export materialize_transform
export QTCIOptions, QTCIDiagnostics, QTCIResultPlaceholder

end
```

- [ ] **Step 4: Run tests to verify quantics passes**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: PASS with grid metadata, transform metadata, and QTCI placeholder coverage.

- [ ] **Step 5: Commit**

```bash
git add src/Tensor4all.jl src/Quantics/Grids.jl src/Quantics/Transforms.jl src/Quantics/QTCI.jl test/runtests.jl test/quantics/grids.jl test/quantics/transforms.jl
git commit -m "feat: add quantics skeleton"
```

---

### Task 6: Move Compatibility to Extension-Only Skeletons

**Files:**
- Modify: `Project.toml`
- Modify: `ext/Tensor4allITensorsExt.jl`
- Create: `ext/Tensor4allHDF5Ext.jl`
- Create: `test/extensions/itensors_ext.jl`
- Create: `test/extensions/hdf5_ext.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the failing extension tests**

```julia
# test/extensions/itensors_ext.jl
using Test
using Tensor4all
using ITensors

@testset "ITensors extension skeleton" begin
    ext = Base.get_extension(Tensor4all, :Tensor4allITensorsExt)
    @test ext !== nothing
    @test_throws Tensor4all.SkeletonNotImplemented ext.to_itensor(Tensor4all.Index(2))
end
```

```julia
# test/extensions/hdf5_ext.jl
using Test
using Tensor4all
using HDF5

@testset "HDF5 extension skeleton" begin
    ext = Base.get_extension(Tensor4all, :Tensor4allHDF5Ext)
    @test ext !== nothing
    @test_throws Tensor4all.SkeletonNotImplemented ext.save_hdf5("tmp.h5", nothing)
end
```

```julia
# test/runtests.jl
using Test
using Tensor4all

include("core/bootstrap.jl")
include("core/index.jl")
include("core/tensor.jl")
include("ttn/tree_tensor_network.jl")
include("quantics/grids.jl")
include("quantics/transforms.jl")
include("extensions/itensors_ext.jl")
include("extensions/hdf5_ext.jl")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: FAIL because the extension modules either do not expose the stub APIs or HDF5 is not wired as an extension yet.

- [ ] **Step 3: Implement extension-only stubs**

```toml
# Project.toml
[deps]
Libdl = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
RustToolChain = "e9dc52e2-edb8-4742-9783-5e542d30dbb5"

[weakdeps]
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
ITensors = "9136182c-28ba-11e9-034c-db9fb085ebd5"

[extensions]
Tensor4allHDF5Ext = ["HDF5"]
Tensor4allITensorsExt = ["ITensors"]

[extras]
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
ITensors = "9136182c-28ba-11e9-034c-db9fb085ebd5"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["HDF5", "ITensors", "Test"]
```

```julia
# ext/Tensor4allITensorsExt.jl
module Tensor4allITensorsExt

using Tensor4all
using ITensors

to_itensor(::Tensor4all.Index) = throw(Tensor4all.SkeletonNotImplemented(:to_itensor, :extensions))
from_itensor(::ITensors.Index) = throw(Tensor4all.SkeletonNotImplemented(:from_itensor, :extensions))

end
```

```julia
# ext/Tensor4allHDF5Ext.jl
module Tensor4allHDF5Ext

using Tensor4all
using HDF5

save_hdf5(args...) = throw(Tensor4all.SkeletonNotImplemented(:save_hdf5, :extensions))
load_hdf5(args...) = throw(Tensor4all.SkeletonNotImplemented(:load_hdf5, :extensions))

end
```

- [ ] **Step 4: Run tests to verify extension wiring passes**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: PASS with both extensions loading only when their packages are available.

- [ ] **Step 5: Commit**

```bash
git add Project.toml ext/Tensor4allITensorsExt.jl ext/Tensor4allHDF5Ext.jl test/runtests.jl test/extensions/itensors_ext.jl test/extensions/hdf5_ext.jl
git commit -m "feat: add extension skeletons"
```

---

### Task 7: Restore the Reviewable API Docs Surface

**Files:**
- Modify: `docs/make.jl`
- Modify: `docs/src/index.md`
- Modify: `docs/src/modules.md`
- Create: `docs/src/api.md`
- Modify: `docs/src/design_documents.md`
- Modify: `docs/src/deferred_rework_plan.md`
- Modify: `README.md`

- [ ] **Step 1: Write the docs changes**

````markdown
# docs/src/api.md
# API Reference

## Core

```@docs
Tensor4all.Index
Tensor4all.Tensor
Tensor4all.SkeletonNotImplemented
Tensor4all.BackendUnavailableError
Tensor4all.backend_library_path
Tensor4all.require_backend
```

## TreeTN

```@docs
Tensor4all.TreeTensorNetwork
Tensor4all.TensorTrain
Tensor4all.MPS
Tensor4all.MPO
Tensor4all.is_chain
Tensor4all.is_mps_like
Tensor4all.is_mpo_like
```

## Quantics

```@docs
Tensor4all.GridVariable
Tensor4all.QuanticsLayout
Tensor4all.QuanticsGrid
Tensor4all.QuanticsTransform
Tensor4all.QTCIOptions
Tensor4all.QTCIDiagnostics
Tensor4all.QTCIResultPlaceholder
```
````

```julia
# docs/make.jl
makedocs(
    sitename="Tensor4all.jl",
    modules=[Tensor4all],
    repo=Documenter.Remotes.GitHub("tensor4all", "Tensor4all.jl"),
    pages=[
        "Home" => "index.md",
        "Architecture Status" => "modules.md",
        "API Reference" => "api.md",
        "Design Documents" => "design_documents.md",
        "Deferred Rework Plan" => "deferred_rework_plan.md",
    ],
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tensor4all.github.io/Tensor4all.jl",
    ),
    warnonly=[:missing_docs],
)
```

```markdown
# README.md

## Current status

`Tensor4all.jl` is in a review-first skeleton phase. The package surface is
being rebuilt around the design documents under `docs/design/`. Public APIs may
exist before backend numerics are implemented; such calls should fail with
actionable skeleton exceptions rather than silently doing work.

## Design and planning docs

- `docs/design/README.md`
- `docs/plans/2026-04-10-tensor4all-rework-followup.md`
```

- [ ] **Step 2: Add docstrings while touching each public type and function**

For each file introduced in Tasks 1-6, add docstrings with:

- one-sentence summary
- note whether the symbol is implemented metadata or stub-only
- a `# Examples` section
- `jldoctest` blocks for pure metadata APIs only
- fenced `julia` blocks for backend-backed placeholders

Minimum docstrings to add:

````julia
"""
    Index(dim; tags=String[], plev=0, id=next_index_id())

Create a Julia-side review skeleton for an indexed tensor leg.

# Examples
```jldoctest
julia> i = Index(4; tags=["x"])
Index(4|x)
```
"""
````

- [ ] **Step 3: Build docs and fix missing-page or docstring issues**

Run:

```bash
julia --project=docs docs/make.jl
```

Expected: PASS with a docs site that shows the new API reference page and clearly distinguishes implemented metadata from stubbed backend behavior.

- [ ] **Step 4: Commit**

```bash
git add docs/make.jl docs/src/index.md docs/src/modules.md docs/src/api.md docs/src/design_documents.md docs/src/deferred_rework_plan.md README.md src
git commit -m "docs: restore skeleton api review surface"
```

---

### Task 8: Final Skeleton Validation and Review Handoff

**Files:**
- Modify: `docs/plans/2026-04-10-tensor4all-rework-followup.md`
- Modify: `docs/src/deferred_rework_plan.md`

- [ ] **Step 1: Run the full verification set**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
julia --project=docs docs/make.jl
rg -n "SimpleTT|TreeTCI|TTFunction|QuanticsTCI" src test docs/src README.md
```

Expected:

- `Pkg.test()` PASS
- `docs/make.jl` PASS
- the final `rg` command returns either no matches or only matches that explain those names are intentionally out of scope or deferred to `BubbleTeaCI`

- [ ] **Step 2: Record review outcomes inline**

At the end of this plan file, add a short implementation note after execution:

```markdown
## Execution Notes

- Implemented through Task N
- Open review decisions:
  - ...
- Follow-up backend gaps:
  - tensor-tensor contraction still needs C API coverage
```

- [ ] **Step 3: Commit**

```bash
git add docs/plans/2026-04-10-tensor4all-rework-followup.md docs/src/deferred_rework_plan.md
git commit -m "chore: record skeleton rework review outcomes"
```

---

## Spec Coverage Check

- `julia_ffi_core.md`: covered by Tasks 1-3
- `julia_ffi_tt.md`: covered by Task 4
- `julia_ffi_quantics.md`: covered by Task 5
- `julia_ffi_extensions.md`: covered by Task 6
- review-first docs protocol: covered by Task 7
- staged review and verification: covered by Task 8

## Important Open Decisions Before Execution

- whether `Index` should remain a pure Julia metadata skeleton during the review phase or carry an explicit nullable backend handle field from day one
- whether `Tensor` should store Julia-owned dense arrays in the skeleton phase or wrap a lightweight backend-handle placeholder object
- whether `HDF5` should move to weak dependency status immediately or in a follow-up commit paired with CI updates
- whether `QuanticsLayout.style` should stay as `Symbol` in the skeleton or use an enum type from the beginning

## Notes for Implementers

- Keep each task small and reviewable. Do not batch Tasks 2-6 into one commit.
- Do not restore any pre-reset files unless the new plan explicitly recreates them.
- Any public API that is not genuinely implemented must throw `SkeletonNotImplemented`.
- Pure metadata behavior may be fully implemented when it helps review and testing.
- Chain-specific behavior must use runtime checks on top of `TreeTensorNetwork`; do not create a separate chain-only core type.
