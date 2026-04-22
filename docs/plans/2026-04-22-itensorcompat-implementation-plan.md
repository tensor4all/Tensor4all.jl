# ITensorCompat Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an opt-in `Tensor4all.ITensorCompat` facade that lets ITensors-style downstream code use Tensor4all tensor trains with minimal local glue.

**Architecture:** Keep `TensorNetworks.TensorTrain` as the primary generic chain type. Add conservative Core and TensorNetworks compatibility primitives, then layer `ITensorCompat.MPS` / `ITensorCompat.MPO` wrappers over existing `TensorTrain` values. All `llim` / `rlim` invalidation stays inside Tensor4all.jl.

**Tech Stack:** Julia 1.11, Tensor4all Core/TensorNetworks modules, existing Rust C API kernels, Documenter.jl, `Test`.

---

## Notes Before Starting

- Read `AGENTS.md` first. The design must preserve the public layer split:
  `Core`, `TensorNetworks`, `SimpleTT`, `TensorCI`, `QuanticsGrids`,
  `QuanticsTCI`, and `QuanticsTransform`.
- Do not add C API functions. This compatibility surface is Julia-owned.
- Do not change `TensorNetworks.siteinds(::TensorTrain)` to return a flat
  vector. Flat site indices are only for `ITensorCompat.MPS`.
- Keep `Tensor` immutable in this first pass. Implement non-mutating tensor
  index replacement. Use train-slot assignment for mutation-like workflows.
- Each task below should be committed separately.
- Focused tests can be run directly with `julia --startup-file=no --project=.`
  unless the host has the AMD EPYC Julia issue described in `AGENTS.md`.

---

### Task 1: Core ITensors-Style Tensor and Index Primitives

**Files:**
- Modify: `src/Core/Index.jl`
- Modify: `src/Core/Tensor.jl`
- Modify: `src/Tensor4all.jl`
- Test: `test/core/index.jl`
- Test: `test/core/tensor.jl`

**Step 1: Write failing index tests**

Append tests to `test/core/index.jl`:

```julia
@testset "ITensors-style Index constructor" begin
    i = Index(3, "x")
    @test dim(i) == 3
    @test tags(i) == ["x"]

    j = Index(4, "x,y"; plev=2)
    @test dim(j) == 4
    @test tags(j) == ["x", "y"]
    @test plev(j) == 2
end
```

**Step 2: Write failing tensor primitive tests**

Append tests to `test/core/tensor.jl`:

```julia
@testset "ITensors-style Tensor primitives" begin
    i = Index(2, "i")
    j = Index(3, "j")
    k = Index(5, "k")
    i2 = sim(i)

    a = Tensor(reshape(collect(1.0:6.0), 2, 3), [i, j])
    b = Tensor(reshape(collect(1.0:15.0), 3, 5), [j, k])

    @test commoninds(a, b) == [j]
    @test uniqueinds(a, b) == [i]
    @test hasinds(a, i)
    @test hasinds(a, i, j)
    @test !hasinds(a, k)
    @test eltype(a) == Float64

    c = a * b
    @test c ≈ contract(a, b)

    scalar_tensor = Tensor(fill(3.5), Index[])
    @test scalar(scalar_tensor) == 3.5

    replaced = replaceind(a, i, i2)
    @test inds(replaced) == [i2, j]
    @test inds(a) == [i, j]
    @test Array(replaced, i2, j) == Array(a, i, j)

    missing = replaceind(a, k, Index(5, "newk"))
    @test inds(missing) == inds(a)

    bad = Index(4, "bad")
    @test_throws ArgumentError replaceind(a, i, bad)
end
```

**Step 3: Run focused tests to verify failure**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/core/index.jl")'
julia --startup-file=no --project=. -e 'include("test/core/tensor.jl")'
```

Expected: failures for missing `Index(dim, tag)`, tensor `commoninds`,
`uniqueinds`, `hasinds`, `scalar`, `replaceind(::Tensor, ...)`, `eltype`, or
`*(::Tensor, ::Tensor)`.

**Step 4: Implement minimal Core API**

In `src/Core/Index.jl`, add:

```julia
function Index(
    dim::Integer,
    tag::AbstractString;
    tags::Union{Nothing, AbstractVector{<:AbstractString}}=nothing,
    kwargs...,
)
    tag_list = tags === nothing ? split(String(tag), ',') : collect(String.(tags))
    tag_list = filter(!isempty, strip.(tag_list))
    return Index(dim; tags=tag_list, kwargs...)
end
```

Also add collection helpers with dimension validation:

```julia
function _replace_index_vector(xs::AbstractVector{Index}, old::Index, new::Index)
    if old in xs && dim(old) != dim(new)
        throw(ArgumentError("Cannot replace index $old of dimension $(dim(old)) with $new of dimension $(dim(new))"))
    end
    return [x == old ? new : x for x in xs]
end

replaceind(xs::AbstractVector{Index}, old::Index, new::Index) =
    _replace_index_vector(xs, old, new)
replaceind(xs::AbstractVector{Index}, replacement::Pair{Index,Index}) =
    replaceind(xs, first(replacement), last(replacement))
```

In `src/Core/Tensor.jl`, add:

```julia
Base.eltype(t::Tensor) = eltype(t.data)

commoninds(a::Tensor, b::Tensor) = commoninds(inds(a), inds(b))
uniqueinds(a::Tensor, b::Tensor) = uniqueinds(inds(a), inds(b))
hasinds(t::Tensor, query::Index...) = all(index -> index in inds(t), query)

function scalar(t::Tensor)
    rank(t) == 0 || throw(ArgumentError("scalar requires a rank-0 Tensor, got rank $(rank(t))"))
    return only(t.data)
end

function replaceind(t::Tensor, old::Index, new::Index)
    return Tensor(copy(t.data), replaceind(inds(t), old, new); backend_handle=t.backend_handle)
end

replaceind(t::Tensor, replacement::Pair{Index,Index}) =
    replaceind(t, first(replacement), last(replacement))

Base.:*(a::Tensor, b::Tensor) = contract(a, b)
```

Export any new public names from `src/Tensor4all.jl`, especially `hasinds` and
`scalar`.

Do not add `replaceind!` in this task because `Tensor` is immutable.

**Step 5: Run tests to verify pass**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/core/index.jl")'
julia --startup-file=no --project=. -e 'include("test/core/tensor.jl")'
```

Expected: both commands pass.

**Step 6: Commit**

```bash
git add src/Core/Index.jl src/Core/Tensor.jl src/Tensor4all.jl test/core/index.jl test/core/tensor.jl
git commit -m "feat(core): add ITensors-style tensor primitives"
```

---

### Task 2: TensorTrain Canonical Invalidation and Topology Mutation

**Files:**
- Modify: `src/TensorNetworks/types.jl`
- Modify: `src/TensorNetworks.jl`
- Test: `test/tensornetworks/llim_rlim.jl`

**Step 1: Write failing invalidation tests**

Append to `test/tensornetworks/llim_rlim.jl`:

```julia
@testset "TensorTrain mutation invalidates canonical window" begin
    s1 = Index(2; tags=["s", "s=1"])
    s2 = Index(2; tags=["s", "s=2"])
    s3 = Index(2; tags=["s", "s=3"])
    l1 = Index(1; tags=["Link", "l=1"])
    l2 = Index(1; tags=["Link", "l=2"])

    t1 = Tensor(ones(2, 1), [s1, l1])
    t2 = Tensor(ones(1, 2), [l1, s2])
    t3 = Tensor(ones(1, 2, 1), [l1, s2, l2])
    t4 = Tensor(ones(1, 2), [l2, s3])

    tt = TensorNetworks.TensorTrain([t1, t2], 1, 3)
    @test TensorNetworks.invalidate_canonical!(tt) === tt
    @test (tt.llim, tt.rlim) == (0, length(tt) + 1)

    tt = TensorNetworks.TensorTrain([t1, t2], 1, 3)
    tt[1] = t1
    @test (tt.llim, tt.rlim) == (0, length(tt) + 1)

    tt = TensorNetworks.TensorTrain([t1, t4], 1, 3)
    insert!(tt, 2, t3)
    @test length(tt) == 3
    @test (tt.llim, tt.rlim) == (0, length(tt) + 1)

    deleteat!(tt, 2)
    @test length(tt) == 2
    @test (tt.llim, tt.rlim) == (0, length(tt) + 1)

    push!(tt, t4)
    @test length(tt) == 3
    @test (tt.llim, tt.rlim) == (0, length(tt) + 1)

    pushfirst!(tt, t1)
    @test length(tt) == 4
    @test (tt.llim, tt.rlim) == (0, length(tt) + 1)
end
```

**Step 2: Run focused test to verify failure**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/tensornetworks/llim_rlim.jl")'
```

Expected: failure for missing `invalidate_canonical!` and mutation methods, or
old local `setindex!` behavior.

**Step 3: Implement invalidation API**

In `src/TensorNetworks/types.jl`, add:

```julia
function invalidate_canonical!(tt::TensorTrain)
    tt.llim = 0
    tt.rlim = length(tt) + 1
    return tt
end

function Base.setindex!(tt::TensorTrain, value::Tensor, i::Int)
    tt.data[i] = value
    invalidate_canonical!(tt)
    return value
end

function Base.insert!(tt::TensorTrain, i::Integer, value::Tensor)
    insert!(tt.data, Int(i), value)
    invalidate_canonical!(tt)
    return tt
end

function Base.deleteat!(tt::TensorTrain, i)
    deleteat!(tt.data, i)
    invalidate_canonical!(tt)
    return tt
end

function Base.push!(tt::TensorTrain, value::Tensor)
    push!(tt.data, value)
    invalidate_canonical!(tt)
    return tt
end

function Base.pushfirst!(tt::TensorTrain, value::Tensor)
    pushfirst!(tt.data, value)
    invalidate_canonical!(tt)
    return tt
end
```

In `src/TensorNetworks.jl`, export `invalidate_canonical!`.

**Step 4: Run tests to verify pass**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/tensornetworks/llim_rlim.jl")'
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/TensorNetworks/types.jl src/TensorNetworks.jl test/tensornetworks/llim_rlim.jl
git commit -m "feat(TensorNetworks): centralize TensorTrain invalidation"
```

---

### Task 3: ITensorCompat Module Skeleton and Basic MPS Wrapper

**Files:**
- Create: `src/ITensorCompat.jl`
- Modify: `src/Tensor4all.jl`
- Modify: `test/runtests.jl`
- Create: `test/itensorcompat/surface.jl`
- Modify: `docs/src/api.md`

**Step 1: Write failing MPS surface tests**

Create `test/itensorcompat/surface.jl`:

```julia
using Test
using Tensor4all

const IC = Tensor4all.ITensorCompat
const TN = Tensor4all.TensorNetworks

@testset "ITensorCompat MPS surface" begin
    s1 = Index(2, "s=1")
    s2 = Index(3, "s=2")
    l1 = Index(4; tags=["Link", "l=1"])

    t1 = Tensor(randn(2, 4), [s1, l1])
    t2 = Tensor(randn(4, 3), [l1, s2])
    m = IC.MPS(TN.TensorTrain([t1, t2]))

    @test length(m) == 2
    @test collect(IC.siteinds(m)) == [s1, s2]
    @test IC.linkinds(m) == [l1]
    @test IC.linkdims(m) == [4]
    @test IC.rank(m) == 4
    @test eltype(m) == Float64
    @test m[1] == t1

    new_t1 = Tensor(randn(2, 4), [s1, l1])
    @test (m[1] = new_t1) == new_t1
    @test m[1] == new_t1
    @test (m.tt.llim, m.tt.rlim) == (0, length(m) + 1)
end

@testset "ITensorCompat MPS validation" begin
    s1 = Index(2, "s=1")
    s2 = Index(2, "s=2")
    t = Tensor(randn(2, 2), [s1, s2])
    @test_throws ArgumentError IC.MPS(TN.TensorTrain([t]))
end
```

Add to `test/runtests.jl`:

```julia
include("itensorcompat/surface.jl")
```

**Step 2: Run focused test to verify failure**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/itensorcompat/surface.jl")'
```

Expected: failure because `Tensor4all.ITensorCompat` does not exist.

**Step 3: Implement module skeleton**

Create `src/ITensorCompat.jl`:

```julia
module ITensorCompat

import LinearAlgebra
import ..Tensor4all: Index, Tensor, dim, inds
import ..TensorNetworks
import ..TensorNetworks: TensorTrain
import ..TensorNetworks: siteinds, linkinds, linkdims

export MPS, MPO
export siteinds, linkinds, linkdims, rank

mutable struct MPS
    tt::TensorTrain
    function MPS(tt::TensorTrain)
        groups = TensorNetworks.siteinds(tt)
        for (position, group) in pairs(groups)
            length(group) == 1 || throw(ArgumentError(
                "MPS expects exactly one site index at tensor $position, got $(length(group))",
            ))
        end
        return new(tt)
    end
end

Base.length(m::MPS) = length(m.tt)
Base.iterate(m::MPS, state...) = iterate(m.tt, state...)
Base.getindex(m::MPS, i::Int) = m.tt[i]
Base.setindex!(m::MPS, tensor::Tensor, i::Int) = setindex!(m.tt, tensor, i)

siteinds(m::MPS) = [only(group) for group in TensorNetworks.siteinds(m.tt)]
linkinds(m::MPS) = TensorNetworks.linkinds(m.tt)
linkdims(m::MPS) = TensorNetworks.linkdims(m.tt)
rank(m::MPS) = maximum(linkdims(m); init=0)
Base.eltype(m::MPS) = eltype(first(m.tt).data)

mutable struct MPO
    tt::TensorTrain
end

end
```

In `src/Tensor4all.jl`, include and export the submodule after
`TensorNetworks.jl`:

```julia
include("ITensorCompat.jl")
export ITensorCompat
```

In `docs/src/api.md`, add an `ITensorCompat` section with an `@autodocs` page
entry for `ITensorCompat.jl`.

**Step 4: Run tests to verify pass**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/itensorcompat/surface.jl")'
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/ITensorCompat.jl src/Tensor4all.jl test/runtests.jl test/itensorcompat/surface.jl docs/src/api.md
git commit -m "feat: add ITensorCompat MPS wrapper"
```

---

### Task 4: MPS Algebra, Dense, Evaluation, and Site Replacement

**Files:**
- Modify: `src/ITensorCompat.jl`
- Test: `test/itensorcompat/surface.jl`

**Step 1: Add failing MPS algebra tests**

Append to `test/itensorcompat/surface.jl`:

```julia
@testset "ITensorCompat MPS algebra and dense operations" begin
    s1 = Index(2, "s=1")
    s2 = Index(2, "s=2")
    a = IC.MPS(TN.random_tt([s1, s2]; linkdims=2))
    b = IC.MPS(TN.random_tt([s1, s2]; linkdims=2))

    @test IC.dot(a, b) == TN.dot(a.tt, b.tt)
    @test IC.inner(a, b) == TN.inner(a.tt, b.tt)
    @test IC.norm(a) ≈ TN.norm(a.tt)
    @test IC.to_dense(a) ≈ TN.to_dense(a.tt)

    c = a + b
    @test c isa IC.MPS
    @test IC.to_dense(c) ≈ TN.to_dense(TN.add(a.tt, b.tt))

    scaled = 2.0 * a
    @test scaled isa IC.MPS
    @test IC.to_dense(scaled) ≈ TN.to_dense(2.0 * a.tt)

    value = IC.evaluate(a, IC.siteinds(a), [1, 2])
    @test value == TN.evaluate(a.tt, IC.siteinds(a), [1, 2])

    new_sites = [sim(s) for s in IC.siteinds(a)]
    replaced = IC.replace_siteinds(a, IC.siteinds(a), new_sites)
    @test IC.siteinds(replaced) == new_sites
    @test IC.siteinds(a) == [s1, s2]
    @test IC.replace_siteinds!(a, IC.siteinds(a), new_sites) === a
    @test IC.siteinds(a) == new_sites
end
```

**Step 2: Run focused test to verify failure**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/itensorcompat/surface.jl")'
```

Expected: missing algebra and dense methods.

**Step 3: Implement MPS forwarding methods**

In `src/ITensorCompat.jl`, add imports and methods:

```julia
import ..Tensor4all: add, dag, dot, inner, norm
import ..TensorNetworks: replace_siteinds, replace_siteinds!, to_dense, evaluate

export add, dag, dot, inner, norm
export replace_siteinds, replace_siteinds!, to_dense, evaluate

dot(a::MPS, b::MPS) = TensorNetworks.dot(a.tt, b.tt)
inner(a::MPS, b::MPS) = TensorNetworks.inner(a.tt, b.tt)
norm(m::MPS) = TensorNetworks.norm(m.tt)
to_dense(m::MPS) = TensorNetworks.to_dense(m.tt)
evaluate(m::MPS, indices, values) = TensorNetworks.evaluate(m.tt, collect(indices), values)

function add(a::MPS, b::MPS; kwargs...)
    return MPS(TensorNetworks.add(a.tt, b.tt; kwargs...))
end

Base.:+(a::MPS, b::MPS) = add(a, b)
Base.:*(alpha::Number, m::MPS) = MPS(alpha * m.tt)
Base.:*(m::MPS, alpha::Number) = alpha * m
Base.:/(m::MPS, alpha::Number) = MPS(m.tt / alpha)
dag(m::MPS) = MPS(TensorNetworks.dag(m.tt))

replace_siteinds(m::MPS, oldsites, newsites) =
    MPS(TensorNetworks.replace_siteinds(m.tt, collect(oldsites), collect(newsites)))

function replace_siteinds!(m::MPS, oldsites, newsites)
    TensorNetworks.replace_siteinds!(m.tt, collect(oldsites), collect(newsites))
    return m
end
```

If `TensorNetworks.add` rejects mismatched site indices, keep that behavior.
Do not auto-align site indices in this task.

**Step 4: Run tests to verify pass**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/itensorcompat/surface.jl")'
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/ITensorCompat.jl test/itensorcompat/surface.jl
git commit -m "feat(ITensorCompat): forward MPS algebra operations"
```

---

### Task 5: ITensors Cutoff Compatibility and Mutating Canonical Operations

**Files:**
- Modify: `src/ITensorCompat.jl`
- Test: `test/itensorcompat/surface.jl`

**Step 1: Add failing cutoff and mutating operation tests**

Append:

```julia
@testset "ITensorCompat cutoff and mutating canonical operations" begin
    s1 = Index(2, "s=1")
    s2 = Index(2, "s=2")
    m = IC.MPS(TN.random_tt([s1, s2]; linkdims=2))

    before_dense = IC.to_dense(m)
    @test IC.orthogonalize!(m, 1) === m
    @test IC.to_dense(m) ≈ before_dense

    @test IC.truncate!(m; maxdim=1) === m
    @test maximum(IC.linkdims(m); init=0) <= 1

    m2 = IC.MPS(TN.random_tt([s1, s2]; linkdims=2))
    @test IC.truncate!(m2; cutoff=1e-12) === m2
    @test_throws ArgumentError IC.truncate!(m2; cutoff=1e-12, threshold=1e-12)
end
```

**Step 2: Run focused test to verify failure**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/itensorcompat/surface.jl")'
```

Expected: missing `orthogonalize!`, `truncate!`, and cutoff translation.

**Step 3: Implement cutoff resolver and mutating wrappers**

In `src/ITensorCompat.jl`, add:

```julia
import ..TensorNetworks: orthogonalize, truncate, SvdTruncationPolicy

export orthogonalize!, truncate!

const ITENSORS_CUTOFF_POLICY = SvdTruncationPolicy(
    measure = :squared_value,
    rule = :discarded_tail_sum,
)

function _compat_truncation_kwargs(; cutoff=0.0, threshold=nothing, svd_policy=nothing, kwargs...)
    if threshold !== nothing && cutoff != 0.0
        throw(ArgumentError("Pass either cutoff or threshold, got cutoff=$cutoff and threshold=$threshold"))
    end
    if cutoff != 0.0
        svd_policy !== nothing && throw(ArgumentError("cutoff implies ITensors cutoff policy; do not also pass svd_policy"))
        return (; threshold=cutoff, svd_policy=ITENSORS_CUTOFF_POLICY, kwargs...)
    end
    if threshold === nothing
        return (; svd_policy=svd_policy, kwargs...)
    end
    return (; threshold=threshold, svd_policy=svd_policy, kwargs...)
end

function orthogonalize!(m::MPS, site::Integer; kwargs...)
    m.tt = TensorNetworks.orthogonalize(m.tt, site; kwargs...)
    return m
end

function truncate!(m::MPS; cutoff=0.0, threshold=nothing, svd_policy=nothing, kwargs...)
    resolved = _compat_truncation_kwargs(; cutoff, threshold, svd_policy, kwargs...)
    m.tt = TensorNetworks.truncate(m.tt; resolved...)
    return m
end
```

If `TensorNetworks.truncate` requires one of `threshold > 0` or `maxdim > 0`,
keep that error. The test uses `maxdim=1` and `cutoff=1e-12`.

**Step 4: Run tests to verify pass**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/itensorcompat/surface.jl")'
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/ITensorCompat.jl test/itensorcompat/surface.jl
git commit -m "feat(ITensorCompat): add cutoff-aware MPS truncation"
```

---

### Task 6: Raw Array MPS Constructors

**Files:**
- Modify: `src/ITensorCompat.jl`
- Test: `test/itensorcompat/surface.jl`

**Step 1: Add failing raw constructor tests**

Append:

```julia
@testset "ITensorCompat raw MPS constructors" begin
    sites = [Index(2, "s=1"), Index(3, "s=2")]
    blocks = [
        reshape(collect(1.0:4.0), 1, 2, 2),
        reshape(collect(1.0:6.0), 2, 3, 1),
    ]

    m = IC.MPS(blocks, sites)
    @test IC.siteinds(m) == sites
    @test IC.linkdims(m) == [2]

    dense = Array(IC.to_dense(m), sites...)
    @test size(dense) == (2, 3)

    inferred = IC.MPS(blocks)
    @test dim.(IC.siteinds(inferred)) == [2, 3]
end
```

**Step 2: Run focused test to verify failure**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/itensorcompat/surface.jl")'
```

Expected: missing raw array constructors.

**Step 3: Implement raw MPS constructors**

Add to `src/ITensorCompat.jl`:

```julia
function _mps_links(blocks::AbstractVector{<:Array{T,3}}) where T
    n = length(blocks)
    return [Index(size(blocks[i], 3); tags=["Link", "l=$i"]) for i in 1:(n - 1)]
end

function _validate_mps_blocks(blocks, sites)
    length(blocks) == length(sites) || throw(DimensionMismatch(
        "Need one MPS block per site, got $(length(blocks)) blocks and $(length(sites)) sites",
    ))
    for i in eachindex(blocks)
        size(blocks[i], 2) == dim(sites[i]) || throw(DimensionMismatch(
            "Block $i physical dimension $(size(blocks[i], 2)) does not match site dimension $(dim(sites[i]))",
        ))
    end
    return nothing
end

function MPS(blocks::AbstractVector{<:Array{T,3}}, sites::Vector{Index}) where T
    isempty(blocks) && throw(ArgumentError("MPS blocks must not be empty"))
    _validate_mps_blocks(blocks, sites)
    links = _mps_links(blocks)
    tensors = Tensor[]
    for i in eachindex(blocks)
        if length(blocks) == 1
            push!(tensors, Tensor(dropdims(blocks[i]; dims=(1, 3)), [sites[i]]))
        elseif i == 1
            push!(tensors, Tensor(dropdims(blocks[i]; dims=1), [sites[i], links[i]]))
        elseif i == length(blocks)
            push!(tensors, Tensor(dropdims(blocks[i]; dims=3), [links[i - 1], sites[i]]))
        else
            push!(tensors, Tensor(blocks[i], [links[i - 1], sites[i], links[i]]))
        end
    end
    return MPS(TensorTrain(tensors))
end

function MPS(blocks::AbstractVector{<:Array{T,3}}) where T
    sites = [Index(size(blocks[i], 2); tags=["site", "site=$i"]) for i in eachindex(blocks)]
    return MPS(blocks, sites)
end
```

Use `collect` if any block is not a contiguous `Array`. The first signature
requires `Array{T,3}`, so contiguous storage is already enforced.

**Step 4: Run tests to verify pass**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/itensorcompat/surface.jl")'
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/ITensorCompat.jl test/itensorcompat/surface.jl
git commit -m "feat(ITensorCompat): construct MPS from raw blocks"
```

---

### Task 7: Narrow MPO Wrapper

**Files:**
- Modify: `src/ITensorCompat.jl`
- Test: `test/itensorcompat/surface.jl`

**Step 1: Add failing MPO wrapper tests**

Append:

```julia
@testset "ITensorCompat MPO surface" begin
    x1 = Index(2, "x=1")
    y1 = Index(2, "y=1")
    x2 = Index(3, "x=2")
    y2 = Index(3, "y=2")
    l1 = Index(2; tags=["Link", "l=1"])

    w1 = Tensor(randn(2, 2, 2), [x1, y1, l1])
    w2 = Tensor(randn(2, 3, 3), [l1, x2, y2])
    W = IC.MPO(TN.TensorTrain([w1, w2]))

    @test length(W) == 2
    @test IC.siteinds(W) == [[x1, y1], [x2, y2]]
    @test IC.linkdims(W) == [2]
    @test IC.rank(W) == 2
    @test W[1] == w1

    new_w1 = Tensor(randn(2, 2, 2), [x1, y1, l1])
    @test (W[1] = new_w1) == new_w1
    @test W[1] == new_w1

    bad = Tensor(randn(2), [x1])
    @test_throws ArgumentError IC.MPO(TN.TensorTrain([bad]))
end
```

**Step 2: Run focused test to verify failure**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/itensorcompat/surface.jl")'
```

Expected: `MPO` is not yet validating or exposing methods.

**Step 3: Implement narrow MPO wrapper**

In `src/ITensorCompat.jl`, replace the placeholder `MPO` with:

```julia
mutable struct MPO
    tt::TensorTrain
    function MPO(tt::TensorTrain)
        groups = TensorNetworks.siteinds(tt)
        for (position, group) in pairs(groups)
            length(group) == 2 || throw(ArgumentError(
                "MPO expects exactly two site indices at tensor $position, got $(length(group))",
            ))
        end
        return new(tt)
    end
end

Base.length(W::MPO) = length(W.tt)
Base.getindex(W::MPO, i::Int) = W.tt[i]
Base.setindex!(W::MPO, tensor::Tensor, i::Int) = setindex!(W.tt, tensor, i)

siteinds(W::MPO) = TensorNetworks.siteinds(W.tt)
linkinds(W::MPO) = TensorNetworks.linkinds(W.tt)
linkdims(W::MPO) = TensorNetworks.linkdims(W.tt)
rank(W::MPO) = maximum(linkdims(W); init=0)
dag(W::MPO) = MPO(TensorNetworks.dag(W.tt))
```

Do not add MPO raw constructors in this task unless a test requires them.

**Step 4: Run tests to verify pass**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/itensorcompat/surface.jl")'
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/ITensorCompat.jl test/itensorcompat/surface.jl
git commit -m "feat(ITensorCompat): add narrow MPO wrapper"
```

---

### Task 8: Documentation and Full Verification

**Files:**
- Modify: `docs/src/api.md`
- Modify if needed: `docs/src/modules.md`
- Modify if needed: `docs/src/index.md`

**Step 1: Confirm API documentation coverage**

Inspect `docs/src/api.md` and ensure the `ITensorCompat` source file is listed:

```markdown
## ITensorCompat

`Tensor4all.ITensorCompat` is an opt-in migration facade...

```@autodocs
Modules = [Tensor4all.ITensorCompat]
Pages = ["ITensorCompat.jl"]
Private = false
Order = [:type, :function]
```
```

If implementation later splits `ITensorCompat` into `src/ITensorCompat/*.jl`,
append all public source files to `Pages = [...]`.

**Step 2: Run autodocs coverage lint**

Run:

```bash
julia --startup-file=no --project=. scripts/check_autodocs_coverage.jl
```

Expected: pass. If it fails, add missing public source file paths to
`docs/src/api.md`.

**Step 3: Run focused test groups**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/core/index.jl"); include("test/core/tensor.jl"); include("test/tensornetworks/llim_rlim.jl"); include("test/itensorcompat/surface.jl")'
```

Expected: pass.

**Step 4: Run full test suite**

Run:

```bash
T4A_SKIP_HDF5_TESTS=1 julia --startup-file=no --project=. test/runtests.jl
```

Expected: pass.

If this host has the AMD EPYC Julia issue from `AGENTS.md`, use the documented
Docker workflow instead and record that in the final notes.

**Step 5: Build docs**

Run:

```bash
julia --startup-file=no --project=docs docs/make.jl
```

Expected: Documenter build succeeds.

**Step 6: Commit**

```bash
git add docs/src/api.md docs/src/modules.md docs/src/index.md
git commit -m "docs: document ITensorCompat facade"
```

If only `docs/src/api.md` changed, stage only that file.

---

## Final Handoff Checklist

- `git status --short` shows only pre-existing unrelated files, or is clean.
- Every task commit is present.
- Focused tests pass.
- Full tests pass, or the final note explains the documented platform blocker.
- `docs/make.jl` passes, or the final note explains the blocker.
- BubbleTeaCI migration can replace local Tensor4all-specific helpers with
  `Tensor4all.ITensorCompat.MPS` methods for the covered operations.
