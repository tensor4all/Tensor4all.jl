# ITensorCompat Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `prime/prime!`, `replaceprime`, `sim(siteinds,...)`, `MPO(Vector{Tensor})`, `maxlinkdim`, `data()`, and `Base.adjoint(::Index)` to `Tensor4all.jl`'s `ITensorCompat` module.

**Architecture:** `MPO(Vector{Tensor})`, `prime/prime!/replaceprime/sim(siteinds)` added to `src/ITensorCompat.jl`. `Base.adjoint(::Index)` added to `src/Core/Index.jl`. All new tests in one file.

**Tech Stack:** Julia, Tensor4all.jl (ITensorCompat, TensorNetworks, Core.Index)

---

## File Map

- Modify: `Tensor4all.jl/src/ITensorCompat.jl` — Layer 1 & 3 functions
- Modify: `Tensor4all.jl/src/Core/Index.jl` — `Base.adjoint(idx::Index)`
- Create: `Tensor4all.jl/test/itensorcompat/operator_overloads.jl` — new tests
- Modify: `Tensor4all.jl/test/runtests.jl` — add include

---

### Task 1: `maxlinkdim` and `data()` accessors

**Files:**
- Modify: `src/ITensorCompat.jl`
- Create: `test/itensorcompat/operator_overloads.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the failing test**

Create `test/itensorcompat/operator_overloads.jl`:
```julia
using Test
using Tensor4all

const IC = Tensor4all.ITensorCompat
const TN = Tensor4all.TensorNetworks

@testset "maxlinkdim" begin
    sites = [Index(2; tags=["s$n"]) for n in 1:3]
    blocks = [
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 3),
        reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2, 4),
        reshape([1.0, 0.0, 0.0, -1.0], 4, 2, 1),
    ]
    m = IC.MPS(blocks, sites)
    @test IC.maxlinkdim(m) == 4

    blocks_mpo = [
        reshape(collect(1.0:12.0), 1, 2, 2, 3),
        reshape(collect(1.0:24.0), 3, 2, 2, 1),
    ]
    input = [Index(2; tags=["in$n"]) for n in 1:2]
    output = [Index(2; tags=["out$n"]) for n in 1:2]
    w = IC.MPO(blocks_mpo, input, output)
    @test IC.maxlinkdim(w) == 3
end

@testset "data accessor" begin
    sites = [Index(2; tags=["s$n"]) for n in 1:2]
    blocks = [
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2),
        reshape([2.0, 0.0, 0.0, -1.0], 2, 2, 1),
    ]
    m = IC.MPS(blocks, sites)
    d = IC.data(m)
    @test d isa Vector{Tensor4all.Tensor}
    @test length(d) == 2

    input = [Index(2; tags=["in$n"]) for n in 1:1]
    output = [Index(2; tags=["out$n"]) for n in 1:1]
    blocks_mpo = [reshape([1.0, 0.0, 0.0, -1.0], 1, 2, 2, 1)]
    w = IC.MPO(blocks_mpo, input, output)
    dw = IC.data(w)
    @test dw isa Vector{Tensor4all.Tensor}
    @test length(dw) == 1

    tt = TN.TensorTrain([Tensor(1.0)])
    dtt = IC.data(tt)
    @test dtt isa Vector{Tensor4all.Tensor}
    @test length(dtt) == 1
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. test/itensorcompat/operator_overloads.jl`
Expected: 2 failures — `maxlinkdim` not defined, `data` not defined

  - [ ] **Step 3: Write minimal implementation with docstrings**

Add to `src/ITensorCompat.jl` before the closing `end`:

```julia
"""
    maxlinkdim(m::MPS) -> Int
    maxlinkdim(w::MPO) -> Int

Return the maximum bond dimension (link index dimension) of `m` or `w`.
Equivalent to `rank(m)` / `rank(w)`. Compatible with `ITensorMPS.maxlinkdim`.
"""
maxlinkdim(m::MPS) = rank(m)
maxlinkdim(w::MPO) = rank(w)

"""
    data(m::MPS) -> Vector{Tensor}
    data(w::MPO) -> Vector{Tensor}
    data(tt::TensorTrain) -> Vector{Tensor}

Return the underlying tensor storage vector. Compatible with `ITensors.data`.
"""
data(m::MPS) = m.tt.data
data(w::MPO) = w.tt.data
data(tt::TensorTrain) = tt.data
```

Add `maxlinkdim` and `data` to the export block:

```julia
export MPS, MPO
export siteinds, linkinds, linkdims, rank, maxlinkdim
export add, dag, dot, evaluate, inner, norm, replace_siteinds, replace_siteinds!, to_dense
export fixinds, suminds, projectinds, scalar
export orthogonalize!, truncate!
export prime, prime!, replaceprime
export data
```

- [ ] **Step 4: Add test to runtests.jl**

Add after the `bubbleteaci_workflow.jl` line in `test/runtests.jl`:

```julia
include("itensorcompat/operator_overloads.jl")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `julia --project=. test/itensorcompat/operator_overloads.jl`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/ITensorCompat.jl test/itensorcompat/operator_overloads.jl test/runtests.jl
git commit -m "feat: add maxlinkdim and data() to ITensorCompat"
```

---

### Task 2: `MPO(Vector{Tensor})` constructor

**Files:**
- Modify: `src/ITensorCompat.jl`
- Modify: `test/itensorcompat/operator_overloads.jl`

- [ ] **Step 1: Add failing test**

Append to `test/itensorcompat/operator_overloads.jl`:

```julia
@testset "MPO from Tensor vector" begin
    i = Index(2; tags=["i"])
    o = Index(2; tags=["o"])
    l1 = Index(3; tags=["Link", "l=1"])
    t1 = Tensor(reshape(collect(1.0:12.0), 2, 2, 3), [i, o, l1])
    t2 = Tensor(reshape(collect(1.0:6.0), 3, 2, 2), [l1, i, o])
    w = IC.MPO([t1, t2])
    @test length(w) == 2
    @test length(IC.siteinds(w)) == 2
    @test IC.siteinds(w)[1] == [i, o]
    @test IC.siteinds(w)[2] == [i, o]
    @test IC.data(w)[1] === t1

    # error case: tensor with wrong number of site indices
    t_bad = Tensor(reshape(collect(1.0:4.0), 2, 2), [i, o])
    @test_throws ArgumentError IC.MPO([t_bad])
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. test/itensorcompat/operator_overloads.jl`
Expected: FAIL — `MethodError: no method matching MPO(::Vector{Tensor})`

  - [ ] **Step 3: Write minimal implementation with docstring**

Add after the existing `MPO(blocks, input_sites, output_sites)` in `src/ITensorCompat.jl`:

```julia
"""
    MPO(tensors::Vector{Tensor})

Construct an MPO from an existing vector of `Tensor4all.Tensor` objects.
Each tensor must have exactly two site indices. Compatible with the ITensorMPS
`MPO(::Vector{ITensor})` constructor.
"""
function MPO(tensors::Vector{Tensor})
    isempty(tensors) && throw(ArgumentError("MPO tensor vector must not be empty"))
    groups = [Tensor.inds(t) for t in tensors]
    for (position, group) in pairs(groups)
        length(group) == 2 || throw(ArgumentError(
            "MPO expects exactly two site indices at tensor $position, got $(length(group))",
        ))
    end
    return MPO(TensorTrain(tensors))
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. test/itensorcompat/operator_overloads.jl`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ITensorCompat.jl test/itensorcompat/operator_overloads.jl
git commit -m "feat: add MPO(Vector{Tensor}) constructor to ITensorCompat"
```

---

### Task 3: `prime`, `prime!`, and `replaceprime` on MPS/MPO

**Files:**
- Modify: `src/ITensorCompat.jl`
- Modify: `test/itensorcompat/operator_overloads.jl`

- [ ] **Step 1: Add failing test**

Append to `test/itensorcompat/operator_overloads.jl`:

```julia
@testset "prime/prime! on MPS" begin
    sites = [Index(2; tags=["s$n"]) for n in 1:2]
    blocks = [
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2),
        reshape([2.0, 0.0, 0.0, -1.0], 2, 2, 1),
    ]
    m = IC.MPS(blocks, sites)

    mp = IC.prime(m)
    @test IC.siteinds(mp)[1].plev == 1
    @test IC.siteinds(m)[1].plev == 0  # original unchanged
    @test mp[1].data === m[1].data     # tensor data shared

    IC.prime!(m)
    @test IC.siteinds(m)[1].plev == 1

    mp2 = IC.prime(m, 2)
    @test IC.siteinds(mp2)[1].plev == 3
end

@testset "prime/prime! on MPO" begin
    i = Index(2; tags=["i"])
    o = Index(2; tags=["o"])
    l = Index(3; tags=["Link", "l=1"])
    t1 = Tensor(reshape(collect(1.0:12.0), 2, 2, 3), [i, o, l])
    t2 = Tensor(reshape(collect(1.0:6.0), 3, 2, 2), [l, i, o])
    w = IC.MPO([t1, t2])

    wp = IC.prime(w)
    @test all(idx.plev == 1 for idx in Iterators.flatten(IC.siteinds(wp)))
    @test all(idx.plev == 0 for idx in Iterators.flatten(IC.siteinds(w)))

    IC.prime!(w)
    @test all(idx.plev == 1 for idx in Iterators.flatten(IC.siteinds(w)))
end

@testset "replaceprime on MPS" begin
    sites = [Index(2; tags=["s$n"], plev=2) for n in 1:2]
    blocks = [
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2),
        reshape([2.0, 0.0, 0.0, -1.0], 2, 2, 1),
    ]
    m = IC.MPS(blocks, sites)
    mr = IC.replaceprime(m, 2 => 0)
    @test IC.siteinds(mr)[1].plev == 0
    @test IC.siteinds(m)[1].plev == 2   # original unchanged
end

@testset "replaceprime on MPO" begin
    i = Index(2; tags=["i"], plev=1)
    o = Index(2; tags=["o"], plev=1)
    l = Index(3; tags=["Link", "l=1"])
    t1 = Tensor(reshape(collect(1.0:12.0), 2, 2, 3), [i, o, l])
    t2 = Tensor(reshape(collect(1.0:6.0), 3, 2, 2), [l, i, o])
    w = IC.MPO([t1, t2])
    wr = IC.replaceprime(w, 1 => 0)
    @test all(idx.plev == 0 for idx in Iterators.flatten(IC.siteinds(wr)))
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. test/itensorcompat/operator_overloads.jl`
Expected: 4 failures — `prime`, `prime!`, `replaceprime` not defined

  - [ ] **Step 3: Write minimal implementation with docstrings**

Each new function must have a docstring. Add to `src/ITensorCompat.jl`:

```julia
"""
    prime(m::MPS, n::Integer=1) -> MPS
    prime(w::MPO, n::Integer=1) -> MPO

Return a copy of `m`/`w` with site index prime levels increased by `n`.
Tensor data is shared (not copied). Compatible with `ITensorMPS.prime`.
"""
function prime(m::MPS, n::Integer=1)
    sites = siteinds(m)
    primed = prime.(sites, Ref(n))
    tt = TensorNetworks.replace_siteinds(m.tt, sites, primed)
    return MPS(tt)
end

function prime(w::MPO, n::Integer=1)
    groups = TensorNetworks.siteinds(w.tt)
    flat_old = reduce(vcat, groups)
    flat_new = [prime(idx, n) for idx in flat_old]
    tt = TensorNetworks.replace_siteinds(w.tt, flat_old, flat_new)
    return MPO(tt)
end

function prime!(m::MPS, n::Integer=1)
    sites = siteinds(m)
    primed = prime.(sites, Ref(n))
    TensorNetworks.replace_siteinds!(m.tt, sites, primed)
    return m
end

function prime!(w::MPO, n::Integer=1)
    groups = TensorNetworks.siteinds(w.tt)
    flat_old = reduce(vcat, groups)
    flat_new = [prime(idx, n) for idx in flat_old]
    TensorNetworks.replace_siteinds!(w.tt, flat_old, flat_new)
    return w
end

"""
    replaceprime(m::MPS, pairs::Pair{Int,Int}...) -> MPS
    replaceprime(w::MPO, pairs::Pair{Int,Int}...) -> MPO

Replace prime levels in site indices of `m`/`w` according to `pairs`.
Each pair `old => new` replaces indices with `plev == old` to `plev == new`.
Compatible with `ITensorMPS.replaceprime`.
"""
function replaceprime(m::MPS, pairs::Pair{Int,Int}...)
    sites = siteinds(m)
    mapped = map(sites) do idx
        for (old, new) in pairs
            plev(idx) == old && return setprime(idx, new)
        end
        return idx
    end
    tt = TensorNetworks.replace_siteinds(m.tt, sites, mapped)
    return MPS(tt)
end

function replaceprime(w::MPO, pairs::Pair{Int,Int}...)
    groups = TensorNetworks.siteinds(w.tt)
    mapped = map(groups) do group
        map(group) do idx
            for (old, new) in pairs
                plev(idx) == old && return setprime(idx, new)
            end
            return idx
        end
    end
    flat_old = reduce(vcat, groups)
    flat_new = reduce(vcat, mapped)
    tt = TensorNetworks.replace_siteinds(w.tt, flat_old, flat_new)
    return MPO(tt)
end
```

Note: `prime` for Index and `setprime` for Index are already available from Tensor4all.Core (used via `import ..Tensor4all: ...`). But since ITensorCompat doesn't import them yet, add them to the existing import line:

```julia
import ..Tensor4all: Index, Tensor, dim, inds, rank, scalar, prime, setprime
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. test/itensorcompat/operator_overloads.jl`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ITensorCompat.jl test/itensorcompat/operator_overloads.jl
git commit -m "feat: add prime/prime!/replaceprime to ITensorCompat"
```

---

### Task 4: `sim(siteinds, ...)` on MPS/MPO

**Files:**
- Modify: `src/ITensorCompat.jl`
- Modify: `test/itensorcompat/operator_overloads.jl`

- [ ] **Step 1: Add failing test**

Append to `test/itensorcompat/operator_overloads.jl`:

```julia
@testset "sim(siteinds, ...) on MPS" begin
    sites = [Index(2; tags=["s$n"], plev=n) for n in 1:2]
    blocks = [
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2),
        reshape([2.0, 0.0, 0.0, -1.0], 2, 2, 1),
    ]
    m = IC.MPS(blocks, sites)
    new_sites = IC.sim(IC.siteinds, m)
    @test new_sites isa Vector{Tensor4all.Index}
    @test length(new_sites) == 2
    @test dim.(new_sites) == dim.(sites)
    @test Tensor4all.tags.(new_sites) == Tensor4all.tags.(sites)
    @test plev.(new_sites) == plev.(sites)
    @test id.(new_sites) != id.(sites)  # fresh IDs
end

@testset "sim(siteinds, ...) on MPO" begin
    i = Index(2; tags=["i"])
    o = Index(2; tags=["o"])
    l = Index(3; tags=["Link", "l=1"])
    t1 = Tensor(reshape(collect(1.0:12.0), 2, 2, 3), [i, o, l])
    t2 = Tensor(reshape(collect(1.0:6.0), 3, 2, 2), [l, i, o])
    w = IC.MPO([t1, t2])
    new_sites = IC.sim(IC.siteinds, w)
    @test new_sites isa Vector{Vector{Tensor4all.Index}}
    @test length(new_sites) == 2
    @test all(length(g) == 2 for g in new_sites)
    @test id.(vcat(new_sites...)) != id.(vcat(i, o, i, o))
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. test/itensorcompat/operator_overloads.jl`
Expected: FAIL — `MethodError: no method matching sim(::typeof(siteinds), ::MPS)`

  - [ ] **Step 3: Write minimal implementation with docstring**

Add to `src/ITensorCompat.jl`:

```julia
"""
    sim(::typeof(siteinds), m::MPS) -> Vector{Index}
    sim(::typeof(siteinds), w::MPO) -> Vector{Vector{Index}}

Return cloned site indices with fresh IDs but matching dimensions, tags, and
prime levels. Compatible with ITensorMPS's `sim(siteinds, mps)` pattern.
"""
sim(::typeof(siteinds), m::MPS) = [sim(idx) for idx in siteinds(m)]
```

For MPO, `TensorNetworks.siteinds(w.tt)` returns `Vector{Vector{Index}}`. We need to map `sim` over each inner vector:

```julia
sim(::typeof(siteinds), w::MPO) = [[sim(idx) for idx in group] for group in TensorNetworks.siteinds(w.tt)]
```

Add `sim` to the import from Tensor4all:
```julia
import ..Tensor4all: Index, Tensor, dim, inds, rank, scalar, prime, setprime, sim
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. test/itensorcompat/operator_overloads.jl`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ITensorCompat.jl test/itensorcompat/operator_overloads.jl
git commit -m "feat: add sim(siteinds, ...) to ITensorCompat"
```

---

### Task 5: `Base.adjoint(::Index)` — `'` operator on Index

**Files:**
- Modify: `src/Core/Index.jl`
- Modify: `test/itensorcompat/operator_overloads.jl`

- [ ] **Step 1: Write the failing test**

Append to `test/itensorcompat/operator_overloads.jl`:

```julia
@testset "Index prime via ' operator" begin
    idx = Index(4; tags=["x"], plev=0)
    idx1 = idx'
    @test Tensor4all.plev(idx1) == 1
    @test Tensor4all.dim(idx1) == 4
    @test Tensor4all.tags(idx1) == ["x"]
    @test Tensor4all.id(idx1) == Tensor4all.id(idx)

    idx2 = idx''
    @test Tensor4all.plev(idx2) == 2

    @test Tensor4all.plev(idx) == 0  # original unchanged
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. test/itensorcompat/operator_overloads.jl`
Expected: FAIL — `MethodError: no method matching adjoint(::Index)`

  - [ ] **Step 3: Write minimal implementation with docstring**

Add at the end of Index.jl's Index operations section (after `setprime`, before `Base.:(==)` or wherever is appropriate) in `src/Core/Index.jl`:

```julia
"""
    Base.adjoint(i::Index)

Return `prime(i)`, enabling the `idx'` sugar syntax for creating a primed copy
of an index. Compatible with ITensors.jl's `idx'` convention.
"""
Base.adjoint(i::Index) = prime(i)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. test/itensorcompat/operator_overloads.jl`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `julia --project=. test/runtests.jl`
Expected: all tests pass (including existing ones)

- [ ] **Step 6: Commit**

```bash
git add src/Core/Index.jl test/itensorcompat/operator_overloads.jl
git commit -m "feat: add Base.adjoint(::Index) for prime sugar syntax"
```
