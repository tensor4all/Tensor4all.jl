# Issue #66 BubbleTeaCI Follow-up Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the BubbleTeaCI follow-up API surface from issue #66 as one Tensor4all.jl PR with phased, independently testable commits.

**Architecture:** Keep `TensorNetworks.TensorTrain` as the primary indexed chain type and add generic behavior in Core / TensorNetworks before adding the opt-in `ITensorCompat` facade. Julia owns mutation semantics, index metadata rewriting, slicing, summation, wrapper construction, and compatibility keyword mapping; Rust is used only for existing tensor storage and tensor-network kernels. Structured diagonal payload support is consumed through the tensor4all-rs C API after the pin is bumped.

**Tech Stack:** Julia 1.11, Tensor4all Core/TensorNetworks/SimpleTT modules, existing tensor4all-rs C API, Documenter.jl, `Test`.

---

## Inputs and Ground Rules

- Start from `origin/main` at `e5dc2b2e859839a28fbdb39b3d94193dd960f334`.
- Read `docs/plans/2026-04-23-issue-66-bubbleteaci-followup-spec-design.md` before implementing.
- Keep this as one PR, but commit each task separately.
- Do not add new Rust C API functions for chain mutation, index replacement, slicing, summation, or ITensors-style compatibility wrappers.
- If structured tensor C API support is not already present on the pinned tensor4all-rs commit, merge the tensor4all-rs PR first and then bump `deps/TENSOR4ALL_RS_PIN`.
- When testing against a local Rust checkout from this worktree, use:

```bash
TENSOR4ALL_RS_PATH=/home/shinaoka/tensor4all/tensor4all-rs julia --startup-file=no --project=. deps/build.jl
```

- If running on the AMD EPYC host described in `AGENTS.md`, use the documented Docker workaround for final full-suite verification.

## Task 1: Pin and Wire Structured Tensor C API Support

**Affected files:**
- Modify: `deps/TENSOR4ALL_RS_PIN`
- Modify: `src/TensorNetworks/backend/capi.jl`
- Modify: `src/TensorNetworks/backend/tensors.jl`
- Test: `test/core/tensor_storage.jl`
- Modify: `test/runtests.jl`

**What to do:**

Add Julia access to the existing tensor4all-rs structured tensor APIs:

- `t4a_tensor_clone`
- `t4a_tensor_storage_kind`
- `t4a_tensor_payload_rank`
- `t4a_tensor_payload_len`
- `t4a_tensor_payload_dims`
- `t4a_tensor_payload_strides`
- `t4a_tensor_axis_classes`
- `t4a_tensor_copy_payload_f64`
- `t4a_tensor_copy_payload_c64`
- `t4a_tensor_new_diag_f64`
- `t4a_tensor_new_diag_c64`
- `t4a_tensor_new_structured_f64`
- `t4a_tensor_new_structured_c64`

Keep the low-level FFI helpers next to the current tensor handle functions in
`src/TensorNetworks/backend/tensors.jl` for this PR. Core public functions can
reach them through the existing `_tensor_networks_module()` pattern already used
by `contract`, `qr`, and `svd`.

Add C enum constants in `src/TensorNetworks/backend/capi.jl`:

```julia
const _T4A_STORAGE_KIND_DENSE = Cint(0)
const _T4A_STORAGE_KIND_DIAGONAL = Cint(1)
const _T4A_STORAGE_KIND_STRUCTURED = Cint(2)
```

Add helpers with this shape:

```julia
function _clone_tensor_handle(ptr::Ptr{Cvoid})
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(_t4a(:t4a_tensor_clone), Cint, (Ptr{Cvoid}, Ref{Ptr{Cvoid}}), ptr, out)
    _check_backend_status(status, "cloning backend tensor")
    return out[]
end

function _storage_kind_from_handle(ptr::Ptr{Cvoid})
    out = Ref{Cint}(0)
    status = ccall(_t4a(:t4a_tensor_storage_kind), Cint, (Ptr{Cvoid}, Ref{Cint}), ptr, out)
    _check_backend_status(status, "querying backend tensor storage kind")
    return out[]
end
```

Use the established two-call buffer pattern from `_tensor_indices_from_handle`
for payload dimensions, strides, axis classes, and payload values.

**Tests:**

Create `test/core/tensor_storage.jl` with failing tests first:

```julia
using Test
using Tensor4all

@testset "structured tensor storage C API" begin
    i = Index(3; tags=["i"])
    j = Index(3; tags=["j"])
    d = Tensor4all.diagtensor([1.0, 2.0, 4.0], [i, j])

    @test Tensor4all.storage_kind(d) == :diagonal
    @test Tensor4all.payload_rank(d) == 1
    @test Tensor4all.payload_dims(d) == [3]
    @test Tensor4all.axis_classes(d) == [0, 0]
    @test Tensor4all.payload(d) == [1.0, 2.0, 4.0]
    @test Array(d, i, j) == [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0]
end
```

Expected initial failure: `diagtensor` / storage metadata functions are
undefined.

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/core/tensor_storage.jl")'
```

After implementation, run:

```bash
julia --startup-file=no --project=. -e 'include("test/core/tensor_storage.jl")'
```

**Commit:**

```bash
git add deps/TENSOR4ALL_RS_PIN src/TensorNetworks/backend/capi.jl src/TensorNetworks/backend/tensors.jl test/core/tensor_storage.jl test/runtests.jl
git commit -m "feat: expose structured tensor storage c api"
```

## Task 2: Establish Tensor Backend Handle Ownership

**Affected files:**
- Modify: `src/Core/Tensor.jl`
- Modify: `src/TensorNetworks/backend/tensors.jl`
- Test: `test/core/tensor_storage.jl`
- Test: `test/core/tensor.jl`

**What to do:**

Define a clear ownership convention for `Tensor.backend_handle` before
structured tensors are used broadly. The current field is a raw pointer and
existing conversion paths rebuild dense backend handles from `tensor.data`.
Structured tensors need compact storage to survive roundtrips.

Recommended implementation:

- Introduce a small handle wrapper in `src/Core/Tensor.jl`:

```julia
mutable struct BackendTensorHandle
    ptr::Ptr{Cvoid}
    owned::Bool
end
```

- Keep the public `Tensor(...; backend_handle=...)` keyword accepting
  `nothing`, `Ptr{Cvoid}`, or `BackendTensorHandle`.
- Treat raw `Ptr{Cvoid}` inputs as borrowed, to keep existing tests with
  `Ptr{Cvoid}(1)` valid.
- Treat handles returned from `t4a_tensor_new_*` and `t4a_tensor_clone` as owned
  and attach a finalizer that calls `TensorNetworks._release_tensor_handle`.
- Add an internal `backend_handle_ptr(t::Tensor)` helper that returns
  `C_NULL` when no handle exists.
- Update `_new_tensor_handle(tensor, scalar_kind)` so tensors with a valid
  backend handle use `t4a_tensor_clone` instead of rebuilding from dense data.
  Fall back to the current dense constructor when no handle exists.

Do not attempt a full storage-polymorphic `Tensor.data` refactor in this PR.
For structured tensors, keep `data` as dense materialization for Julia array
compatibility while preserving compact storage in the backend handle.

**Tests:**

Extend `test/core/tensor_storage.jl`:

```julia
@testset "structured handle survives backend roundtrip" begin
    i = Index(2; tags=["i"])
    j = Index(2; tags=["j"])
    d = Tensor4all.delta(i, j)
    clone = Tensor4all.contract(d, Tensor4all.delta(j, i))
    @test Tensor4all.storage_kind(clone) == :diagonal
    @test Tensor4all.payload(clone) == [1.0, 1.0]
end
```

If tensor4all-rs structured contraction still densifies some contractions,
weaken this test to assert that `_new_tensor_handle` preserves the diagonal
storage kind when cloning a standalone tensor handle, and add a TODO comment
pointing at the structured contraction Rust issue.

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/core/tensor_storage.jl")'
julia --startup-file=no --project=. -e 'include("test/core/tensor.jl")'
```

**Commit:**

```bash
git add src/Core/Tensor.jl src/TensorNetworks/backend/tensors.jl test/core/tensor_storage.jl test/core/tensor.jl
git commit -m "feat: preserve tensor backend storage handles"
```

## Task 3: Add Core Tag, Scalar, and Tensor Constructor Conveniences

**Affected files:**
- Modify: `src/Core/Index.jl`
- Modify: `src/Core/Tensor.jl`
- Modify: `src/Tensor4all.jl`
- Test: `test/core/index.jl`
- Test: `test/core/tensor.jl`

**What to do:**

Add generic Core helpers needed by the later compatibility facade:

- `tagstring(index::Index)`
- `tagstring(tags::AbstractVector{<:AbstractString})`
- `Base.eltype(t::Tensor)`
- `Tensor(data, inds::Index...)`
- `Tensor(value::Number)`
- `scalar(t::Tensor)`
- `const ITensor = Tensor`

Use `tagstring` from `Base.show(::Index)`:

```julia
tagstring(tags::AbstractVector{<:AbstractString}) =
    isempty(tags) ? "-" : join(String.(tags), ",")
tagstring(i::Index) = tagstring(tags(i))
```

Add scalar validation:

```julia
function scalar(t::Tensor)
    rank(t) == 0 || throw(ArgumentError("scalar requires a rank-0 Tensor, got rank $(rank(t))"))
    return t.data[]
end
```

Export the new public symbols from `src/Tensor4all.jl`.

**Tests:**

Extend `test/core/index.jl`:

```julia
@testset "tagstring" begin
    i = Index(2; tags=["alpha", "site=1"])
    @test Tensor4all.tagstring(i) == "alpha,site=1"
    @test Tensor4all.tagstring(String[]) == "-"
    @test occursin("alpha,site=1", sprint(show, i))
end
```

Extend `test/core/tensor.jl`:

```julia
@testset "Tensor scalar and ITensor conveniences" begin
    i = Index(2; tags=["i"])
    t = Tensor([1.0, 2.0], i)
    @test inds(t) == [i]
    @test eltype(t) == Float64
    @test Tensor4all.ITensor === Tensor4all.Tensor
    @test Tensor4all.scalar(Tensor(3.5)) == 3.5
    @test_throws ArgumentError Tensor4all.scalar(t)
end
```

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/core/index.jl")'
julia --startup-file=no --project=. -e 'include("test/core/tensor.jl")'
```

**Commit:**

```bash
git add src/Core/Index.jl src/Core/Tensor.jl src/Tensor4all.jl test/core/index.jl test/core/tensor.jl
git commit -m "feat: add core tag and scalar conveniences"
```

## Task 4: Add Structured Diagonal and Identity Constructors

**Affected files:**
- Create: `src/Core/TensorStorage.jl`
- Modify: `src/Tensor4all.jl`
- Modify: `docs/src/api.md`
- Test: `test/core/tensor_storage.jl`

**What to do:**

Move public storage metadata and constructors into a focused Core source file:

- `storage_kind(t::Tensor)::Symbol`
- `payload_rank(t::Tensor)::Int`
- `payload_len(t::Tensor)::Int`
- `payload_dims(t::Tensor)::Vector{Int}`
- `payload_strides(t::Tensor)::Vector{Int}`
- `axis_classes(t::Tensor)::Vector{Int}`
- `payload(t::Tensor)::Vector`
- `diagtensor(values, inds::AbstractVector{<:Index})`
- `delta(i::Index, j::Index, more::Index...; T=Float64)`
- `identity_tensor(i::Index, j::Index; T=Float64)`

Include the file after `Core/Tensor.jl`:

```julia
include("Core/TensorStorage.jl")
```

Update `docs/src/api.md` Core autodocs `Pages = [...]` with
`"Core/TensorStorage.jl"`.

Constructor semantics:

- `diagtensor(values, inds)` validates all logical index dimensions match
  `length(values)` unless the rank is 0.
- `delta(i, j, more...)` is `diagtensor(ones(T, dim(i)), [i, j, more...])`.
- `identity_tensor(i, j)` is an alias for `delta(i, j)`.
- Throw `DimensionMismatch` for mismatched dimensions.
- Preserve backend Rust error messages through `_check_backend_status`.

**Tests:**

Extend `test/core/tensor_storage.jl` with:

```julia
@testset "delta and identity constructors" begin
    i = Index(3; tags=["i"])
    j = Index(3; tags=["j"])
    k = Index(3; tags=["k"])
    bad = Index(2; tags=["bad"])

    d2 = Tensor4all.delta(i, j)
    @test Tensor4all.storage_kind(d2) == :diagonal
    @test Tensor4all.axis_classes(d2) == [0, 0]
    @test Tensor4all.payload(d2) == ones(3)

    d3 = Tensor4all.delta(i, j, k)
    @test Tensor4all.axis_classes(d3) == [0, 0, 0]
    @test Tensor4all.payload(d3) == ones(3)

    @test Array(Tensor4all.identity_tensor(i, j), i, j) == Matrix{Float64}(I, 3, 3)
    @test_throws DimensionMismatch Tensor4all.delta(i, bad)
end
```

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/core/tensor_storage.jl")'
julia --startup-file=no --project=docs docs/make.jl
```

**Commit:**

```bash
git add src/Core/TensorStorage.jl src/Tensor4all.jl docs/src/api.md test/core/tensor_storage.jl
git commit -m "feat: add structured diagonal tensor constructors"
```

## Task 5: Public TensorTrain Mutation Helpers

**Affected files:**
- Create: `src/TensorNetworks/mutation.jl`
- Modify: `src/TensorNetworks.jl`
- Modify: `src/TensorNetworks/types.jl`
- Modify: `docs/src/api.md`
- Test: `test/tensornetworks/llim_rlim.jl`
- Test: `test/tensornetworks/tensortrain.jl`

**What to do:**

Move `Base.setindex!` out of `src/TensorNetworks/types.jl` into the new
focused mutation file, then add:

- `invalidate_canonical!(tt)`
- `invalidate_canonical!(tt, i)`
- `replaceblock!(tt, i, tensor)`
- `insert_site!(tt, position, tensor)`
- `delete_site!(tt, position)`
- `Base.insert!(tt, position, tensor)`
- `Base.deleteat!(tt, position)`
- `Base.push!(tt, tensor)`
- `Base.pushfirst!(tt, tensor)`

Keep `setindex!` return behavior compatible with Julia and the current code:
return the assigned tensor. Make `replaceblock!`, `insert_site!`,
`delete_site!`, `insert!`, `deleteat!`, `push!`, and `pushfirst!` return the
mutated `TensorTrain`.

Local invalidation:

```julia
function invalidate_canonical!(tt::TensorTrain, i::Integer)
    1 <= i <= length(tt) || throw(BoundsError(tt.data, i))
    tt.llim = min(tt.llim, Int(i) - 1)
    tt.rlim = max(tt.rlim, Int(i) + 1)
    return tt
end
```

Full invalidation:

```julia
function invalidate_canonical!(tt::TensorTrain)
    tt.llim = 0
    tt.rlim = length(tt) + 1
    return tt
end
```

Export public helpers from `src/TensorNetworks.jl` and update
`docs/src/api.md` `Pages = [...]` with `"TensorNetworks/mutation.jl"`.

**Tests:**

Extend `test/tensornetworks/llim_rlim.jl`:

```julia
@testset "public TensorTrain mutation helpers" begin
    i1 = Index(2; tags=["s1"])
    i2 = Index(2; tags=["s2"])
    l = Index(2; tags=["Link", "l=1"])
    t1 = Tensor(randn(2, 2), [i1, l])
    t2 = Tensor(randn(2, 2), [l, i2])
    tt = TN.TensorTrain([t1, t2], 1, 2)

    @test TN.invalidate_canonical!(tt, 1) === tt
    @test (tt.llim, tt.rlim) == (0, 2)

    tt = TN.TensorTrain([t1, t2], 1, 2)
    @test TN.replaceblock!(tt, 2, t2) === tt
    @test (tt.llim, tt.rlim) == (1, 3)

    newsite = Index(2; tags=["s3"])
    newtensor = Tensor(randn(2), [newsite])
    @test insert!(tt, 2, newtensor) === tt
    @test (tt.llim, tt.rlim) == (0, length(tt) + 1)
    @test deleteat!(tt, 2) === tt
    @test (tt.llim, tt.rlim) == (0, length(tt) + 1)
end
```

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/tensornetworks/llim_rlim.jl")'
julia --startup-file=no --project=. -e 'include("test/tensornetworks/tensortrain.jl")'
```

**Commit:**

```bash
git add src/TensorNetworks.jl src/TensorNetworks/types.jl src/TensorNetworks/mutation.jl docs/src/api.md test/tensornetworks/llim_rlim.jl test/tensornetworks/tensortrain.jl
git commit -m "feat: add tensortrain mutation helpers"
```

## Task 6: Make Site Index Replacement Metadata-Only

**Affected files:**
- Modify: `src/TensorNetworks/site_helpers.jl`
- Test: `test/tensornetworks/index_queries.jl`
- Test: `test/core/tensor_storage.jl`

**What to do:**

Update `_replace_tensor_indices` and `replace_siteinds!` to use Tensor-level
`replaceinds!` where possible. Avoid constructing new dense tensors for
mutating site replacement.

Desired mutating behavior:

```julia
function _replace_tensor_indices!(tensor::Tensor, replacements::Dict{Index, Index})
    old = Index[]
    new = Index[]
    for index in inds(tensor)
        haskey(replacements, index) || continue
        push!(old, index)
        push!(new, replacements[index])
    end
    isempty(old) && return tensor
    return replaceinds!(tensor, old, new)
end
```

`replace_siteinds!` should not alter `tt.llim` or `tt.rlim`.

For non-mutating `replace_siteinds`, continue returning a non-aliased train,
but preserve storage metadata and backend handles by copying tensors through a
metadata-preserving helper. Do not use `Tensor(tensor.data, inds(tensor))`
without `backend_handle`.

**Tests:**

Update tests in `test/tensornetworks/index_queries.jl` that currently assert
`replaced[2].data !== fixture.tt[2].data`. The new contract should assert
metadata and values, not forced dense-array copying:

```julia
@test Array(replaced[2], inds(replaced[2])...) == Array(fixture.tt[2], inds(fixture.tt[2])...)
```

Add canonical preservation checks:

```julia
fixture = mps_like_fixture()
fixture.tt.llim = 1
fixture.tt.rlim = 3
TN.replace_siteinds!(fixture.tt, [fixture.sites[1]], [Index(2; tags=["y", "y=1"])])
@test (fixture.tt.llim, fixture.tt.rlim) == (1, 3)
```

Add structured storage preservation:

```julia
@testset "replace_siteinds preserves diagonal storage" begin
    i = Index(2; tags=["x", "x=1"])
    j = Index(2; tags=["y", "y=1"])
    jp = Index(2; tags=["z", "z=1"])
    tt = TN.TensorTrain([Tensor4all.delta(i, j)], 0, 2)
    TN.replace_siteinds!(tt, [j], [jp])
    @test Tensor4all.storage_kind(tt[1]) == :diagonal
    @test Tensor4all.axis_classes(tt[1]) == [0, 0]
end
```

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/tensornetworks/index_queries.jl")'
julia --startup-file=no --project=. -e 'include("test/core/tensor_storage.jl")'
```

**Commit:**

```bash
git add src/TensorNetworks/site_helpers.jl test/tensornetworks/index_queries.jl test/core/tensor_storage.jl
git commit -m "feat: preserve metadata during site index replacement"
```

## Task 7: Add Lazy Fix, Sum, Project, and One-Hot Primitives

**Affected files:**
- Create: `src/Core/IndexOps.jl`
- Create: `src/TensorNetworks/index_ops.jl`
- Modify: `src/Tensor4all.jl`
- Modify: `src/TensorNetworks.jl`
- Modify: `docs/src/api.md`
- Test: `test/core/index_ops.jl`
- Test: `test/tensornetworks/index_ops.jl`
- Modify: `test/runtests.jl`

**What to do:**

Add Core operations:

- `onehot(index => value; T=Float64)`
- `fixinds(t::Tensor, replacements::Pair{Index,<:Integer}...)`
- `suminds(t::Tensor, indices::Index...)`
- `projectinds(t::Tensor, replacements::Pair{Index,<:AbstractVector{<:Integer}}...)`

`onehot` should be a lightweight object. It materializes only when the user
calls `Tensor(onehot(...))` or `Array(Tensor(onehot(...)), index)`. Contracting
a Tensor with a onehot should slice instead of allocating a dense one-hot
operand.

Add TensorTrain operations:

- `fixinds(tt::TensorTrain, replacements...)`
- `suminds(tt::TensorTrain, indices...)`
- `projectinds(tt::TensorTrain, replacements...)`
- optional mutating variants only if the implementation stays small and clear.

For the non-mutating TensorTrain path:

1. copy the TensorTrain;
2. find affected site tensors with `findsite` / `findsites`;
3. apply Tensor-level operations to affected tensors;
4. remove or replace indices in local tensors;
5. locally invalidate affected positions.

Do not use dense one-hot vectors internally for `fixinds`.

**Tests:**

Create `test/core/index_ops.jl`:

```julia
using Test
using Tensor4all

@testset "Tensor index fixing and summation" begin
    i = Index(2; tags=["i"])
    j = Index(3; tags=["j"])
    t = Tensor(reshape(collect(1.0:6.0), 2, 3), [i, j])

    @test inds(Tensor4all.onehot(i => 2)) == [i]
    @test Array(Tensor(Tensor4all.onehot(i => 2)), i) == [0.0, 1.0]
    @test Array(Tensor4all.fixinds(t, i => 2), j) == Array(t, i, j)[2, :]
    @test Array(Tensor4all.suminds(t, i), j) == vec(sum(Array(t, i, j); dims=1))

    p = Tensor4all.projectinds(t, i => [2])
    @test dims(p) == (1, 3)
    @test dim(inds(p)[1]) == 1
end
```

Create `test/tensornetworks/index_ops.jl` using a two-site MPS fixture. Compare
`fixinds`, `suminds`, and `projectinds` results against `to_dense`.

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/core/index_ops.jl")'
julia --startup-file=no --project=. -e 'include("test/tensornetworks/index_ops.jl")'
```

**Commit:**

```bash
git add src/Core/IndexOps.jl src/TensorNetworks/index_ops.jl src/Tensor4all.jl src/TensorNetworks.jl docs/src/api.md test/core/index_ops.jl test/tensornetworks/index_ops.jl test/runtests.jl
git commit -m "feat: add lazy index fix sum project primitives"
```

## Task 8: Add MPS Identity Insertion Helpers

**Affected files:**
- Create: `src/TensorNetworks/identity_helpers.jl`
- Modify: `src/TensorNetworks.jl`
- Modify: `docs/src/api.md`
- Test: `test/tensornetworks/identity_helpers.jl`
- Modify: `test/runtests.jl`

**What to do:**

Add public helpers:

- `identity_link_tensor(left::Index, right::Index, site::Index; T=Float64)`
- `insert_identity!(tt::TensorTrain, newsite::Index, position::Integer; T=nothing)`

Semantics:

- `position == 0` inserts before the first site.
- `position == length(tt)` inserts after the last site.
- `0 <= position <= length(tt)` or throw `ArgumentError`.
- Interior insertion uses the shared link between `tt[position]` and
  `tt[position + 1]`, creates a new link with the same dimension, rewires the
  right tensor to the new link, and inserts a structured identity/copy tensor.
- Boundary insertion creates a dimension-1 link and updates the adjacent block
  through public TensorTrain mutation helpers.
- Full canonical invalidation is acceptable for topology changes.

Use `delta` / `diagtensor` internally where possible. Avoid `diagm`.

**Tests:**

Create `test/tensornetworks/identity_helpers.jl`:

```julia
using Test
using Tensor4all

const TN_ID = Tensor4all.TensorNetworks

@testset "insert_identity!" begin
    sites = [Index(2; tags=["x", "x=$n"]) for n in 1:2]
    link = Index(2; tags=["Link", "l=1"])
    tt = TN_ID.TensorTrain([
        Tensor([1.0 0.0; 0.0 1.0], [sites[1], link]),
        Tensor([2.0 0.0; 0.0 -1.0], [link, sites[2]]),
    ], 1, 2)

    newsite = Index(2; tags=["x", "x=mid"])
    @test TN_ID.insert_identity!(tt, newsite, 1) === tt
    @test length(tt) == 3
    @test (tt.llim, tt.rlim) == (0, length(tt) + 1)
    @test newsite in only.(TN_ID.siteinds(tt)[2:2])
end
```

Add a dense/evaluate preservation check when the inserted site is fixed or
summed out, comparing against the original dense tensor.

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/tensornetworks/identity_helpers.jl")'
```

**Commit:**

```bash
git add src/TensorNetworks/identity_helpers.jl src/TensorNetworks.jl docs/src/api.md test/tensornetworks/identity_helpers.jl test/runtests.jl
git commit -m "feat: add tensortrain identity insertion helpers"
```

## Task 9: Add ITensorCompat Module Skeleton

**Affected files:**
- Create: `src/ITensorCompat.jl`
- Modify: `src/Tensor4all.jl`
- Modify: `docs/src/api.md`
- Test: `test/itensorcompat/surface.jl`
- Modify: `test/runtests.jl`

**What to do:**

Create an opt-in `Tensor4all.ITensorCompat` module with semantic wrappers:

```julia
module ITensorCompat

using ..Tensor4all: Index, Tensor, inds, rank, dim
using ..Tensor4all.TensorNetworks

mutable struct MPS
    tt::TensorNetworks.TensorTrain
end

mutable struct MPO
    tt::TensorNetworks.TensorTrain
end

end
```

Top-level `Tensor4all.jl` should include and export only the module name:

```julia
include("ITensorCompat.jl")
export ITensorCompat
```

Add validation helpers:

- `MPS(tt)` accepts one site-like index per tensor.
- `MPS(TensorTrain(Tensor[]))` is accepted as an empty/scalar MPS.
- `MPO(tt)` accepts two site-like indices per tensor.
- Reject invalid structures with `ArgumentError` including tensor position and
  actual site arity.

Add basic wrapper methods:

- `Base.length`
- `Base.iterate`
- `Base.getindex`
- `Base.setindex!`
- `siteinds(::MPS)::Vector{Index}`
- `siteinds(::MPO)::Vector{Vector{Index}}`
- `linkinds`, `linkdims`, `rank`, `Base.eltype`

**Tests:**

Create `test/itensorcompat/surface.jl`:

```julia
using Test
using Tensor4all

const IC = Tensor4all.ITensorCompat
const TN_IC = Tensor4all.TensorNetworks

@testset "ITensorCompat wrapper validation" begin
    s1 = Index(2; tags=["s", "s=1"])
    s2 = Index(2; tags=["s", "s=2"])
    link = Index(1; tags=["Link", "l=1"])
    tt = TN_IC.TensorTrain([
        Tensor(ones(2, 1), [s1, link]),
        Tensor(ones(1, 2), [link, s2]),
    ])

    m = IC.MPS(tt)
    @test length(m) == 2
    @test IC.siteinds(m) == [s1, s2]
    @test m[1] === tt[1]

    bad = TN_IC.TensorTrain([Tensor(ones(2, 2), [s1, s2])])
    @test_throws ArgumentError IC.MPS(bad)
end
```

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/itensorcompat/surface.jl")'
julia --startup-file=no --project=docs docs/make.jl
```

**Commit:**

```bash
git add src/ITensorCompat.jl src/Tensor4all.jl docs/src/api.md test/itensorcompat/surface.jl test/runtests.jl
git commit -m "feat: add itensorcompat wrapper module"
```

## Task 10: ITensorCompat MPS Operations, Cutoff Mapping, and Scalar Behavior

**Affected files:**
- Modify: `src/ITensorCompat.jl`
- Test: `test/itensorcompat/surface.jl`

**What to do:**

Forward MPS operations to `TensorNetworks`:

- `inner`, `dot`, `norm`
- `add`
- `Base.:+`
- scalar `*` and `/`
- `dag`
- `replace_siteinds!`
- `replace_siteinds`
- `fixinds`, `suminds`, `projectinds`
- `to_dense`
- `evaluate`
- `scalar`

Add mutating canonical operations:

- `orthogonalize!(m::MPS, site::Integer; kwargs...)`
- `truncate!(m::MPS; cutoff=0.0, maxdim=0, kwargs...)`

Cutoff mapping:

```julia
const ITENSORS_CUTOFF_POLICY = TensorNetworks.SvdTruncationPolicy(
    measure = :squared_value,
    rule = :discarded_tail_sum,
)
```

Reject native Tensor4all truncation controls:

```julia
function _reject_native_truncation_kwargs(kwargs)
    bad = intersect(keys(kwargs), (:threshold, :svd_policy))
    isempty(bad) || throw(ArgumentError(
        "ITensorCompat truncation is cutoff-only; got native Tensor4all keyword(s) $(Tuple(bad)). Use TensorNetworks for threshold or svd_policy.",
    ))
end
```

`truncate!(m; cutoff=0.0, maxdim=0)` should require `cutoff > 0 || maxdim > 0`,
then assign back into `m.tt` and return `m`.

Scalar / empty MPS:

- Represent a scalar MPS as `MPS(TensorNetworks.TensorTrain([Tensor(value)]))`
  for computation, but expose `length(m) == 0` and `siteinds(m) == Index[]`
  when the wrapped train has one rank-0 tensor.
- `scalar(m)` returns `scalar(only(m.tt.data))` for scalar MPS.
- `to_dense(m)` returns the rank-0 tensor for scalar MPS.
- `evaluate(m)` returns the scalar for scalar MPS.

**Tests:**

Extend `test/itensorcompat/surface.jl`:

```julia
@testset "ITensorCompat cutoff-only truncation" begin
    m = IC.MPS(make_test_mps_for_itensorcompat())
    @test IC.truncate!(m; cutoff=1e-10) === m
    @test_throws ArgumentError IC.truncate!(m; threshold=1e-10)
    @test_throws ArgumentError IC.truncate!(m; svd_policy=TN_IC.SvdTruncationPolicy())
end

@testset "ITensorCompat scalar MPS" begin
    m = IC.MPS(TN_IC.TensorTrain([Tensor(3.5)]))
    @test length(m) == 0
    @test IC.siteinds(m) == Index[]
    @test IC.scalar(m) == 3.5
    @test Tensor4all.scalar(IC.to_dense(m)) == 3.5
    @test IC.evaluate(m) == 3.5
end
```

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/itensorcompat/surface.jl")'
```

**Commit:**

```bash
git add src/ITensorCompat.jl test/itensorcompat/surface.jl
git commit -m "feat: add itensorcompat mps operations"
```

## Task 11: Raw MPS and Narrow MPO Constructors

**Affected files:**
- Modify: `src/ITensorCompat.jl`
- Test: `test/itensorcompat/raw_blocks.jl`
- Modify: `test/runtests.jl`

**What to do:**

Add raw block constructors:

- `MPS(blocks::AbstractVector{<:Array{T,3}}, sites::Vector{Index})`
- `MPS(blocks::AbstractVector{<:Array{T,3}})`
- `MPO(blocks::AbstractVector{<:Array{T,4}}, input_sites, output_sites)`

Use the existing indexed/raw bridge from `src/TensorNetworks/bridge.jl` where
possible:

```julia
stt = Tensor4all.SimpleTT.TensorTrain{T,3}(collect(Array{T,3}.(blocks)))
return MPS(TensorNetworks.TensorTrain(stt, sites))
```

For generated MPS site indices, use stable tags:

```julia
sites = [Index(size(blocks[i], 2); tags=["Site", "n=$i"]) for i in eachindex(blocks)]
```

MPO orientation must be documented as `(left_link, input_site, output_site,
right_link)` because `TensorNetworks.TensorTrain(stt, input_inds, output_inds)`
uses that order. If BubbleTeaCI requires the opposite order, add an explicit
keyword later rather than guessing silently.

**Tests:**

Create `test/itensorcompat/raw_blocks.jl`:

```julia
using Test
using Tensor4all

const IC_RAW = Tensor4all.ITensorCompat

@testset "raw MPS block constructor" begin
    sites = [Index(2; tags=["s", "s=$n"]) for n in 1:2]
    blocks = [
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2),
        reshape([2.0, 0.0, 0.0, -1.0], 2, 2, 1),
    ]
    m = IC_RAW.MPS(blocks, sites)
    @test IC_RAW.siteinds(m) == sites
    dense = Array(IC_RAW.to_dense(m), sites...)
    @test dense == [2.0 0.0; 0.0 -1.0]
end
```

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/itensorcompat/raw_blocks.jl")'
```

**Commit:**

```bash
git add src/ITensorCompat.jl test/itensorcompat/raw_blocks.jl test/runtests.jl
git commit -m "feat: add itensorcompat raw block constructors"
```

## Task 12: BubbleTeaCI-Shaped Integration Test and Docs

**Affected files:**
- Create: `test/itensorcompat/bubbleteaci_workflow.jl`
- Modify: `test/runtests.jl`
- Modify: `docs/src/api.md`
- Modify: `docs/src/modules.md`
- Modify: `docs/src/index.md` if public workflow wording changes
- Review: `README.md`

**What to do:**

Add one compact integration test using only public APIs:

1. construct an `ITensorCompat.MPS` from raw TCI-like blocks;
2. replace site indices without touching `.data`;
3. insert a dummy identity site;
4. fix and sum selected indices;
5. add two MPS values;
6. truncate and orthogonalize with `cutoff`;
7. evaluate and materialize to dense;
8. verify scalar / zero-site behavior.

Add documentation notes:

- `ITensorCompat` is an opt-in migration facade.
- `TensorNetworks.TensorTrain` remains the primary indexed chain type.
- `ITensorCompat` is cutoff-only; native `threshold` / `svd_policy` remain in
  `TensorNetworks`.
- `fixinds`, `suminds`, `projectinds`, and structured identity constructors are
  generic Tensor4all APIs, not BubbleTeaCI-specific code.
- Raw MPO block orientation is documented.

Ensure `docs/src/api.md` includes:

````markdown
## ITensorCompat

```@autodocs
Modules = [Tensor4all.ITensorCompat]
Pages = ["ITensorCompat.jl"]
Private = false
Order = [:type, :function]
```
````

**Tests:**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/itensorcompat/bubbleteaci_workflow.jl")'
julia --startup-file=no --project=docs docs/make.jl
```

**Commit:**

```bash
git add test/itensorcompat/bubbleteaci_workflow.jl test/runtests.jl docs/src/api.md docs/src/modules.md docs/src/index.md README.md
git commit -m "docs: document itensorcompat follow-up workflow"
```

## Task Ordering

1. Task 1 must happen before structured constructors or storage metadata tests.
2. Task 2 must happen before preserving structured metadata through TensorTrain
   operations.
3. Task 3 is pure Core API and can happen before or after Task 1, but doing it
   after the pin keeps related Core tests together.
4. Task 4 depends on Tasks 1 and 2.
5. Task 5 is independent of structured storage and can be implemented while
   Rust pin work is being reviewed, but it should merge before ITensorCompat.
6. Task 6 depends on Tasks 2 and 5.
7. Task 7 depends on Task 3 and benefits from Task 5 canonical invalidation.
8. Task 8 depends on Tasks 4, 5, and 7.
9. Tasks 9-11 depend on the generic Core and TensorNetworks API being in place.
10. Task 12 is last.

## Test Strategy

Run focused tests after every task, then a final full verification:

```bash
T4A_SKIP_HDF5_TESTS=1 julia --startup-file=no --project=. test/runtests.jl
julia --startup-file=no --project=docs docs/make.jl
julia --startup-file=no scripts/check_autodocs_coverage.jl
```

If HDF5 is available and the host Julia does not hit the AMD EPYC issue, also
run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

For backend-dependent tasks, rebuild the Rust library after bumping the pin:

```bash
TENSOR4ALL_RS_PATH=/home/shinaoka/tensor4all/tensor4all-rs julia --startup-file=no --project=. deps/build.jl
```

## Acceptance Criteria Mapping

- TensorTrain mutation and canonical invalidation: Tasks 5 and 6.
- No downstream `.data` / `.llim` / `.rlim` mutation: Tasks 5, 8, 10, and 12.
- Metadata-only index replacement preservation: Tasks 2 and 6.
- `tagstring`: Task 3.
- Structured diagonal / identity constructors: Tasks 1, 2, and 4.
- `fixinds`, `suminds`, `projectinds`: Task 7.
- Dummy-site identity insertion: Task 8.
- `ITensorCompat.MPS` / `MPO`: Tasks 9-11.
- Cutoff-only compatibility APIs: Task 10.
- Scalar and empty-site MPS: Task 10.
- Raw constructors: Task 11.
- API docs and autodocs coverage: Tasks 4, 5, 7, 9, and 12.
- Full test and docs verification: Task 12 final checks.

## Open Questions Before Implementation

- Confirm the exact tensor4all-rs remote commit to put in
  `deps/TENSOR4ALL_RS_PIN`. The local checkout currently contains structured
  storage C API support, but the Tensor4all.jl PR must point at a merged remote
  commit.
- Confirm whether raw MPO blocks from BubbleTeaCI are ordered
  `(left, input, output, right)` or `(left, output, input, right)`. The plan
  uses the existing `TensorNetworks.TensorTrain(stt, input_inds, output_inds)`
  convention.
- Confirm whether `insert_identity!` should be exported from
  `TensorNetworks` immediately or kept documented but namespaced only as
  `TensorNetworks.insert_identity!`.
