# Tensor4all Phase 1 Implementation Handoff Plan

> Status: active as of 2026-04-15. This is the single authoritative implementation handoff document for the current repo-local phase.
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** backend-enable `Core.Index` and `Core.Tensor`, implement the remaining `TensorNetworks` helper surface with explicit physical-leg semantics, preserve the pure-Julia HDF5 round-trip, and leave `LinearOperator` materialization / `apply` deferred.

**Architecture:** keep the approved Julia module split already present in the repo:

- `Core` owns `Index`, `Tensor`, backend loading, low-level FFI helpers, and error handling
- `TensorNetworks` owns `TensorTrain = Vector{Tensor} + llim/rlim`, `LinearOperator`, and chain helper APIs
- `SimpleTT` keeps pure Julia raw-array numerics
- `TensorCI` keeps the `TensorCI2` return boundary
- `QuanticsGrids` / `QuanticsTCI` stay adopted re-export wrappers
- `QuanticsTransform` stays a quantics-specific operator-constructor layer
- the HDF5 boundary stays on `TensorNetworks.save_as_mps` / `load_tt`

**Tech Stack:** Julia 1.9+, `tensor4all-capi` from `../tensor4all-rs`, `LinearAlgebra`, `HDF5`, `Test`, `Documenter.jl`.

---

## Pre-Execution Requirements

- Execute this plan in a dedicated git worktree. Do not execute it on `main`.
- Run from the package root: `/Users/nepomuk/Documents/GitHub/Tensor4all.jl`.
- Treat `/Users/nepomuk/Documents/GitHub/tensor4all-rs` as the backend source of truth for this phase.
- Do not modify `tensor4all-rs` from this plan.
- Do not reintroduce `TreeTensorNetwork`, `MPS`, or `MPO` as public top-level `Tensor4all` types.
- Treat `docs/plans/2026-04-15-issue-35-skeleton-api-alignment.md` and `docs/plans/2026-04-15-remaining-skeleton-surface.md` as historical status notes, not active execution plans.

## Current Verified Baseline

These facts were re-checked against the current source tree before consolidating this handoff document.

- The two 2026-04-15 skeleton plans are already reflected in the source tree.
- `Tensor4all` no longer exports `TreeTensorNetwork`, `MPS`, or `MPO`.
- `TensorNetworks` already owns `TensorTrain`, `LinearOperator`, and `apply`.
- `QuanticsTransform` already returns `TensorNetworks.LinearOperator` placeholders.
- `TensorCI.crossinterpolate2` already returns `TensorCI2`.
- `SimpleTT` compression and MPO contraction are implemented in pure Julia.
- The HDF5 extension already attaches real methods to `TensorNetworks.save_as_mps` and `TensorNetworks.load_tt`.
- `Index` and `Tensor` are still metadata-owning Julia structs, not backend-handle wrappers.
- `test/core/bootstrap.jl`, `test/core/index.jl`, and `test/core/tensor.jl` exist, but `test/runtests.jl` does not include them yet.
- `Manifest.toml` is stale: `julia --startup-file=no --project=. test/runtests.jl` currently fails before running tests because `QuanticsGrids` is not installed in the active manifest.
- `Manifest.toml` still carries stale `Tensor4allITensorsExt` metadata even though the source tree has no ITensors extension file and `Project.toml` declares only `Tensor4allHDF5Ext`.
- `README.md` still contains pre-alignment claims such as `TensorCI` returning `SimpleTT` and `TreeTensorNetwork` still being part of the repository story.
- `ext/Tensor4allHDF5Ext.jl` currently depends on direct `Tensor` field access (`t.data`).
- A sibling backend checkout exists at `../tensor4all-rs`; `deps/build.jl` should continue to treat that checkout as the source of truth for this phase.

## Locked Decisions

These decisions should not be reopened during implementation unless the user explicitly asks for it.

1. **No Rust-side changes in this phase**
   - Do not modify `../tensor4all-rs` from this repository.
   - Do not add new C-API entrypoints in this phase.
   - If a Julia feature requires a missing backend symbol, leave that feature deferred and record the exact missing symbol.

2. **Keep the restored chain-oriented public split**
   - Do not reintroduce `TreeTensorNetwork`, `MPS`, or `MPO` as public types.
   - Keep `TensorNetworks.TensorTrain` as `Vector{Tensor} + llim/rlim`.
   - Keep `SimpleTT`, `TensorCI`, `QuanticsGrids`, `QuanticsTCI`, and `QuanticsTransform` as separate public modules.

3. **Keep `SimpleTT` and `TensorCI` behavior intact**
   - Do not move raw-array numerics out of `SimpleTT`.
   - Do not change `TensorCI.crossinterpolate2` back to returning a local tensor-train type.

4. **Keep HDF5 supported in pure Julia**
   - `TensorNetworks.save_as_mps` and `TensorNetworks.load_tt` must keep working.
   - The HDF5 extension may adapt to new `Index` / `Tensor` internals, but it must not be dropped or turned back into a stub.

5. **Keep quantics materialization deferred**
   - `QuanticsTransform` constructors stay metadata-only in this phase.
   - `TensorNetworks.apply` stays deferred unless the already-existing local C API clearly exposes the exact chain-kernel path needed with no new ABI work.
   - QTCI execution remains deferred.

6. **No ITensors scope in this phase**
   - Do not add or restore an ITensors extension in this pass.
   - The stale manifest metadata should disappear when the manifest is refreshed.

7. **Physical-leg convention is fixed**
   - Use ITensorMPS-style site semantics for chain helpers.
   - A tensor may carry zero, one, or many physical legs.
   - Physical legs belong to exactly one tensor; shared indices are link legs.
   - Site-helper APIs operate on the physical-leg subset, while the search helpers may still inspect all indices attached to a tensor.

## File Structure To Lock In Before Coding

- `src/Core/CAPI.jl`
  - Internal-only `ccall` wrappers, status-code constants, query-buffer helpers, pointer finalization helpers, and storage-kind helpers.
  - No public exports.
- `src/Core/Backend.jl`
  - Keeps lazy `dlopen` behavior and remains the only backend-loader entry point.
- `src/Core/Index.jl`
  - Public `Index` wrapper around a backend-owned handle.
  - Julia-side validation, tag normalization, equality, hashing, and display.
- `src/Core/Tensor.jl`
  - Public `Tensor` wrapper around a backend-owned handle.
  - Dense constructors, dense extraction, axis permutation, and tensor contraction.
- `src/TensorNetworks.jl`
  - Public chain container plus all chain-helper algorithms.
  - Physical-leg discovery, tag helpers, regrouping, diagonalization, and site matching.
- `ext/Tensor4allHDF5Ext.jl`
  - Pure-Julia schema writer/reader that uses public accessors and internal dense extraction helpers instead of struct fields.
- `test/core/index.jl`
  - Exact `Index` public contract tests.
- `test/core/tensor.jl`
  - Exact `Tensor` public contract tests, including contraction and numeric axis permutation.
- `test/tensornetworks/skeleton_surface.jl`
  - Real behavior tests for `TensorNetworks` helpers.
- `test/extensions/hdf5_roundtrip.jl`
  - HDF5 regression tests against backend-backed `Tensor` objects.
- `README.md`, `docs/src/api.md`, `docs/src/modules.md`, `docs/src/index.md`, `docs/src/deferred_rework_plan.md`, `docs/design/julia_ffi_core.md`, `docs/design/julia_ffi_tensornetworks.md`
  - Must match the implemented behavior exactly.

## Ground Truth That This Plan Assumes

### Actual `tensor4all-capi` Surface At The Local Backend Checkout

The backend symbols below were verified from `/Users/nepomuk/Documents/GitHub/tensor4all-rs/crates/tensor4all-capi` and should be treated as authoritative for this phase.

**Status codes**

- `T4A_SUCCESS = 0`
- `T4A_NULL_POINTER = -1`
- `T4A_INVALID_ARGUMENT = -2`
- `T4A_TAG_OVERFLOW = -3`
- `T4A_TAG_TOO_LONG = -4`
- `T4A_BUFFER_TOO_SMALL = -5`
- `T4A_INTERNAL_ERROR = -6`
- `T4A_NOT_IMPLEMENTED = -7`

**Error-message query**

- `t4a_last_error_message(buf, buf_len, out_len)` supports query-then-fill semantics.

**Index lifecycle and metadata**

- `t4a_index_new`
- `t4a_index_new_with_tags`
- `t4a_index_new_with_id`
- `t4a_index_release`
- `t4a_index_clone`
- `t4a_index_is_assigned`
- `t4a_index_dim`
- `t4a_index_id`
- `t4a_index_get_plev`
- `t4a_index_get_tags`
- `t4a_index_set_tags_csv`
- `t4a_index_set_plev`
- `t4a_index_prime`
- `t4a_index_has_tag`

**Tensor lifecycle and metadata**

- `t4a_tensor_new_dense_f64`
- `t4a_tensor_new_dense_c64`
- `t4a_tensor_release`
- `t4a_tensor_clone`
- `t4a_tensor_is_assigned`
- `t4a_tensor_get_rank`
- `t4a_tensor_get_dims`
- `t4a_tensor_get_indices`
- `t4a_tensor_get_storage_kind`
- `t4a_tensor_get_data_f64`
- `t4a_tensor_get_data_c64`
- `t4a_tensor_contract`

**Storage kinds**

- `DenseF64 = 0`
- `DenseC64 = 1`
- `DiagF64 = 2`
- `DiagC64 = 3`

### Data Layout Rule

- The backend code returns tensor data in **column-major** order.
- The `tensor4all-capi` README still says row-major; that README is stale.
- Julia wrappers must trust the Rust source code and tests, not the stale README.
- Therefore `_dense_array(t)` should use `reshape` directly on the returned flat data, with no row-major transposition.

### Physical-Leg Semantics To Implement

These rules are fixed for this execution plan.

1. A **physical leg** is an index that appears on exactly one tensor in `tt.data`, comparing indices by full Julia equality.
2. A **link leg** is an index that appears on exactly two tensors and those tensors must be adjacent in chain order.
3. An index appearing on more than two tensors is invalid chain structure and helper algorithms must throw `ArgumentError`.
4. An index shared by non-adjacent tensors is invalid chain structure and helper algorithms must throw `ArgumentError`.
5. A tensor may carry zero, one, or many physical legs.
6. Search helpers `findsite` and `findsites` inspect **all** indices attached to each tensor.
7. Tag-based and site-layout helpers operate on the **physical-leg subset** only.
8. Tag-based enumeration follows the `Quantics.jl` rule: for `tag="x"`, search for unique unprimed physical indices tagged `x=1`, `x=2`, `x=3`, ... and stop at the first missing number.
9. `findallsiteinds_by_tag` returns those matching unprimed physical indices in increasing tag-number order.
10. `findallsites_by_tag` returns the corresponding 1-based tensor positions in that same order.

### Canonical Index Order For Helper-Generated Site Tensors

When any helper in `TensorNetworks.jl` creates a new site tensor, the index order must be:

- left link, if present
- all physical legs for that site, in the exact order requested by the public API
- right link, if present

This convention applies to tensors created by `rearrange_siteinds`, `makesitediagonal`, `extractdiagonal`, and `matchsiteinds`.

### `TensorTrain` Bounds Rule

- For helpers that preserve the number of tensors, keep `llim` and `rlim` unchanged.
- For helpers that change the number of tensors (`rearrange_siteinds` and `matchsiteinds`), return `TensorTrain(new_data, 0, length(new_data) + 1)`.

## Task 0: Repair Project State And Lock The Baseline

**Files:**

- Modify: `Manifest.toml` (via `Pkg.resolve()` / `Pkg.instantiate()`; do not edit manually)
- Modify: `test/runtests.jl`

- [ ] **Step 1: Refresh the manifest from the current project**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

Expected:

- `QuanticsGrids`, `QuanticsTCI`, and `TensorCrossInterpolation` are installed.
- The stale `Tensor4allITensorsExt` manifest entry disappears.

- [ ] **Step 2: Build the local backend library**

Run:

```bash
julia --startup-file=no --project=. deps/build.jl
```

Expected:

- `deps/libtensor4all_capi.dylib` exists on macOS.
- The build uses the sibling backend checkout instead of cloning a fallback copy.

- [ ] **Step 3: Add the core tests to the default test entry point**

Replace the top of `test/runtests.jl` with:

```julia
using Test
using Tensor4all

include("core/bootstrap.jl")
include("core/index.jl")
include("core/tensor.jl")
include("api/skeleton_alignment.jl")
include("tensornetworks/tensortrain.jl")
include("tensornetworks/skeleton_surface.jl")
include("simplett/surface.jl")
include("simplett/compress.jl")
include("simplett/contraction.jl")
include("tensorci/surface.jl")
include("tensorci/crossinterpolate2.jl")
include("quanticsgrids/surface.jl")
include("quanticstci/surface.jl")
if get(ENV, "T4A_SKIP_HDF5_TESTS", "0") != "1" && Base.find_package("HDF5") !== nothing
   include("extensions/hdf5_roundtrip.jl")
end
include("quanticstransform/surface.jl")
```

- [ ] **Step 4: Run the current baseline tests before changing behavior**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected:

- PASS with the current skeleton implementation.
- This gives a clean pre-change baseline before the backend wrapper work starts.

- [ ] **Step 5: Commit the baseline repair**

Run:

```bash
git add Manifest.toml test/runtests.jl
git commit -m "chore: repair baseline manifest and core test wiring"
```

## Task 1: Create The CAPI Layer And Replace `Index`

**Files:**

- Create: `src/Core/CAPI.jl`
- Modify: `src/Tensor4all.jl`
- Modify: `src/Core/Index.jl`
- Modify: `test/core/index.jl`
- Modify: `docs/src/api.md`
- Modify: `docs/design/julia_ffi_core.md`

- [ ] **Step 1: Replace the `Index` tests with the real backend-wrapper contract**

Update `test/core/index.jl` to:

```julia
using Test
using Tensor4all

@testset "Index backend wrapper" begin
   i = Tensor4all.Index(4; tags=["x", "site", "x"], plev=1, id=42)
   j = Tensor4all.sim(i)

   @test Tensor4all.dim(i) == 4
   @test Tensor4all.tags(i) == ["site", "x"]
   @test Tensor4all.plev(i) == 1
   @test Tensor4all.id(i) == 42
   @test Tensor4all.hastag(i, "x")
   @test sprint(show, i) == "Index(4|site,x; plev=1)"

   same = Tensor4all.Index(4; tags=["site", "x"], plev=1, id=42)
   @test same == i
   @test hash(same) == hash(i)

   @test Tensor4all.id(j) != Tensor4all.id(i)
   @test Tensor4all.dim(j) == Tensor4all.dim(i)
   @test Tensor4all.tags(j) == Tensor4all.tags(i)
   @test Tensor4all.plev(j) == Tensor4all.plev(i)

   ip = Tensor4all.prime(i, 2)
   @test Tensor4all.plev(ip) == 3
   @test Tensor4all.id(ip) == Tensor4all.id(i)
   @test Tensor4all.plev(Tensor4all.noprime(ip)) == 0
   @test Tensor4all.plev(Tensor4all.setprime(i, 7)) == 7

   xs = [i, j, ip]
   ys = [j, ip]
   @test Tensor4all.commoninds(xs, ys) == [j, ip]
   @test Tensor4all.uniqueinds(xs, ys) == [i]

   @test_throws ArgumentError Tensor4all.Index(0)
   @test_throws ArgumentError Tensor4all.Index(2; plev=-1)
end
```

- [ ] **Step 2: Run the new `Index` tests to confirm they fail on the current skeleton**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/core/index.jl")'
```

Expected:

- FAIL because tags are not normalized and the current struct is not a backend-handle wrapper.

- [ ] **Step 3: Create `src/Core/CAPI.jl` with the internal FFI helpers**

Create `src/Core/CAPI.jl` with this content:

```julia
const _StatusCode = Cint
const _T4A_SUCCESS = _StatusCode(0)
const _T4A_BUFFER_TOO_SMALL = _StatusCode(-5)

const _T4A_STORAGE_DENSE_F64 = Cint(0)
const _T4A_STORAGE_DENSE_C64 = Cint(1)
const _T4A_STORAGE_DIAG_F64 = Cint(2)
const _T4A_STORAGE_DIAG_C64 = Cint(3)

function _last_error_message()
   lib = require_backend()
   out_len = Ref{Csize_t}(0)
   status = ccall((:t4a_last_error_message, lib), _StatusCode,
      (Ptr{UInt8}, Csize_t, Ref{Csize_t}), C_NULL, 0, out_len)
   status == _T4A_SUCCESS || return "no backend error available"
   buf = Vector{UInt8}(undef, out_len[])
   status = ccall((:t4a_last_error_message, lib), _StatusCode,
      (Ptr{UInt8}, Csize_t, Ref{Csize_t}), buf, length(buf), out_len)
   status == _T4A_SUCCESS || return "no backend error available"
   nul = findfirst(==(0x00), buf)
   return String(isnothing(nul) ? buf : buf[1:(nul - 1)])
end

function _throw_last_error(context::AbstractString)
   msg = _last_error_message()
   isempty(msg) && (msg = "no backend error available")
   throw(ErrorException("tensor4all-capi failure in $(context): $(msg)"))
end

function _check_status(status::_StatusCode, context::AbstractString)
   status == _T4A_SUCCESS && return
   _throw_last_error(context)
end

function _check_ptr(ptr::Ptr{Cvoid}, context::AbstractString)
   ptr != C_NULL && return ptr
   _throw_last_error(context)
end

function _query_cstring(symbol::Symbol, ptr::Ptr{Cvoid}, context::AbstractString)
   lib = require_backend()
   out_len = Ref{Csize_t}(0)
   status = ccall((symbol, lib), _StatusCode,
      (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ref{Csize_t}), ptr, C_NULL, 0, out_len)
   _check_status(status, context)
   buf = Vector{UInt8}(undef, out_len[])
   status = ccall((symbol, lib), _StatusCode,
      (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ref{Csize_t}), ptr, buf, length(buf), out_len)
   _check_status(status, context)
   nul = findfirst(==(0x00), buf)
   raw = isnothing(nul) ? buf : buf[1:(nul - 1)]
   return String(raw)
end

function _release_handle!(release_symbol::Symbol, obj, field::Symbol)
   ptr = getfield(obj, field)
   ptr == C_NULL && return
   lib = require_backend()
   ccall((release_symbol, lib), Cvoid, (Ptr{Cvoid},), ptr)
   setfield!(obj, field, C_NULL)
   return
end
```

- [ ] **Step 4: Include `CAPI.jl` before `Index.jl` and `Tensor.jl`**

Update the core include block in `src/Tensor4all.jl` to:

```julia
include("Core/Errors.jl")
include("Core/Backend.jl")
include("Core/CAPI.jl")
include("Core/Index.jl")
include("Core/Tensor.jl")
```

- [ ] **Step 5: Replace `src/Core/Index.jl` with a backend-handle wrapper**

Use this implementation shape:

```julia
mutable struct Index
   ptr::Ptr{Cvoid}
end

function _normalized_tags(tags::AbstractVector{<:AbstractString})
   normalized = String.(tags)
   sort!(unique!(normalized))
   return normalized
end

_tags_csv(tags::AbstractVector{String}) = join(tags, ",")

function _adopt_index(ptr::Ptr{Cvoid}, context::AbstractString)
   ptr = _check_ptr(ptr, context)
   index = Index(ptr)
   finalizer(i -> _release_handle!(:t4a_index_release, i, :ptr), index)
   return index
end

function _clone_index(i::Index)
   lib = require_backend()
   ptr = ccall((:t4a_index_clone, lib), Ptr{Cvoid}, (Ptr{Cvoid},), i.ptr)
   return _adopt_index(ptr, "t4a_index_clone")
end

function Index(dim::Integer; tags::AbstractVector{<:AbstractString}=String[], plev::Integer=0, id=nothing)
   dim > 0 || throw(ArgumentError("Index dimension must be positive, got $(dim)"))
   plev >= 0 || throw(ArgumentError("Index prime level must be nonnegative, got $(plev)"))
   normalized_tags = _normalized_tags(tags)
   tags_csv = _tags_csv(normalized_tags)
   lib = require_backend()
   ptr = if id === nothing
      isempty(normalized_tags) ?
         ccall((:t4a_index_new, lib), Ptr{Cvoid}, (Csize_t,), dim) :
         ccall((:t4a_index_new_with_tags, lib), Ptr{Cvoid}, (Csize_t, Cstring), dim, tags_csv)
   else
      isempty(normalized_tags) ?
         ccall((:t4a_index_new_with_id, lib), Ptr{Cvoid}, (Csize_t, UInt64, Ptr{Cchar}), dim, UInt64(id), C_NULL) :
         ccall((:t4a_index_new_with_id, lib), Ptr{Cvoid}, (Csize_t, UInt64, Cstring), dim, UInt64(id), tags_csv)
   end
   index = _adopt_index(ptr, "t4a_index_new")
   plev == 0 || _check_status(ccall((:t4a_index_set_plev, lib), _StatusCode, (Ptr{Cvoid}, Int64), index.ptr, plev), "t4a_index_set_plev")
   return index
end

function dim(i::Index)
   lib = require_backend()
   out = Ref{Csize_t}(0)
   _check_status(ccall((:t4a_index_dim, lib), _StatusCode, (Ptr{Cvoid}, Ref{Csize_t}), i.ptr, out), "t4a_index_dim")
   return Int(out[])
end

function id(i::Index)
   lib = require_backend()
   out = Ref{UInt64}(0)
   _check_status(ccall((:t4a_index_id, lib), _StatusCode, (Ptr{Cvoid}, Ref{UInt64}), i.ptr, out), "t4a_index_id")
   return out[]
end

function tags(i::Index)
   raw = _query_cstring(:t4a_index_get_tags, i.ptr, "t4a_index_get_tags")
   return isempty(raw) ? String[] : split(raw, ",")
end

function plev(i::Index)
   lib = require_backend()
   out = Ref{Int64}(0)
   _check_status(ccall((:t4a_index_get_plev, lib), _StatusCode, (Ptr{Cvoid}, Ref{Int64}), i.ptr, out), "t4a_index_get_plev")
   return Int(out[])
end

function hastag(i::Index, tag::AbstractString)
   lib = require_backend()
   rc = ccall((:t4a_index_has_tag, lib), Cint, (Ptr{Cvoid}, Cstring), i.ptr, String(tag))
   rc >= 0 || _throw_last_error("t4a_index_has_tag")
   return rc == 1
end

sim(i::Index) = Index(dim(i); tags=tags(i), plev=plev(i))

function prime(i::Index, n::Integer=1)
   new_plev = plev(i) + Int(n)
   new_plev >= 0 || throw(ArgumentError("Index prime level must stay nonnegative, got $(new_plev)"))
   j = _clone_index(i)
   lib = require_backend()
   _check_status(ccall((:t4a_index_set_plev, lib), _StatusCode, (Ptr{Cvoid}, Int64), j.ptr, new_plev), "t4a_index_set_plev")
   return j
end

noprime(i::Index) = setprime(i, 0)

function setprime(i::Index, n::Integer)
   n >= 0 || throw(ArgumentError("Index prime level must be nonnegative, got $(n)"))
   j = _clone_index(i)
   lib = require_backend()
   _check_status(ccall((:t4a_index_set_plev, lib), _StatusCode, (Ptr{Cvoid}, Int64), j.ptr, n), "t4a_index_set_plev")
   return j
end
```

Keep the existing `replaceind`, `replaceinds`, `commoninds`, `uniqueinds`, `==`, `hash`, and `show` public behavior, but rewrite them to call the new getter functions rather than reading struct fields.

- [ ] **Step 6: Run the `Index` tests and the bootstrap smoke test**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/core/bootstrap.jl"); include("test/core/index.jl")'
```

Expected:

- PASS.

- [ ] **Step 7: Commit the `Index` backend wrapper**

Run:

```bash
git add src/Core/CAPI.jl src/Tensor4all.jl src/Core/Index.jl test/core/index.jl docs/src/api.md docs/design/julia_ffi_core.md
git commit -m "feat: backend-enable core index wrapper"
```

## Task 2: Replace `Tensor` And Remove HDF5 Field Coupling

**Files:**

- Modify: `src/Core/Tensor.jl`
- Modify: `ext/Tensor4allHDF5Ext.jl`
- Modify: `test/core/tensor.jl`
- Modify: `test/extensions/hdf5_roundtrip.jl`
- Modify: `docs/src/api.md`
- Modify: `docs/design/julia_ffi_core.md`

- [ ] **Step 1: Replace the tensor tests with real backend and numeric-behavior assertions**

Update `test/core/tensor.jl` to:

```julia
using Test
using Tensor4all

@testset "Tensor backend wrapper" begin
   i = Tensor4all.Index(2; tags=["i"])
   j = Tensor4all.Index(3; tags=["j"])
   k = Tensor4all.Index(4; tags=["k"])

   data = reshape(collect(1.0:6.0), 2, 3)
   tensor = Tensor4all.Tensor(data, [i, j])

   @test Tensor4all.rank(tensor) == 2
   @test Tensor4all.dims(tensor) == (2, 3)
   @test Tensor4all.inds(tensor) == [i, j]

   primed = Tensor4all.prime(tensor)
   @test Tensor4all.inds(primed) == [Tensor4all.prime(i), Tensor4all.prime(j)]

   swapped = Tensor4all.swapinds(tensor, i, j)
   swapped_data, swapped_inds = Tensor4all._dense_array(swapped)
   @test swapped_inds == [j, i]
   @test swapped_data == permutedims(data, (2, 1))

   a = Tensor4all.Tensor(reshape(collect(1.0:6.0), 2, 3), [i, j])
   b = Tensor4all.Tensor(reshape(collect(1.0:12.0), 3, 4), [j, k])
   c = Tensor4all.contract(a, b)
   c_data, c_inds = Tensor4all._dense_array(c)
   @test c_inds == [i, k]
   @test c_data == reshape(collect(1.0:6.0), 2, 3) * reshape(collect(1.0:12.0), 3, 4)

   zdata = ComplexF64[1 + 2im 3 + 4im 5 + 6im; 7 + 8im 9 + 10im 11 + 12im]
   ztensor = Tensor4all.Tensor(zdata, [i, j])
   zback, zinds = Tensor4all._dense_array(ztensor)
   @test zinds == [i, j]
   @test zback == zdata

   bad = PermutedDimsArray(reshape(collect(1.0:8.0), 2, 2, 2), (2, 1, 3))
   @test_throws ArgumentError Tensor4all.Tensor(bad, [i, j, Tensor4all.Index(2; tags=["k"])])
   @test_throws DimensionMismatch Tensor4all.Tensor(data, [i])
end
```

- [ ] **Step 2: Run the new tensor tests to confirm they fail on the current skeleton**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/core/tensor.jl")'
```

Expected:

- FAIL because `contract` still throws and `swapinds` does not permute numeric axes.

- [ ] **Step 3: Replace `src/Core/Tensor.jl` with a backend-handle wrapper**

Use this implementation shape:

```julia
mutable struct Tensor
   ptr::Ptr{Cvoid}
end

function _adopt_tensor(ptr::Ptr{Cvoid}, context::AbstractString)
   ptr = _check_ptr(ptr, context)
   tensor = Tensor(ptr)
   finalizer(t -> _release_handle!(:t4a_tensor_release, t, :ptr), tensor)
   return tensor
end

function _clone_tensor(t::Tensor)
   lib = require_backend()
   ptr = ccall((:t4a_tensor_clone, lib), Ptr{Cvoid}, (Ptr{Cvoid},), t.ptr)
   return _adopt_tensor(ptr, "t4a_tensor_clone")
end

function _storage_kind(t::Tensor)
   lib = require_backend()
   out = Ref{Cint}(0)
   _check_status(ccall((:t4a_tensor_get_storage_kind, lib), _StatusCode, (Ptr{Cvoid}, Ref{Cint}), t.ptr, out), "t4a_tensor_get_storage_kind")
   return out[]
end

function Tensor(data::Array{Float64, N}, inds::AbstractVector{Index}) where {N}
   length(inds) == N || throw(DimensionMismatch("Tensor rank $(N) requires $(N) indices, got $(length(inds))"))
   Tuple(dim.(inds)) == size(data) || throw(DimensionMismatch("Tensor dimensions $(Tuple(dim.(inds))) do not match data size $(size(data))"))
   lib = require_backend()
   dims = Csize_t.(size(data))
   index_ptrs = Ptr{Cvoid}[i.ptr for i in inds]
   ptr = ccall((:t4a_tensor_new_dense_f64, lib), Ptr{Cvoid},
      (Csize_t, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Cdouble}, Csize_t),
      N, index_ptrs, dims, data, length(data))
   return _adopt_tensor(ptr, "t4a_tensor_new_dense_f64")
end

function Tensor(data::Array{ComplexF64, N}, inds::AbstractVector{Index}) where {N}
   length(inds) == N || throw(DimensionMismatch("Tensor rank $(N) requires $(N) indices, got $(length(inds))"))
   Tuple(dim.(inds)) == size(data) || throw(DimensionMismatch("Tensor dimensions $(Tuple(dim.(inds))) do not match data size $(size(data))"))
   lib = require_backend()
   dims = Csize_t.(size(data))
   index_ptrs = Ptr{Cvoid}[i.ptr for i in inds]
   raw = Vector{Float64}(undef, 2 * length(data))
   for n in eachindex(data)
      raw[2n - 1] = real(data[n])
      raw[2n] = imag(data[n])
   end
   ptr = ccall((:t4a_tensor_new_dense_c64, lib), Ptr{Cvoid},
      (Csize_t, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Cdouble}, Csize_t),
      N, index_ptrs, dims, raw, length(data))
   return _adopt_tensor(ptr, "t4a_tensor_new_dense_c64")
end

function Tensor(data::AbstractArray, inds::AbstractVector{Index}; backend_handle=nothing)
   throw(ArgumentError("Array must be contiguous in memory for C API. Got $(typeof(data)). Use collect(data) to make a contiguous copy."))
end

function inds(t::Tensor)
   lib = require_backend()
   out_rank = Ref{Csize_t}(0)
   _check_status(ccall((:t4a_tensor_get_rank, lib), _StatusCode, (Ptr{Cvoid}, Ref{Csize_t}), t.ptr, out_rank), "t4a_tensor_get_rank")
   handles = Vector{Ptr{Cvoid}}(undef, out_rank[])
   _check_status(ccall((:t4a_tensor_get_indices, lib), _StatusCode, (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Csize_t), t.ptr, handles, length(handles)), "t4a_tensor_get_indices")
   return [_adopt_index(handle, "t4a_tensor_get_indices") for handle in handles]
end

function rank(t::Tensor)
   lib = require_backend()
   out_rank = Ref{Csize_t}(0)
   _check_status(ccall((:t4a_tensor_get_rank, lib), _StatusCode, (Ptr{Cvoid}, Ref{Csize_t}), t.ptr, out_rank), "t4a_tensor_get_rank")
   return Int(out_rank[])
end

function dims(t::Tensor)
   lib = require_backend()
   n = rank(t)
   out_dims = Vector{Csize_t}(undef, n)
   _check_status(ccall((:t4a_tensor_get_dims, lib), _StatusCode, (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t), t.ptr, out_dims, length(out_dims)), "t4a_tensor_get_dims")
   return Tuple(Int.(out_dims))
end

function _dense_array(t::Tensor)
   shape = dims(t)
   tensor_inds = inds(t)
   len = prod(shape)
   kind = _storage_kind(t)
   lib = require_backend()
   if kind == _T4A_STORAGE_DENSE_F64 || kind == _T4A_STORAGE_DIAG_F64
      out_len = Ref{Csize_t}(0)
      buf = Vector{Float64}(undef, len)
      _check_status(ccall((:t4a_tensor_get_data_f64, lib), _StatusCode,
         (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ref{Csize_t}), t.ptr, buf, len, out_len), "t4a_tensor_get_data_f64")
      return reshape(buf, shape...), tensor_inds
   else
      out_len = Ref{Csize_t}(0)
      raw = Vector{Float64}(undef, 2 * len)
      _check_status(ccall((:t4a_tensor_get_data_c64, lib), _StatusCode,
         (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ref{Csize_t}), t.ptr, raw, len, out_len), "t4a_tensor_get_data_c64")
      data = ComplexF64[ComplexF64(raw[2n - 1], raw[2n]) for n in 1:len]
      return reshape(data, shape...), tensor_inds
   end
end

function prime(t::Tensor, n::Integer=1)
   data, tensor_inds = _dense_array(t)
   return Tensor(copy(data), prime.(tensor_inds, Ref(n)))
end

function swapinds(t::Tensor, a::Index, b::Index)
   data, tensor_inds = _dense_array(t)
   pa = findall(==(a), tensor_inds)
   pb = findall(==(b), tensor_inds)
   length(pa) == 1 || throw(ArgumentError("Index $(a) must appear exactly once"))
   length(pb) == 1 || throw(ArgumentError("Index $(b) must appear exactly once"))
   perm = collect(1:length(tensor_inds))
   perm[pa[1]], perm[pb[1]] = perm[pb[1]], perm[pa[1]]
   return Tensor(permutedims(data, Tuple(perm)), tensor_inds[perm])
end

function contract(a::Tensor, b::Tensor)
   lib = require_backend()
   out = Ref{Ptr{Cvoid}}(C_NULL)
   _check_status(ccall((:t4a_tensor_contract, lib), _StatusCode, (Ptr{Cvoid}, Ptr{Cvoid}, Ref{Ptr{Cvoid}}), a.ptr, b.ptr, out), "t4a_tensor_contract")
   return _adopt_tensor(out[], "t4a_tensor_contract")
end
```

Do **not** add diagonal constructors on the Julia side in this phase. The current backend does not export `t4a_tensor_new_diag_*` symbols.

- [ ] **Step 4: Remove direct `Tensor` field access from the HDF5 extension and tests**

Make these exact substitutions.

In `ext/Tensor4allHDF5Ext.jl`, replace `_write_itensor` with:

```julia
function _write_itensor(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, t::Tensor4all.Tensor)
   g = create_group(parent, name)
   attributes(g)["type"] = _ITENSOR_TYPE
   attributes(g)["version"] = _VERSION
   data, inds = Tensor4all._dense_array(t)
   _write_indexset(g, "inds", inds)
   storage = create_group(g, "storage")
   attributes(storage)["type"] = _dense_typestr(eltype(data))
   attributes(storage)["version"] = _VERSION
   write(storage, "data", vec(data))
   return g
end
```

In `test/extensions/hdf5_roundtrip.jl`, replace direct `.data` assertions with:

```julia
tt2_data1, _ = Tensor4all._dense_array(tt2.data[1])
orig_data1, _ = Tensor4all._dense_array(t)
@test tt2_data1 == orig_data1
```

and similarly for the 2-site case.

- [ ] **Step 5: Run the tensor and HDF5 tests**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/core/tensor.jl")'
julia --startup-file=no --project=. -e 'using HDF5; include("test/extensions/hdf5_roundtrip.jl")'
```

Expected:

- PASS.

- [ ] **Step 6: Commit the tensor wrapper and HDF5 migration**

Run:

```bash
git add src/Core/Tensor.jl ext/Tensor4allHDF5Ext.jl test/core/tensor.jl test/extensions/hdf5_roundtrip.jl docs/src/api.md docs/design/julia_ffi_core.md
git commit -m "feat: backend-enable tensor wrapper and hdf5 accessors"
```

## Task 3: Implement Search And Tag Helpers In `TensorNetworks`

**Files:**

- Modify: `src/TensorNetworks.jl`
- Modify: `test/tensornetworks/skeleton_surface.jl`
- Modify: `docs/src/api.md`
- Modify: `docs/design/julia_ffi_tensornetworks.md`

- [ ] **Step 1: Replace the skeleton-surface tests for the search/tag helpers**

At the top of `test/tensornetworks/skeleton_surface.jl`, replace the current stub assertions with:

```julia
using Test
using Tensor4all

function _fixture_tt()
   TN = Tensor4all.TensorNetworks
   x1 = Index(2; tags=["x=1"])
   y1 = Index(2; tags=["y=1"])
   x2 = Index(2; tags=["x=2"])
   y2 = Index(2; tags=["y=2"])
   l12 = Index(3; tags=["Link", "l=1"])

   t1 = Tensor(reshape(collect(1.0:12.0), 2, 2, 3), [x1, y1, l12])
   t2 = Tensor(reshape(collect(1.0:12.0), 3, 2, 2), [l12, x2, y2])
   return TN.TensorTrain([t1, t2], 0, 3), (; x1, y1, x2, y2, l12)
end

@testset "TensorNetworks search and tag helpers" begin
   TN = Tensor4all.TensorNetworks
   tt, inds = _fixture_tt()

   @test TN.findsite(tt, inds.x1) == 1
   @test TN.findsite(tt, inds.l12) == 1
   @test TN.findsites(tt, inds.l12) == [1, 2]
   @test TN.findsites(tt, [inds.y1, inds.x2]) == [1, 2]
   @test TN.findsite(tt, Index(2; tags=["missing"])) === nothing

   @test TN.findallsiteinds_by_tag(tt; tag="x") == [inds.x1, inds.x2]
   @test TN.findallsiteinds_by_tag(tt; tag="y") == [inds.y1, inds.y2]
   @test TN.findallsites_by_tag(tt; tag="x") == [1, 2]
   @test TN.findallsites_by_tag(tt; tag="y") == [1, 2]
   @test isempty(TN.findallsiteinds_by_tag(tt; tag="z"))
   @test_throws ArgumentError TN.findallsiteinds_by_tag(tt; tag="x=1")
end
```

- [ ] **Step 2: Run the new helper tests to confirm they fail on the current stubs**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/tensornetworks/skeleton_surface.jl")'
```

Expected:

- FAIL because all helper methods still throw `SkeletonNotImplemented`.

- [ ] **Step 3: Add the internal physical-leg and tag helpers to `src/TensorNetworks.jl`**

Insert these helpers near the top of the module, before the public methods:

```julia
_valid_site_tag(tag::String) = !occursin("=", tag)

function _site_occurrences(tt::TensorTrain)
   occurrences = Dict{Index, Vector{Int}}()
   for (sitepos, tensor) in enumerate(tt.data)
      for ind in inds(tensor)
         push!(get!(occurrences, ind, Int[]), sitepos)
      end
   end
   return occurrences
end

function _physical_site_layout(tt::TensorTrain)
   occurrences = _site_occurrences(tt)
   layout = Vector{Vector{Index}}(undef, length(tt))
   for (sitepos, tensor) in enumerate(tt.data)
      physical = Index[]
      for ind in inds(tensor)
         sites = occurrences[ind]
         if length(sites) == 1
            push!(physical, ind)
         elseif length(sites) == 2
            abs(sites[1] - sites[2]) == 1 || throw(ArgumentError("Link index $(ind) connects non-adjacent tensors $(sites)"))
         else
            throw(ArgumentError("Index $(ind) appears on $(length(sites)) tensors; expected 1 or 2"))
         end
      end
      layout[sitepos] = physical
   end
   return layout
end

function _findallsites_by_tag_layout(layout::Vector{Vector{Index}}; tag::String, maxnsites::Int=1000)
   _valid_site_tag(tag) || throw(ArgumentError("Invalid tag: $(tag)"))
   positions = Int[]
   for n in 1:maxnsites
      target = "$(tag)=$(n)"
      matches = [(i, j) for i in eachindex(layout) for j in eachindex(layout[i]) if plev(layout[i][j]) == 0 && hastag(layout[i][j], target)]
      isempty(matches) && break
      length(matches) == 1 || throw(ArgumentError("Found more than one physical leg tagged $(target)"))
      push!(positions, matches[1][1])
   end
   return positions
end
```

- [ ] **Step 4: Implement the public search/tag helpers with Quantics-compatible semantics**

Replace the stub methods with:

```julia
function findsite(tt::TensorTrain, is)
   query = is isa Index ? [is] : collect(is)
   return findfirst(t -> !isempty(commoninds(inds(t), query)), tt.data)
end

findsite(tt::TensorTrain, i::Index) = findsite(tt, [i])

function findsites(tt::TensorTrain, is)
   query = is isa Index ? [is] : collect(is)
   return findall(t -> !isempty(commoninds(inds(t), query)), tt.data)
end

findsites(tt::TensorTrain, i::Index) = findsites(tt, [i])

function findallsiteinds_by_tag(tt::TensorTrain; tag::String="x", maxnsites::Int=1000)
   layout = _physical_site_layout(tt)
   positions = _findallsites_by_tag_layout(layout; tag=tag, maxnsites=maxnsites)
   return [only(filter(ind -> plev(ind) == 0 && hastag(ind, "$(tag)=$(n)"), layout[pos])) for (n, pos) in enumerate(positions)]
end

function findallsites_by_tag(tt::TensorTrain; tag::String="x", maxnsites::Int=1000)
   layout = _physical_site_layout(tt)
   return _findallsites_by_tag_layout(layout; tag=tag, maxnsites=maxnsites)
end
```

- [ ] **Step 5: Run the targeted helper tests**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/tensornetworks/skeleton_surface.jl")'
```

Expected:

- PASS for the search/tag helper section.

- [ ] **Step 6: Commit the search/tag helper implementation**

Run:

```bash
git add src/TensorNetworks.jl test/tensornetworks/skeleton_surface.jl docs/src/api.md docs/design/julia_ffi_tensornetworks.md
git commit -m "feat: implement tensornetworks search and tag helpers"
```

## Task 4: Implement Replacement And Rearrangement Helpers

**Files:**

- Modify: `src/TensorNetworks.jl`
- Modify: `test/tensornetworks/skeleton_surface.jl`
- Modify: `docs/src/api.md`
- Modify: `docs/design/julia_ffi_tensornetworks.md`

- [ ] **Step 1: Add the replacement and regrouping tests**

Append this testset to `test/tensornetworks/skeleton_surface.jl`:

```julia
function _dense_train(tt::Tensor4all.TensorNetworks.TensorTrain, physical_order)
   tensor = foldl(Tensor4all.contract, tt.data)
   data, inds = Tensor4all._dense_array(tensor)
   perm = [findfirst(==(ind), inds) for ind in physical_order]
   return permutedims(data, Tuple(perm))
end

@testset "TensorNetworks replacement and rearrangement helpers" begin
   TN = Tensor4all.TensorNetworks
   tt, inds = _fixture_tt()

   x1n = Index(2; tags=["xnew=1"])
   x2n = Index(2; tags=["xnew=2"])
   tt2 = TN.replace_siteinds(tt, [inds.x1, inds.x2], [x1n, x2n])
   @test TN.findallsiteinds_by_tag(tt2; tag="xnew") == [x1n, x2n]
   @test TN.findallsiteinds_by_tag(tt; tag="x") == [inds.x1, inds.x2]

   TN.replace_siteinds!(tt, [inds.y1], [Index(2; tags=["ynew=1"])])
   @test TN.findallsites_by_tag(tt; tag="ynew") == [1]

   tt3, inds3 = _fixture_tt()
   y1n = Index(2; tags=["ynew=1"])
   TN.replace_siteinds_part!(tt3, [inds3.y1], [y1n])
   @test TN.findallsiteinds_by_tag(tt3; tag="x") == [inds3.x1, inds3.x2]
   @test TN.findallsites_by_tag(tt3; tag="ynew") == [1]

   x1 = Index(2; tags=["x=1"])
   y1 = Index(2; tags=["y=1"])
   x2 = Index(2; tags=["x=2"])
   y2 = Index(2; tags=["y=2"])
   l1 = Index(2; tags=["Link", "l=1"])
   l2 = Index(2; tags=["Link", "l=2"])
   l3 = Index(2; tags=["Link", "l=3"])

   t1 = Tensor(reshape(collect(1.0:4.0), 2, 2), [x1, l1])
   t2 = Tensor(reshape(collect(1.0:8.0), 2, 2, 2), [l1, y1, l2])
   t3 = Tensor(reshape(collect(1.0:8.0), 2, 2, 2), [l2, x2, l3])
   t4 = Tensor(reshape(collect(1.0:4.0), 2, 2), [l3, y2])
   tt4 = TN.TensorTrain([t1, t2, t3, t4], 0, 5)

   fused = TN.rearrange_siteinds(tt4, [[x1, y1], [x2, y2]])
   @test length(fused) == 2
   @test _dense_train(fused, [x1, y1, x2, y2]) == _dense_train(tt4, [x1, y1, x2, y2])

   unfused = TN.rearrange_siteinds(fused, [[x1], [y1], [x2], [y2]])
   @test _dense_train(unfused, [x1, y1, x2, y2]) == _dense_train(tt4, [x1, y1, x2, y2])
end
```

- [ ] **Step 2: Run the expanded test file to confirm failure**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/tensornetworks/skeleton_surface.jl")'
```

Expected:

- FAIL because the replacement and rearrangement helpers still throw.

- [ ] **Step 3: Implement physical-leg replacement helpers**

Add these helper functions and public methods to `src/TensorNetworks.jl`:

```julia
function _replace_tensor_inds(t::Tensor, replacements::Dict{Index, Index})
   data, tensor_inds = _dense_array(t)
   new_inds = [get(replacements, ind, ind) for ind in tensor_inds]
   return Tensor(copy(data), new_inds)
end

function _validate_site_replacements(tt::TensorTrain, oldinds, newinds)
   length(oldinds) == length(newinds) || throw(ArgumentError("Length mismatch between oldinds and newinds"))
   layout = _physical_site_layout(tt)
   physical = Set(Iterators.flatten(layout))
   length(unique(oldinds)) == length(oldinds) || throw(ArgumentError("oldinds contains duplicates"))
   for (old, new) in zip(oldinds, newinds)
      old in physical || throw(ArgumentError("$(old) is not a physical leg of the tensor train"))
      dim(old) == dim(new) || throw(DimensionMismatch("Replacement dimension mismatch for $(old) and $(new)"))
   end
end

function replace_siteinds(tt::TensorTrain, oldinds, newinds)
   _validate_site_replacements(tt, oldinds, newinds)
   replacements = Dict(zip(oldinds, newinds))
   return TensorTrain([_replace_tensor_inds(t, replacements) for t in tt.data], tt.llim, tt.rlim)
end

function replace_siteinds!(tt::TensorTrain, oldinds, newinds)
   replaced = replace_siteinds(tt, oldinds, newinds)
   tt.data = replaced.data
   return tt
end

function replace_siteinds_part!(tt::TensorTrain, oldinds, newinds)
   _validate_site_replacements(tt, oldinds, newinds)
   replacements = Dict(zip(oldinds, newinds))
   for i in eachindex(tt.data)
      tt.data[i] = _replace_tensor_inds(tt.data[i], replacements)
   end
   return tt
end
```

- [ ] **Step 4: Implement `rearrange_siteinds` using dense contraction and QR**

Add these internal helpers to `src/TensorNetworks.jl`:

```julia
function _contract_prefix!(carry::Union{Nothing, Tensor}, tensors::Vector{Tensor}, target::Set{Index})
   carry_tensor = carry
   while carry_tensor === nothing || !all(ind -> ind in inds(carry_tensor), target)
      isempty(tensors) && throw(ArgumentError("Target physical layout references indices not present in the remaining tensors"))
      next_tensor = popfirst!(tensors)
      carry_tensor = carry_tensor === nothing ? next_tensor : contract(carry_tensor, next_tensor)
      target ⊆ Set(inds(carry_tensor)) && break
   end
   return carry_tensor
end

function _matrix_qr_split(data, left_dims)
   m = prod(left_dims)
   mat = reshape(data, m, :)
   F = qr(mat)
   k = min(size(mat)...)
   Q = Matrix(F.Q)[:, 1:k]
   R = Matrix(F.R[1:k, :])
   return Q, R, k
end

function rearrange_siteinds(tt::TensorTrain, target_layout::Vector{Vector{Index}})
   current_layout = _physical_site_layout(tt)
   Set(Iterators.flatten(target_layout)) == Set(Iterators.flatten(current_layout)) || throw(ArgumentError("target_layout must contain exactly the current physical legs"))

   remaining = copy(tt.data)
   carry = nothing
   new_tensors = Tensor[]
   left_link = nothing

   for sitepos in eachindex(target_layout)
      target = Set(target_layout[sitepos])
      carry = _contract_prefix!(carry, remaining, target)
      data, carry_inds = _dense_array(carry)
      left_inds = left_link === nothing ? copy(target_layout[sitepos]) : vcat([left_link], target_layout[sitepos])
      right_inds = [ind for ind in carry_inds if ind ∉ left_inds]
      sitepos == length(target_layout) && isempty(remaining) && begin
         push!(new_tensors, Tensor(data, carry_inds))
         break
      end
      perm = vcat([findfirst(==(ind), carry_inds) for ind in left_inds], [findfirst(==(ind), carry_inds) for ind in right_inds])
      data_perm = permutedims(data, Tuple(perm))
      Q, R, rank = _matrix_qr_split(data_perm, dim.(left_inds))
      new_link = Index(rank; tags=["Link", "l=$(sitepos)"])
      left_tensor = Tensor(reshape(Q, dim.(left_inds)..., rank), vcat(left_inds, [new_link]))
      right_tensor = Tensor(reshape(R, rank, dim.(right_inds)...), vcat([new_link], right_inds))
      push!(new_tensors, left_tensor)
      carry = right_tensor
      left_link = new_link
   end

   return TensorTrain(new_tensors, 0, length(new_tensors) + 1)
end
```

- [ ] **Step 5: Run the replacement/rearrangement tests**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/tensornetworks/skeleton_surface.jl")'
```

Expected:

- PASS for replacement and rearrangement helper sections.

- [ ] **Step 6: Commit the replacement and rearrangement helpers**

Run:

```bash
git add src/TensorNetworks.jl test/tensornetworks/skeleton_surface.jl docs/src/api.md docs/design/julia_ffi_tensornetworks.md
git commit -m "feat: implement tensornetworks site replacement and regrouping"
```

## Task 5: Implement Diagonalization, Site Matching, And Final Deferred Boundaries

**Files:**

- Modify: `src/TensorNetworks.jl`
- Modify: `test/tensornetworks/skeleton_surface.jl`
- Modify: `docs/src/api.md`
- Modify: `docs/design/julia_ffi_tensornetworks.md`
- Modify: `docs/src/modules.md`

- [ ] **Step 1: Add the diagonalization and site-matching tests**

Append this testset to `test/tensornetworks/skeleton_surface.jl`:

```julia
@testset "TensorNetworks diagonalization and site matching" begin
   TN = Tensor4all.TensorNetworks
   x1 = Index(2; tags=["x=1"])
   x2 = Index(2; tags=["x=2"])
   l12 = Index(2; tags=["Link", "l=1"])

   t1 = Tensor(reshape([1.0, 2.0, 3.0, 4.0], 2, 2), [x1, l12])
   t2 = Tensor(reshape([5.0, 6.0, 7.0, 8.0], 2, 2), [l12, x2])
   psi = TN.TensorTrain([t1, t2], 0, 3)

   mpo = TN.makesitediagonal(psi, "x")
   mpo_dense = _dense_train(mpo, [prime(x1), x1, prime(x2), x2])
   psi_dense = _dense_train(psi, [x1, x2])
   @test mpo_dense[1, 1, 1, 1] == psi_dense[1, 1]
   @test mpo_dense[2, 1, 1, 1] == 0.0

   extracted = TN.extractdiagonal(mpo, "x")
   @test _dense_train(extracted, [x1, x2]) == psi_dense

   sites = [x1, Index(2; tags=["x=extra"]), x2]
   matched = TN.matchsiteinds(psi, sites)
   @test length(matched) == 3
   @test _dense_train(matched, sites)[:, 1, :] == psi_dense

   op = TN.LinearOperator(metadata=(; kind=:placeholder))
   @test_throws Tensor4all.SkeletonPhaseError TN.apply(op, psi)
end
```

- [ ] **Step 2: Run the expanded test file to confirm failure**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/tensornetworks/skeleton_surface.jl")'
```

Expected:

- FAIL because diagonalization and matching helpers are still stubs.

- [ ] **Step 3: Implement diagonalization helpers with Quantics-compatible semantics**

Add these helpers and methods to `src/TensorNetworks.jl`:

```julia
function _asdiagonal_tensor(t::Tensor, site::Index)
   data, tensor_inds = _dense_array(t)
   pos = findfirst(==(site), tensor_inds)
   pos === nothing && throw(ArgumentError("$(site) not found in tensor"))
   primed_site = prime(site)
   perm = vcat(setdiff(1:length(tensor_inds), [pos]), [pos])
   raw = permutedims(data, Tuple(perm))
   rest_shape = size(raw)[1:end-1]
   site_dim = dim(site)
   diagonal = zeros(eltype(data), rest_shape..., site_dim, site_dim)
   for prefix in CartesianIndices(rest_shape), i in 1:site_dim
      diagonal[Tuple(prefix)..., i, i] = raw[Tuple(prefix)..., i]
   end
   new_inds = vcat(tensor_inds[setdiff(1:length(tensor_inds), [pos])], [primed_site, site])
   return Tensor(diagonal, new_inds)
end

function _extract_diagonal_tensor(t::Tensor, site::Index, site2::Index)
   data, tensor_inds = _dense_array(t)
   p1 = findfirst(==(site), tensor_inds)
   p2 = findfirst(==(site2), tensor_inds)
   p1 === nothing && throw(ArgumentError("$(site) not found in tensor"))
   p2 === nothing && throw(ArgumentError("$(site2) not found in tensor"))
   dim(site) == dim(site2) || throw(DimensionMismatch("Diagonal extraction requires equal dimensions"))
   rest = setdiff(1:length(tensor_inds), [p1, p2])
   perm = vcat(rest, [p1, p2])
   raw = permutedims(data, Tuple(perm))
   reduced = zeros(eltype(data), size(raw)[1:end-2]..., dim(site))
   for prefix in CartesianIndices(size(raw)[1:end-2]), i in 1:dim(site)
      reduced[Tuple(prefix)..., i] = raw[Tuple(prefix)..., i, i]
   end
   return Tensor(reduced, vcat(tensor_inds[rest], [site]))
end

function makesitediagonal(tt::TensorTrain, site::Index)
   positions = findsites(tt, site)
   length(positions) == 1 || throw(ArgumentError("$(site) must appear on exactly one tensor"))
   data = copy(tt.data)
   data[only(positions)] = _asdiagonal_tensor(data[only(positions)], site)
   return TensorTrain(data, tt.llim, tt.rlim)
end

function makesitediagonal(tt::TensorTrain, tag::String)
   out = tt
   for site in findallsiteinds_by_tag(tt; tag=tag)
      out = makesitediagonal(out, site)
   end
   return out
end

function extractdiagonal(tt::TensorTrain, tag::String)
   out = TensorTrain(copy(tt.data), tt.llim, tt.rlim)
   for site in findallsiteinds_by_tag(tt; tag=tag)
      pos = only(findsites(out, site))
      out.data[pos] = _extract_diagonal_tensor(out.data[pos], prime(site), site)
   end
   return out
end
```

- [ ] **Step 4: Implement `matchsiteinds` with explicit MPS-like / MPO-like restrictions and keep `apply` deferred**

Add these methods to `src/TensorNetworks.jl`:

```julia
function _is_mps_like_layout(layout)
   return all(length(site) == 1 for site in layout)
end

function _is_mpo_like_layout(layout)
   return all(length(site) == 2 && id(site[1]) == id(noprime(site[2])) for site in layout)
end

function matchsiteinds(tt::TensorTrain, sites::AbstractVector{Index})
   layout = _physical_site_layout(tt)
   base_sites = [noprime(site[1]) for site in layout if !isempty(site)]

   _is_mps_like_layout(layout) || throw(ArgumentError("matchsiteinds only supports one-physical-leg-per-site tensor trains in this phase"))

   positions = [findfirst(==(s), noprime.(sites)) for s in base_sites]
   any(isnothing, positions) && throw(ArgumentError("Current physical legs must be a subset of the target sites"))
   positions = Int[p for p in positions]
   issorted(positions) || throw(ArgumentError("Target sites must preserve ascending chain order"))

   tensors = Tensor[]
   src_pos = 1
   for dst_pos in eachindex(sites)
      if src_pos <= length(positions) && positions[src_pos] == dst_pos
         replacement = replace_siteinds(TensorTrain([tt.data[src_pos]], tt.llim, tt.rlim), [layout[src_pos][1]], [sites[dst_pos]])
         push!(tensors, only(replacement.data))
         src_pos += 1
      else
         filler = zeros(Float64, dim(sites[dst_pos]))
         filler[1] = 1.0
         push!(tensors, Tensor(filler, [sites[dst_pos]]))
      end
   end
   return TensorTrain(tensors, 0, length(tensors) + 1)
end

apply(::LinearOperator, ::TensorTrain; kwargs...) = throw(
   SkeletonPhaseError(
      "TensorNetworks.apply remains deferred: QuanticsTransform still constructs metadata-only LinearOperator values, and this phase does not materialize operators into TensorTrain data.",
   ),
)
```

Do **not** attempt to implement `LinearOperator` materialization in this phase, even though the backend has `t4a_linop_apply`. That backend path currently targets `t4a_treetn`, not the restored `TensorNetworks.TensorTrain` container.

- [ ] **Step 5: Run the full TensorNetworks helper test file**

Run:

```bash
julia --startup-file=no --project=. -e 'include("test/tensornetworks/skeleton_surface.jl")'
```

Expected:

- PASS.

- [ ] **Step 6: Commit diagonalization, site matching, and final deferred boundaries**

Run:

```bash
git add src/TensorNetworks.jl test/tensornetworks/skeleton_surface.jl docs/src/api.md docs/design/julia_ffi_tensornetworks.md docs/src/modules.md
git commit -m "feat: implement tensornetworks diagonal and site matching helpers"
```

## Task 6: Rewrite Public Docs To Match The Implemented State

**Files:**

- Modify: `README.md`
- Modify: `docs/src/index.md`
- Modify: `docs/src/modules.md`
- Modify: `docs/src/api.md`
- Modify: `docs/src/deferred_rework_plan.md`
- Modify: `docs/design/julia_ffi_core.md`
- Modify: `docs/design/julia_ffi_tensornetworks.md`

- [ ] **Step 1: Replace the stale README architecture bullets**

Replace the current `Current Direction` and `What Works In This POC` bullets with this text:

```md
## Current Direction

This branch keeps the restored Julia frontend architecture:

- `TensorNetworks.TensorTrain` is the indexed chain type
- `SimpleTT.TensorTrain{T,N}` is the raw-array tensor-train layer
- `TensorCI.crossinterpolate2` returns `TensorCI2`
- `QuanticsTransform` remains a metadata-only operator-constructor layer
- HDF5 roundtrip is handled in pure Julia through `save_as_mps` / `load_tt`

## What Works In This Phase

- backend-backed `Index` and `Tensor` wrappers over `tensor4all-capi`
- backend-backed tensor contraction in `Tensor4all.contract`
- `TensorNetworks` search, replacement, regrouping, diagonalization, and site-matching helpers
- pure Julia `SimpleTT` compression with `:LU`, `:CI`, and `:SVD`
- pure Julia MPO-MPO contraction for `SimpleTT` with `:naive` and `:zipup`
- pure Julia HDF5 MPS-schema roundtrip through the HDF5 extension
- adopted quantics grid re-exports from `QuanticsGrids.jl`
```

- [ ] **Step 2: Update `docs/design/julia_ffi_core.md` to match the actual C API**

Make these exact corrections:

- Remove `t4a_tensor_new_diag_f64` and `t4a_tensor_new_diag_c64` from the listed exported C API surface.
- Add `t4a_tensor_get_storage_kind` to the tensor getter list.
- Add a note that `t4a_tensor_get_data_f64` / `t4a_tensor_get_data_c64` return data in column-major order and can materialize diagonal storage into dense vectors.
- Add `t4a_last_error_message` and the status-code contract to the FFI/error-handling section.

- [ ] **Step 3: Update `docs/design/julia_ffi_tensornetworks.md` with the fixed physical-leg convention**

Add these exact bullets under the helper-surface section:

```md
- A physical leg is any index that appears on exactly one tensor in `tt.data`.
- A link leg is any index that appears on exactly two adjacent tensors.
- Tag-based helper APIs operate on unprimed physical legs only.
- `findallsiteinds_by_tag(tt; tag="x")` follows `x=1`, `x=2`, ... ordering and stops at the first missing tag number.
- `rearrange_siteinds` may change the number of site tensors by regrouping physical legs and refactorizing the chain with QR.
- `matchsiteinds` is intentionally restricted to one-physical-leg-per-site layouts in this phase.
```

- [ ] **Step 4: Update the public docs pages and deferred-plan wording**

Ensure these exact facts appear in the public docs:

- `TensorCI.crossinterpolate2` returns `TensorCI2`, not `SimpleTT.TensorTrain`.
- `Index` and `Tensor` are backend wrappers, not metadata-only structs.
- `TensorNetworks.apply` remains deferred because `LinearOperator` materialization is still metadata-only.
- HDF5 support is real and pure Julia.

- [ ] **Step 5: Build the docs and inspect for stale TreeTN-first language**

Run:

```bash
julia --startup-file=no --project=docs docs/make.jl
grep -R -n 'TreeTensorNetwork still exists\|TensorCI returns SimpleTT\|row-major' README.md docs/src docs/design
```

Expected:

- The docs build succeeds.
- The grep returns no stale claims.

- [ ] **Step 6: Commit the documentation rewrite**

Run:

```bash
git add README.md docs/src/index.md docs/src/modules.md docs/src/api.md docs/src/deferred_rework_plan.md docs/design/julia_ffi_core.md docs/design/julia_ffi_tensornetworks.md
git commit -m "docs: align public docs with phase 1 backend enablement"
```

## Task 7: Final Verification

**Files:**

- Test only; no planned file edits

- [ ] **Step 1: Run the full test suite**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected:

- PASS.

- [ ] **Step 2: Run the package-load smoke test**

Run:

```bash
julia --startup-file=no --project=. -e 'using Tensor4all'
```

Expected:

- PASS.

- [ ] **Step 3: Run the docs build**

Run:

```bash
julia --startup-file=no --project=docs docs/make.jl
```

Expected:

- PASS.

- [ ] **Step 4: Inspect the final diff**

Run:

```bash
git status --short
git diff --stat
```

Expected:

- Only the files named in this plan have changed.

- [ ] **Step 5: Commit the verification checkpoint**

Run:

```bash
git add -A
git commit -m "test: verify phase 1 backend enablement end-to-end"
```

## Self-Review Checklist For The Implementing Agent

- `Manifest.toml` was refreshed by `Pkg.resolve()` / `Pkg.instantiate()`, not by manual editing.
- `src/Core/CAPI.jl` contains no public exports.
- `Index` and `Tensor` are `mutable struct`s so finalizers can zero their pointers.
- `Tensor` dense extraction assumes column-major order and does **not** transpose the backend data.
- `TensorNetworks` physical-leg helpers operate only on indices that appear on exactly one tensor.
- `findallsiteinds_by_tag` uses `tag=1`, `tag=2`, ... semantics and rejects tags containing `=`.
- `rearrange_siteinds` is the only helper in this phase that may change tensor count.
- `matchsiteinds` is explicitly restricted to MPS-like layouts in this phase.
- `TensorNetworks.apply` still throws a clear deferred error.
- HDF5 tests no longer read `Tensor` struct fields directly.