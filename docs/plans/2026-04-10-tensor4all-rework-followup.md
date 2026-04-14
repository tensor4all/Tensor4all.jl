# Tensor4all Repo-Only Backend Enablement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `Tensor4all.jl` from a review-only skeleton into a repo-local, backend-enabled Julia frontend for core tensor, index, and TreeTN operations, without modifying `tensor4all-rs` or its C API yet.

**Architecture:** Keep the public Julia API centered on `TreeTensorNetwork{V}`, `TensorTrain`, `MPS`, and `MPO`. Rework `Index` and `Tensor` into thin wrappers over the currently available C API, and represent `TreeTensorNetwork{V}` as a Julia label-mapping layer over a backend `t4a_treetn` handle. Keep quantics transform materialization, QTCI execution, and HDF5 persistence deferred until the later `tensor4all-rs` / C-API rework.

**Tech Stack:** Julia 1.9+, existing `tensor4all-rs` C API as-is, `QuanticsGrids.jl`, `ITensors.jl`, Documenter.jl, `Test`.

---

## Summary

This plan replaces the old skeleton-building plan. The skeleton already exists in
the repository. The next phase should therefore stop adding placeholder surface
area and instead make the existing core and TreeTN APIs real where the current
C API already supports them.

This phase is intentionally limited to the `Tensor4all.jl` repository. It must
not modify:

- `../tensor4all-rs`
- the Rust C API
- any sibling downstream repository such as `BubbleTeaCI`

This phase must deliver:

- a consistent local Julia project state (`Manifest.toml` fixed so `Pkg.test()` runs)
- real backend wrappers for `Index`
- real backend wrappers for `Tensor`
- real backend wrappers for `TreeTensorNetwork{V}` and chain aliases
- minimal real `ITensors` conversions for `Index` and `Tensor`
- updated docs and tests for the newly real surfaces

This phase must keep deferred:

- quantics transform materialization and execution
- QTCI execution
- HDF5 persistence and round-trips
- any new C-API entrypoint or Rust-side behavior

## Locked Decisions

These decisions are fixed by this plan and must not be reopened during
implementation unless the user explicitly changes them.

1. **No upstream Rust changes in this phase**
   - The implementation may call only the C API that already exists today.
   - If a Julia-level feature would require a new C-API function, that feature stays deferred.

2. **Keep the TreeTN-general public model**
   - `TreeTensorNetwork{V}` remains the primary public network type.
   - `TensorTrain = TreeTensorNetwork{Int}` remains the primary chain alias.
   - `MPS` and `MPO` remain aliases of `TensorTrain`, distinguished only by runtime structure checks.

3. **Use thin backend-handle wrappers for core types**
   - `Index` becomes a thin owned wrapper around a backend index handle.
   - `Tensor` becomes a thin owned wrapper around a backend tensor handle.
   - No long-lived Julia metadata cache is added for either type.

4. **Keep the package in a partially implemented phase**
   - `SKELETON_PHASE` stays exported and remains `true` in this phase.
   - `SkeletonNotImplemented` stays in use only for APIs that remain intentionally deferred.
   - The docs must explain that the package is now partially backend-enabled rather than fully stubbed.

5. **Keep quantics transforms deferred**
   - `QuanticsTransform` constructors remain metadata-level descriptors.
   - `materialize_transform` remains deferred and must keep throwing a clear deferred-feature error.
   - `QTCIOptions`, `QTCIDiagnostics`, and `QTCIResultPlaceholder` remain placeholders.

6. **Keep HDF5 deferred**
   - `Tensor4allHDF5Ext` remains stub-only in this phase.
   - The HDF5 tests remain deferred-behavior tests rather than real round-trip tests.

7. **Implement only minimal real `ITensors` extension behavior**
   - Implement real `Index` and `Tensor` conversions only.
   - Do not implement `TreeTensorNetwork` / `MPS` / `MPO` conversions in this phase.

## Target File Changes

### Source files

- Modify: `src/Tensor4all.jl`
  - include any new internal helper file
  - keep the current public export set unless this plan explicitly says otherwise
  - update the module docstring to describe the partially enabled state
- Modify: `src/Core/Errors.jl`
  - keep current public errors
  - update wording only where needed to distinguish deferred features from fully stubbed layers
- Modify: `src/Core/Backend.jl`
  - keep lazy backend loading
  - do not `dlopen` during `using Tensor4all`
- Create: `src/Core/CAPI.jl`
  - internal-only FFI helpers
- Modify: `src/Core/Index.jl`
  - convert from metadata-only struct to backend-handle wrapper
- Modify: `src/Core/Tensor.jl`
  - convert from metadata-only struct to backend-handle wrapper
  - add Julia-side dense extraction and axis-permutation helpers needed for `prime` and `swapinds`
- Modify: `src/TreeTN/TreeTensorNetwork.jl`
  - convert from metadata-only struct to a hybrid Julia-metadata + backend-handle wrapper
- Modify: `src/Quantics/Transforms.jl`
  - keep deferred behavior but make the deferred boundary explicit in docstrings and errors
- Leave functionally unchanged: `src/Quantics/QuanticsGridsBridge.jl`
  - no new exports in this phase
- Leave functionally unchanged: `src/Quantics/QTCI.jl`
  - placeholder types remain placeholders

### Extension files

- Modify: `ext/Tensor4allITensorsExt.jl`
  - replace stubs with real `Index` and `Tensor` conversions
- Keep stubbed: `ext/Tensor4allHDF5Ext.jl`

### Test files

- Modify: `test/runtests.jl`
- Modify: `test/core/bootstrap.jl`
- Modify: `test/core/index.jl`
- Modify: `test/core/tensor.jl`
- Modify: `test/ttn/tree_tensor_network.jl`
- Modify: `test/quantics/transforms.jl`
- Modify: `test/extensions/itensors_ext.jl`
- Keep deferred-style test: `test/extensions/hdf5_ext.jl`

### Documentation files

- Modify: `README.md`
- Modify: `docs/src/index.md`
- Modify: `docs/src/modules.md`
- Modify: `docs/src/api.md`
- Modify: `docs/src/design_documents.md`
- Modify: `docs/src/deferred_rework_plan.md`
- Modify only if necessary: `docs/make.jl`

### Project files

- Modify: `Manifest.toml`
- Leave `Project.toml` unchanged unless manifest refresh proves that a project-level dependency declaration is actually wrong

## Public API Contract To Implement

### `Index`

`Index` becomes:

```julia
mutable struct Index
    ptr::Ptr{Cvoid}
end
```

The concrete field name may differ, but the representation must remain a thin
owned wrapper around a backend handle with a finalizer.

Behavior:

- `Index(dim; tags=String[], plev=0, id=nothing)` requires a backend.
- `dim > 0` is validated in Julia before the C call.
- `tags` are normalized to `sort!(unique!(collect(String.(tags))))` before sending to the backend.
- `tags(i)` returns normalized, sorted strings.
- `plev >= 0` is validated in Julia.
- if `id === nothing`, use backend-generated ID creation
- if `id !== nothing`, use the explicit-ID constructor path
- `sim(i)` returns a new backend index with same `dim`, normalized `tags`, and same `plev`, but a fresh ID
- `prime(i, n)` clones and raises `plev` by `n`; the resulting `plev` must remain nonnegative
- `noprime(i)` clones and sets `plev = 0`
- `setprime(i, n)` clones and sets `plev = n`
- `hastag(i, tag)` queries the backend
- `==(a, b)` and `hash(a)` are based on `(dim(a), id(a), tags(a), plev(a))`, not pointer identity
- `show(io, i)` preserves the current public format

### `Tensor`

`Tensor` becomes:

```julia
mutable struct Tensor
    ptr::Ptr{Cvoid}
end
```

Behavior:

- `Tensor(data::Array{Float64,N}, inds)` and `Tensor(data::Array{ComplexF64,N}, inds)` are the supported dense constructors in this phase.
- Generic `AbstractArray` inputs still throw the existing contiguity-guidance error unless explicitly `collect`ed first.
- Constructor validation in Julia:
  - `length(inds) == ndims(data)`
  - `Tuple(dim.(inds)) == size(data)`
  - `data` is contiguous in memory
- `inds(t)` queries backend indices and returns newly wrapped `Index` objects in backend order.
- `rank(t)` and `dims(t)` query the backend every time; no cache is added.
- Add an internal helper `_dense_array(t)` that returns `(data, inds)` where `data` is a Julia dense array with the correct real or complex element type.
- `prime(t, n)` is implemented in Julia by:
  - extracting dense data
  - cloning and priming the indices
  - reconstructing a new backend tensor
- `swapinds(t, a, b)` is implemented in Julia by:
  - extracting dense data and index order
  - checking that `a` and `b` each occur exactly once
  - permuting both axes and index order
  - reconstructing a new backend tensor
- `contract(a, b)` uses the existing backend tensor-tensor contraction API and no longer throws `SkeletonNotImplemented`

### `TreeTensorNetwork{V}`

`TreeTensorNetwork{V}` becomes a hybrid wrapper:

```julia
mutable struct TreeTensorNetwork{V}
    ptr::Ptr{Cvoid}
    vertex_order::Vector{V}
    vertex_to_backend::Dict{V,Int}
    backend_to_vertex::Dict{Int,V}
    adjacency::Dict{V,Vector{V}}
    site_index_map::Dict{V,Vector{Index}}
    link_index_map::Dict{Tuple{V,V},Index}
end
```

The exact field names may differ, but all of this information must be stored.

Constructor contract:

```julia
TreeTensorNetwork(
    tensors;
    vertex_order=nothing,
    adjacency,
    siteinds,
    linkinds,
)
```

Rules:

- If `vertex_order === nothing`:
  - for the chain case with `V == Int` and keys `1:n`, use `1:n`
  - otherwise throw `ArgumentError("vertex_order must be specified for non-chain or non-Int TreeTensorNetwork construction in this phase")`
- Validate in Julia before touching the backend:
  - `Set(keys(tensors)) == Set(vertex_order)`
  - `Set(keys(adjacency)) == Set(vertex_order)`
  - `Set(keys(siteinds)) == Set(vertex_order)`
  - adjacency is symmetric
  - every edge in `adjacency` has exactly one corresponding link index
  - each tensor contains exactly the declared site indices for its vertex plus the link indices for its incident edges
- Build the backend TreeTN by passing tensors in `vertex_order`, which therefore maps backend node names `0:(n-1)` to that order.
- Store `link_index_map` in both directions: both `(a, b)` and `(b, a)` must exist after construction.
- After backend construction, query backend site and link indices and verify they match the supplied Julia metadata. If not, release the backend handle and throw.

Accessor contract:

- `vertices(ttn)` returns `copy(ttn.vertex_order)`
- `neighbors(ttn, v)` returns `copy(ttn.adjacency[v])`
- `siteinds(ttn, v)` returns `copy(ttn.site_index_map[v])`
- `linkind(ttn, a, b)` looks up `ttn.link_index_map[(a, b)]` directly

Predicate contract:

- `is_chain(ttn)` is true exactly when:
  - `V == Int`
  - `vertex_order == collect(1:length(vertex_order))`
  - degree pattern is the expected chain pattern
- `is_mps_like(ttn)` means every vertex has exactly one site index
- `is_mpo_like(ttn)` means every vertex has exactly two site indices

Operation contract:

- `orthogonalize!(ttn, v)`:
  - validates `v ∈ vertices(ttn)`
  - maps `v` to backend node name and calls backend orthogonalization
  - refreshes `site_index_map` and `link_index_map` afterward
- `truncate!(ttn; rtol=0.0, cutoff=0.0, maxdim=0)`:
  - passes `rtol`, `cutoff`, and `maxdim` directly to backend semantics
  - refreshes `site_index_map` and `link_index_map` afterward
- `inner(a, b)`:
  - requires identical `vertex_order` and identical adjacency
  - returns `Float64` if the imaginary part is zero, otherwise `ComplexF64`
- `norm(ttn)`:
  - returns `Float64`
  - may mutate the backend canonical form internally, but must not change public vertex metadata
- `to_dense(ttn)`:
  - returns a `Tensor`
  - does **not** try to reorder backend output axes in this phase
  - the returned tensor index order is therefore backend order and must be documented as such
- `evaluate(ttn, assignments::AbstractDict{Index,<:Integer})`:
  - requires an assignment for every site index exactly once
  - rejects missing keys, duplicate logical assignments, and out-of-range values in Julia before the C call
  - returns `Float64` if the imaginary part is zero, otherwise `ComplexF64`
- `evaluate(tt::TensorTrain, values::AbstractVector{<:Integer})` and `evaluate(tt::TensorTrain, values::Integer...)`:
  - require `is_chain(tt)`
  - require `is_mps_like(tt)`
  - use `vertex_order == 1:n`
  - map the values to the single site index on each chain vertex in order
- `contract(a, b; alg=:zipup, rtol=0.0, cutoff=0.0, maxdim=0)`:
  - accepts `alg` as `:zipup`, `:fit`, `:naive`, or the strings `"zipup"`, `"fit"`, `"naive"`
  - requires `a.vertex_order == b.vertex_order`
  - requires identical adjacency
  - reuses the left operand’s `vertex_order` and adjacency
  - after the backend call, refreshes `site_index_map` and `link_index_map`
  - if backend contraction changes node count or topology, throw `ArgumentError("TreeTN contraction result topology differs from input topology; this case is deferred in this phase")`

### Quantics

Behavior locked for this phase:

- do not widen the `QuanticsGrids.jl` re-export
- do not implement real transform materialization
- do not implement QTCI execution
- keep `materialize_transform` throwing `SkeletonNotImplemented(:materialize_transform, :quantics)`
- update the error text and docstrings so they explicitly say the feature is deferred until the later `tensor4all-rs` / C-API rework

### Extensions

`Tensor4allITensorsExt.jl` must implement:

- `to_itensor(::Tensor4all.Index) -> ITensors.Index`
- `from_itensor(::ITensors.Index) -> Tensor4all.Index`
- `to_itensor(::Tensor4all.Tensor) -> ITensors.ITensor`
- `from_itensor(::ITensors.ITensor) -> Tensor4all.Tensor`

Rules:

- preserve `dim`, `id`, normalized `tags`, and `plev` in `Index` round-trips
- preserve tensor index order in `Tensor` round-trips
- do not add `TreeTensorNetwork` / `MPS` / `MPO` conversions in this phase

`Tensor4allHDF5Ext.jl` stays deferred:

- keep `save_hdf5` and `load_hdf5` as explicit deferred-feature errors
- update wording only if needed for clarity

## Task Breakdown

### Task 0: Refresh the Julia project state

**Files:**

- Modify: `Manifest.toml`

- [ ] Refresh the manifest so it matches the current `Project.toml`, including `QuanticsGrids` and both package extensions.
- [ ] Verify that `julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'` succeeds.
- [ ] Verify that the previous failure mode about `QuanticsGrids` missing from the manifest is gone before continuing.

**Acceptance criteria:**

- `Pkg.instantiate()` succeeds from the package root.
- `Manifest.toml` includes `QuanticsGrids` and the declared extension metadata.

### Task 1: Add the internal FFI helper layer

**Files:**

- Create: `src/Core/CAPI.jl`
- Modify: `src/Tensor4all.jl`
- Modify: `src/Core/Backend.jl`
- Modify: `src/Core/Errors.jl`
- Modify: `test/core/bootstrap.jl`

- [ ] Add internal helpers for:
  - status checking
  - retrieving `t4a_last_error_message()`
  - query-then-fill buffers for strings and vectors
  - handle finalizers and release helpers
  - normalized tag handling
  - real/complex buffer marshaling
- [ ] Keep all helper names internal-only, prefixed with `_` where appropriate.
- [ ] Keep `require_backend()` lazy and unchanged at the public API level.
- [ ] Update bootstrap tests so they still verify:
  - `using Tensor4all` works without the backend library present
  - backend-backed operations fail only on first actual use

**Acceptance criteria:**

- importing the package does not call `dlopen`
- backend-backed constructors and operations report Rust-side error text when the C API returns a failure status

### Task 2: Rework `Index` into a real backend wrapper

**Files:**

- Modify: `src/Core/Index.jl`
- Modify: `src/Tensor4all.jl`
- Modify: `test/core/index.jl`
- Modify: `docs/src/api.md`

- [ ] Convert `Index` from a metadata struct into an owned backend-handle wrapper with a finalizer.
- [ ] Implement the constructor paths with Julia-side validation and tag normalization.
- [ ] Implement `dim`, `id`, `tags`, `plev`, `hastag`, `sim`, `prime`, `noprime`, and `setprime`.
- [ ] Keep `replaceind`, `replaceinds`, `commoninds`, and `uniqueinds` as pure Julia helpers over `Vector{Index}`.
- [ ] Preserve equality and hashing by metadata, not pointer identity.
- [ ] Add or update docstrings so every exported `Index`-related symbol touched here has a `# Examples` section.

**Acceptance criteria:**

- `test/core/index.jl` passes using the real backend wrapper
- `sim(i)` preserves dimension, tags, and `plev`, but changes `id`
- tag order is normalized and deterministic in tests

### Task 3: Rework `Tensor` into a real backend wrapper

**Files:**

- Modify: `src/Core/Tensor.jl`
- Modify: `src/Tensor4all.jl`
- Modify: `test/core/tensor.jl`
- Modify: `docs/src/api.md`

- [ ] Convert `Tensor` from a metadata struct into an owned backend-handle wrapper with a finalizer.
- [ ] Implement real dense constructors for `Float64` and `ComplexF64`.
- [ ] Keep the contiguity error for unsupported `AbstractArray` inputs, with the current actionable message style from `AGENTS.md`.
- [ ] Add internal `_dense_array(t)` and storage-kind helpers for tests and extensions.
- [ ] Implement `inds`, `rank`, and `dims` as backend queries.
- [ ] Implement `prime(t, n)` in Julia via dense extraction + index rewrite + reconstruction.
- [ ] Implement `swapinds(t, a, b)` in Julia via dense extraction + axis permutation + reconstruction.
- [ ] Implement real tensor-tensor `contract(a, b)` through the current backend C API.
- [ ] Add or update docstrings so every exported `Tensor`-related symbol touched here has a `# Examples` section.

**Acceptance criteria:**

- the tensor-contraction test uses a deterministic small fixture and compares against a known dense reference
- `swapinds` changes both metadata order and numeric axis order
- real and complex constructor paths both round-trip through `_dense_array`

### Task 4: Rework `TreeTensorNetwork{V}` into a hybrid Julia/backend wrapper

**Files:**

- Modify: `src/TreeTN/TreeTensorNetwork.jl`
- Modify: `src/Tensor4all.jl`
- Modify: `test/ttn/tree_tensor_network.jl`
- Modify: `docs/src/api.md`

- [ ] Add `vertex_order` as an explicit constructor keyword.
- [ ] Make non-chain or non-`Int` construction require `vertex_order`.
- [ ] Build the backend TreeTN using tensors in `vertex_order`.
- [ ] Store `vertex_order`, `vertex_to_backend`, `backend_to_vertex`, adjacency, site map, and bidirectional link map in Julia.
- [ ] Add internal metadata-refresh helpers after backend operations.
- [ ] Implement `vertices`, `neighbors`, `siteinds`, `linkind`, `is_chain`, `is_mps_like`, and `is_mpo_like`.
- [ ] Implement real `orthogonalize!`, `truncate!`, `inner`, `norm`, `to_dense`, `evaluate`, and `contract`.
- [ ] Keep chain-only runtime checks explicit and descriptive.
- [ ] Document the phase-specific contraction restriction: both operands must have identical topology and identical `vertex_order`.
- [ ] Add or update docstrings so every exported TreeTN-related symbol touched here has a `# Examples` section.

**Acceptance criteria:**

- the TreeTN constructor rejects inconsistent adjacency/site/link metadata before touching the backend
- `vertices(ttn)` preserves explicit `vertex_order`
- `norm`, `inner`, `to_dense`, and `evaluate` no longer throw `SkeletonNotImplemented`
- `contract` rejects topology-changing cases with a Julia-side `ArgumentError`

### Task 5: Keep quantics deferred, but make the deferred boundary explicit

**Files:**

- Modify: `src/Quantics/Transforms.jl`
- Modify: `test/quantics/transforms.jl`
- Modify: `README.md`
- Modify: `docs/src/index.md`
- Modify: `docs/src/modules.md`
- Modify: `docs/src/deferred_rework_plan.md`

- [ ] Keep the current transform constructors metadata-only.
- [ ] Keep `materialize_transform` deferred.
- [ ] Update docstrings and deferred errors so they explicitly point to the later `tensor4all-rs` / C-API rework as the reason they remain deferred.
- [ ] Keep `src/Quantics/QTCI.jl` placeholder-only in this phase.
- [ ] Do not widen the `QuanticsGrids.jl` re-export set.

**Acceptance criteria:**

- quantics re-export tests still pass
- transform tests still assert deferred behavior
- docs no longer imply that transform materialization is part of this phase

### Task 6: Replace only the `ITensors` stubs with real conversions

**Files:**

- Modify: `ext/Tensor4allITensorsExt.jl`
- Modify: `test/extensions/itensors_ext.jl`
- Leave stubbed: `ext/Tensor4allHDF5Ext.jl`
- Leave deferred-style: `test/extensions/hdf5_ext.jl`

- [ ] Implement real `Index` conversion both ways.
- [ ] Implement real `Tensor` conversion both ways.
- [ ] Keep `HDF5` functionality stubbed in this phase.
- [ ] Make the extension tests reflect the new boundary exactly:
  - `ITensors` conversions are real
  - `HDF5` calls remain deferred

**Acceptance criteria:**

- `test/extensions/itensors_ext.jl` verifies actual round-trips
- `test/extensions/hdf5_ext.jl` still verifies explicit deferred errors

### Task 7: Update docs and run full verification

**Files:**

- Modify: `README.md`
- Modify: `docs/src/index.md`
- Modify: `docs/src/modules.md`
- Modify: `docs/src/api.md`
- Modify: `docs/src/design_documents.md`
- Modify: `docs/src/deferred_rework_plan.md`
- Modify only if needed: `docs/make.jl`

- [ ] Update the package status text from “all skeleton” to “core and TreeTN backend-enabled; quantics materialization, QTCI, and HDF5 still deferred”.
- [ ] Update API reference pages so they describe the real `Index`, `Tensor`, and TreeTN behavior.
- [ ] Keep the owner/re-export boundary around `QuanticsGrids.jl` explicit.
- [ ] Keep `TTFunction` explicitly out of scope and owned by `BubbleTeaCI`.
- [ ] Ensure every exported symbol touched in this phase has a docstring with `# Examples`.
- [ ] Run the full verification suite.

**Verification commands:**

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --startup-file=no --project=docs docs/make.jl
```

If running on `primerose`, use the Docker workaround from `AGENTS.md` instead of direct local testing.

**Acceptance criteria:**

- `Pkg.test()` passes
- `docs/make.jl` passes
- README and docs do not claim any behavior that this phase still defers

## Specific Test Fixtures To Use

These fixtures are locked to reduce implementation-time choice churn.

### Core tensor fixture

Use a deterministic tensor-contraction test of:

- `A :: (2, 3)` with indices `[i, j]`
- `B :: (3, 4)` with indices `[j, k]`
- expected result shape `(2, 4)`
- expected dense values equal to matrix multiplication `A * B`

### TreeTN evaluation fixture

Use a 2-site chain MPS-like network with deterministic small tensors so that:

- `evaluate(tt, [1, 1])`
- `evaluate(tt, [1, 2])`
- `evaluate(tt, [2, 1])`
- `evaluate(tt, [2, 2])`

can all be checked against direct dense reference values.

### TreeTN contraction restriction fixture

Use two identical-topology 3-site chains for the passing case, and one topology
mismatch case that must fail in Julia before the backend call.

### ITensors extension fixture

Use:

- one `Index` with nontrivial `id`, `tags`, and `plev`
- one small dense real `Tensor`
- one small dense complex `Tensor`

and verify round-trip preservation of metadata and dense values.

## Explicit Non-Goals For This Phase

- no changes to `tensor4all-rs`
- no new C-API entrypoints
- no transform materialization
- no QTCI execution
- no HDF5 persistence
- no TreeTN / MPS / MPO `ITensors` conversions
- no new top-level public exports beyond the optional `vertex_order` constructor keyword
- no attempt to solve topology-changing TreeTN contraction results

## Final Deliverable Checklist

The implementation phase is complete only when all of the following are true:

- [ ] `Manifest.toml` is synced and `Pkg.instantiate()` works
- [ ] `Index` is a real backend wrapper
- [ ] `Tensor` is a real backend wrapper
- [ ] `TreeTensorNetwork{V}` is a real hybrid Julia/backend wrapper
- [ ] `Tensor` contraction works through the backend
- [ ] TreeTN `orthogonalize!`, `truncate!`, `inner`, `norm`, `to_dense`, `evaluate`, and restricted `contract` work
- [ ] quantics transforms remain explicitly deferred
- [ ] `ITensors` `Index` and `Tensor` conversions are real
- [ ] HDF5 remains explicitly deferred
- [ ] README and docs describe the new partial-enable boundary accurately
- [ ] `Pkg.test()` passes
- [ ] `docs/make.jl` passes
