# Issue #66 BubbleTeaCI Follow-up Spec and Design

## Summary

Add the remaining TensorTrain, Tensor, and `ITensorCompat` APIs needed by the
BubbleTeaCI Tensor4all backend so downstream code can stop reaching into
`TensorTrain.data`, `llim`, `rlim`, dense one-hot workarounds, and
Tensor4all-native truncation controls.

The implementation should land as one Tensor4all.jl PR based on `origin/main`,
organized as phased commits. Each phase must be independently testable and
should preserve the public layer split in `AGENTS.md`.

## Motivation

BubbleTeaCI is porting code that naturally looks like ITensors.jl /
ITensorMPS.jl code. The current Tensor4all.jl surface already contains much of
the indexed chain machinery, but downstream glue is still needed for:

- local TensorTrain block mutation and canonical-window repair;
- metadata-only index rewriting that preserves storage and canonical metadata;
- lazy index fixing, projection, summation, and structured identity/delta
  tensors;
- cutoff-only ITensors-style truncation entry points;
- scalar and empty-site MPS behavior;
- raw block MPS/MPO construction; and
- public tag formatting for display and debugging.

Keeping these upstream reduces downstream coupling and keeps BubbleTeaCI close
to ITensors-style source structure.

## Current origin/main baseline

This design starts from `origin/main` at
`e5dc2b2e859839a28fbdb39b3d94193dd960f334`.

Relevant baseline facts:

- Issue #60 is already merged: `Tensor` and index-collection
  `replaceind` / `replaceinds` APIs exist.
- `TensorNetworks.TensorTrain` is the public indexed chain type with
  `data::Vector{Tensor}`, `llim::Int`, and `rlim::Int`.
- `Base.setindex!(tt, tensor, i)` already performs local canonical
  invalidation:

  ```julia
  tt.llim = min(tt.llim, i - 1)
  tt.rlim = max(tt.rlim, i + 1)
  ```

- `replace_siteinds!` currently reconstructs affected tensors instead of
  rewriting index metadata in place.
- `Tensor` currently stores dense Julia data and a backend handle, but Julia
  does not yet expose Rust structured tensor storage metadata.
- `ITensorCompat`, `onehot`, `delta`, `fixinds`, `suminds`, `projectinds`, and
  `tagstring` are not present on `origin/main`.
- The local tensor4all-rs repository has structured storage C API support
  (`t4a_tensor_new_diag_*`, `t4a_tensor_new_structured_*`,
  `t4a_tensor_storage_kind`, payload metadata, and axis-class accessors), but
  Tensor4all.jl still needs a pin bump and Julia wrappers to consume it.

## Goals

- Provide a public mutation contract so downstream packages do not mutate
  `tt.data`, `tt.llim`, or `tt.rlim` directly.
- Document and test local canonical invalidation for single-block replacement.
- Make topology-changing TensorTrain operations available through public
  helpers or Base methods that invalidate canonical metadata consistently.
- Preserve tensor values, backend handles, structured storage metadata, and
  canonical bounds for metadata-only index rewrites.
- Add lazy index-fixing and summation primitives that avoid dense one-hot
  vector construction for slicing and evaluation workflows.
- Expose ergonomic structured diagonal / identity constructors usable for
  dummy-site insertion, delta tensors, and projectors.
- Add an opt-in `Tensor4all.ITensorCompat` facade for cutoff-only ITensors-like
  MPS/MPO operations.
- Define scalar and empty-site MPS behavior in `ITensorCompat`.
- Keep generic operator ownership in `TensorNetworks`; `QuanticsTransform`
  remains only the quantics-specific constructor layer.

## Non-goals

- Do not make `TensorNetworks.TensorTrain` itself an `MPS` or `MPO` type.
  MPS-like versus MPO-like interpretation remains structural at the
  `TensorNetworks` layer.
- Do not change `TensorNetworks.siteinds(::TensorTrain)` to sometimes return a
  flat vector. Flat MPS site indices belong in `ITensorCompat.MPS`.
- Do not add TensorTrain-based automatic binding for
  `LinearOperator` input/output spaces.
- Do not add Rust C API functions for Julia-composable operations such as
  localized block mutation, index replacement, or chain-level slicing.
- Do not require BubbleTeaCI-specific grid types in Tensor4all.jl constructors.
  BubbleTeaCI should pass explicit `Vector{Index}` values.

## Proposed Approach

Use one PR with phased commits. The phases are ordered so foundational Core and
TensorNetworks behavior is available before the compatibility facade is added.

### Phase 1: TensorTrain Mutation Contract

Keep the existing local invalidation semantics for `setindex!`, document them,
and add explicit tests. Replacing one block invalidates only that block:

```julia
tt.llim = min(tt.llim, i - 1)
tt.rlim = max(tt.rlim, i + 1)
```

Add public helpers:

- `invalidate_canonical!(tt)` for full invalidation;
- `invalidate_canonical!(tt, i)` for local single-site invalidation;
- `replaceblock!(tt, i, tensor)` as the public chain-level block replacement
  helper that returns `tt`;
- `insert_site!(tt, position, tensor)`;
- `delete_site!(tt, position)`.

Also implement and document Base vector-like methods where missing:

- `Base.insert!(tt::TensorTrain, position, tensor)`;
- `Base.deleteat!(tt::TensorTrain, position)`;
- `Base.push!(tt::TensorTrain, tensor)`;
- `Base.pushfirst!(tt::TensorTrain, tensor)`.

Topology-changing operations should use full invalidation:

```julia
tt.llim = 0
tt.rlim = length(tt) + 1
```

This keeps local replacement precise while treating shape/topology changes
conservatively.

### Phase 2: Metadata-Only Index and Structured Storage Primitives

Move metadata-only index replacement onto the already merged Tensor-level
`replaceind!` / `replaceinds!` primitives whenever possible. For
`replace_siteinds!`, update tensors in place instead of reconstructing dense
`Tensor` values. Preserve `llim` and `rlim` because pure index metadata changes
do not alter canonicality.

Add public tag formatting helpers in Core:

- `tagstring(index::Index)`;
- `tagstring(tags::AbstractVector{<:AbstractString})`.

Use `tagstring` from `show(::Index)` so downstream display code can share the
same formatting without field access.

Expose Rust structured tensor storage through Julia wrappers:

- storage kind: dense, diagonal, structured;
- payload rank, payload dimensions, payload strides, payload length;
- axis-class metadata;
- compact payload readback for `Float64` and `ComplexF64`;
- constructors for diagonal and general structured tensors.

Add ergonomic constructors on top of those wrappers:

- `delta(i::Index, j::Index, more::Index...; T=Float64)`;
- `diagtensor(values, inds::Vector{Index})`;
- `identity_tensor(i::Index, j::Index; T=Float64)`.

`delta` and identity constructors should use structured diagonal storage when
the bumped Rust pin provides it. Dense fallback should be limited to tests or
backend-unavailable paths and should be documented as a fallback, not the target
representation.

### Phase 3: Lazy Fix, Sum, Project, and Dummy-Site Helpers

Add lazy index manipulation primitives for both `Tensor` and
`TensorNetworks.TensorTrain`:

- `fixinds(x, index => value, ...)` removes fixed indices;
- `suminds(x, indices...)` sums out listed indices;
- `projectinds(x, index => values, ...)` restricts indices to listed values and
  keeps a smaller replacement index.

For `Tensor`, these can be implemented with Julia slicing, summation, and
metadata rewriting. For `TensorTrain`, dispatch should locate the affected site
tensors and apply the Tensor-level operation locally. Non-mutating variants
return a copied train with affected canonical bounds invalidated locally; any
mutating variants should return the mutated train.

Add public dummy-site helpers:

- `insert_identity!(tt, newsite, position)` for MPS-like chains;
- `identity_link_tensor(left, right, site)` or an equivalent constructor for
  the tensor inserted between neighboring blocks.

These helpers should use structured diagonal / identity tensors rather than
dense `diagm` or dense one-hot arrays when backend support is available.

### Phase 4: ITensorCompat Facade

Add an opt-in module:

```julia
module Tensor4all.ITensorCompat

mutable struct MPS
    tt::TensorNetworks.TensorTrain
end

mutable struct MPO
    tt::TensorNetworks.TensorTrain
end

end
```

`MPS` and `MPO` are semantic wrappers around `TensorNetworks.TensorTrain`.
They should validate structure on construction:

- `MPS(tt)` accepts one site-like index per tensor, or an empty/scalar train;
- `MPO(tt)` accepts two site-like indices per tensor;
- failures use `ArgumentError` with the tensor position and actual site arity.

Initial `MPS` surface:

- `length`, `iterate`, `getindex`, `setindex!`;
- `siteinds(::MPS)::Vector{Index}`;
- `linkinds`, `linkdims`, `rank`, `eltype`;
- `inner`, `dot`, `norm`, `add`, `+`, scalar `*`, scalar `/`, `dag`;
- `orthogonalize!`, `truncate!`;
- `replace_siteinds!`, `replace_siteinds`;
- `fixinds`, `suminds`, `projectinds`;
- `to_dense`, `evaluate`, and `scalar`.

`ITensorCompat` truncating APIs should be cutoff-only. They should reject
Tensor4all-native `threshold` and `svd_policy` kwargs with `ArgumentError`.
The mapping is:

```julia
cutoff -> threshold
SvdTruncationPolicy(measure=:squared_value, rule=:discarded_tail_sum)
```

This matches the documented ITensors-style discarded-weight policy.

Scalar / empty-site MPS behavior:

- `length(mps) == 0`;
- `siteinds(mps) == Index[]`;
- `scalar(mps)` returns the rank-0 scalar value;
- `to_dense(mps)` returns a rank-0 `Tensor`;
- `evaluate(mps)` returns the scalar;
- display should not invent sentinel site indices.

Raw constructors:

- `MPS(blocks::AbstractVector{<:Array{T,3}}, sites::Vector{Index})`;
- `MPS(blocks::AbstractVector{<:Array{T,3}})` with generated site indices;
- `MPO(blocks::AbstractVector{<:Array{T,4}}, input_sites, output_sites)`.

The MPO raw constructor must document the physical leg orientation. If the
orientation is still not stable enough for public use, keep `MPO` raw
construction as a documented limitation while implementing the MPS constructor.

### Phase 5: Documentation and BubbleTeaCI Smoke Coverage

Add docstrings for every exported type and function. If new source files are
introduced under `src/`, update `docs/src/api.md` `@autodocs` `Pages = [...]`
lists in the same PR.

Add a compact integration-style test that mirrors the BubbleTeaCI migration
path:

1. construct an `ITensorCompat.MPS` from raw TCI-like blocks;
2. replace site indices without touching `.data`;
3. insert a dummy identity site;
4. fix and sum selected indices;
5. add two MPS values;
6. truncate and orthogonalize with `cutoff`;
7. evaluate and materialize to dense;
8. verify scalar / zero-site behavior.

## Alternatives Considered

### Split Into Multiple PRs

This would reduce review size, but the downstream BubbleTeaCI migration needs
these APIs as one coherent compatibility surface. The chosen compromise is one
PR with clear phase commits and tests, so maintainers can review incrementally
without forcing BubbleTeaCI to target several transient Tensor4all.jl states.

### Put Everything Under `ITensorCompat`

Rejected. Some operations are generic Tensor4all primitives, not compatibility
shims: canonical invalidation, metadata-only index replacement, structured
diagonal constructors, and lazy index slicing belong in Core or
`TensorNetworks`. `ITensorCompat` should adapt names and keyword semantics, not
own generic tensor-network behavior.

### Add More Rust Kernels

Rejected for the Julia-facing helpers in this issue. The Rust C API should
expose general multi-language kernels and storage primitives. Chain mutation,
index metadata rewriting, and high-level MPS convenience operations are
Julia-owned and can be composed from tensors, indices, and contraction.

### Full Tensor Storage Refactor First

A full replacement of `Tensor.data::Array` with a completely storage-polymorphic
Julia object would be cleaner long term, but it is larger than this issue
requires. This PR should expose enough structured storage metadata and
constructors to preserve diagonal identity/delta payloads, then keep dense
materialization available through existing APIs such as `Array(t, inds...)`.

## Acceptance Criteria

- [ ] `setindex!(tt, tensor, i)` local canonical invalidation is documented and
      explicitly tested.
- [ ] Public TensorTrain mutation helpers exist for block replacement,
      insertion, deletion, push, and pushfirst; topology-changing operations
      fully invalidate canonical bounds.
- [ ] Downstream code can perform common MPS-like mutations without touching
      `tt.data`, `tt.llim`, or `tt.rlim` directly.
- [ ] Metadata-only index replacement preserves tensor values, backend handles,
      structured storage metadata, and canonical bounds.
- [ ] `tagstring` supports public tag formatting without field access.
- [ ] Structured diagonal / identity constructors use compact payload storage
      when the Rust backend is available.
- [ ] `fixinds`, `suminds`, and `projectinds` work for `Tensor` and
      `TensorTrain` without constructing dense one-hot vectors for slicing.
- [ ] `insert_identity!` or equivalent dummy-site insertion uses structured
      identity tensors where available.
- [ ] `Tensor4all.ITensorCompat` exposes `MPS` and a narrow `MPO` wrapper.
- [ ] `ITensorCompat` truncating APIs accept `cutoff` and reject
      `threshold` / `svd_policy`.
- [ ] Scalar and empty-site MPS behavior is documented and tested.
- [ ] Raw MPS block construction is documented and tested; raw MPO construction
      is either supported or clearly documented as a current limitation.
- [ ] `docs/src/api.md` autodocs coverage is updated for every new public
      source file.
- [ ] `Pkg.test()` and `julia --project=docs docs/make.jl` pass for the PR.

## Test Strategy

Use focused tests per phase:

- `test/tensornetworks/llim_rlim.jl` for mutation and canonical bounds.
- `test/core/index.jl` and `test/core/tensor.jl` for tag formatting,
  structured storage metadata, diagonal constructors, and metadata-only
  replacement.
- new Tensor and TensorTrain slicing tests for `fixinds`, `suminds`, and
  `projectinds`.
- new `test/itensorcompat/` coverage for wrapper validation, cutoff mapping,
  scalar MPS behavior, raw constructors, and MPS workflows.
- one BubbleTeaCI-shaped integration test that uses only public APIs.

Run the full package test suite and docs build before opening the PR. On the
AMD EPYC Julia x64 host described in `AGENTS.md`, use the documented Docker
workaround for final verification.

## Cross-Repository Dependency

This issue should not require new Rust C API functions for high-level Julia
helpers. It does require Tensor4all.jl to consume the existing structured tensor
storage C API from tensor4all-rs:

- `t4a_tensor_new_diag_f64`
- `t4a_tensor_new_diag_c64`
- `t4a_tensor_new_structured_f64`
- `t4a_tensor_new_structured_c64`
- `t4a_tensor_storage_kind`
- payload rank, length, dims, strides, axis classes, and payload copy helpers

If the target tensor4all-rs commit containing those APIs is not already merged
on the remote branch used by `deps/TENSOR4ALL_RS_PIN`, merge that Rust PR first,
then bump the pin in the Tensor4all.jl PR.
