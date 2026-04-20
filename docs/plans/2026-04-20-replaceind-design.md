# Tensor Replaceind Design

## Summary

Issue #60 reports that `Tensor4all.Tensor` is missing a `replaceind` method.
Users currently need to reconstruct a tensor manually from copied dense data
and rewritten index metadata. This is inconsistent with the existing
`replaceind` / `replaceinds` helpers on index collections and with the
`ITensors.jl` API the project wants to emulate.

## Goals

- Add `Tensor`-level index replacement APIs that follow the `ITensors.jl`
  calling style closely enough for downstream code migration.
- Keep the implementation Julia-owned and metadata-only. Replacing indices on a
  `Tensor` must not require new Rust or C API support.
- Preserve the current `Tensor` representation and constructor invariants.

## Non-goals

- No C API changes.
- No dense data permutation or tensor contraction changes.
- No attempt to preserve ITensors storage-view semantics. Tensor4all tensors are
  value-like Julia objects whose constructors currently copy dense data.

## Proposed public API

Add the following exported methods:

- `replaceind(t::Tensor, old::Index, new::Index)`
- `replaceind(t::Tensor, replacement::Pair{Index,Index})`
- `replaceinds(t::Tensor, oldinds, newinds)`
- `replaceinds(t::Tensor, replacements::Pair...)`
- `replaceind!(t::Tensor, old::Index, new::Index)`
- `replaceind!(t::Tensor, replacement::Pair{Index,Index})`
- `replaceinds!(t::Tensor, oldinds, newinds)`
- `replaceinds!(t::Tensor, replacements::Pair...)`

Also extend the existing index-collection helpers so the calling contract
matches the tensor methods:

- `replaceind(xs::AbstractVector{Index}, replacement::Pair{Index,Index})`
- `replaceinds(xs::AbstractVector{Index})`
- `replaceinds(xs::AbstractVector{Index}, ())`
- `replaceinds(xs::AbstractVector{Index}, oldinds, newinds)`

## Semantics

The semantics should intentionally follow `ITensors.jl` where practical:

- Missing indices are ignored. Replacing an index that does not occur is a
  no-op.
- The old and new index collections must have matching lengths.
- Any replacement that targets a present index must preserve dimension:
  `dim(old) == dim(new)`. Otherwise throw `ArgumentError`.
- Pair-based multi-replacement is resolved against the original index set, not
  by sequentially rewriting the output of prior replacements. This avoids
  accidental cascading replacements and matches the ITensors intent.
- Non-mutating tensor methods return a new `Tensor` that keeps the same dense
  data values and the same `backend_handle`.
- Mutating tensor methods rewrite only `t.inds` in place. They must not touch
  `t.data`.

## Architecture

Implement the replacement contract once in `src/Core/Index.jl` as a shared
helper that:

1. normalizes the user-facing replacement forms,
2. validates collection lengths,
3. checks dimension compatibility for indices that are actually present, and
4. returns the rewritten `Vector{Index}`.

`src/Core/Tensor.jl` should delegate all index rewriting to those shared
helpers and only handle tensor-specific object construction versus in-place
metadata mutation.

This keeps all replacement rules centralized in `Core`, avoids code drift
between vector and tensor methods, and fits the project policy that Julia owns
semantic validation.

## Error handling

Use early Julia validation with actionable messages:

- Length mismatch between `oldinds` and `newinds`: `DimensionMismatch`
- Attempted replacement with different dimensions for a present index:
  `ArgumentError` including both indices and dimensions
- Unsupported pair payloads such as `[] => []` for `Tensor`: allow on index
  collections for ITensors-style convenience, but do not add loose `Any`
  tensor signatures unless a concrete downstream need appears

## Testing strategy

Add focused unit tests that mirror the representative `ITensors.jl` cases:

- `replaceind` on `Tensor` is non-mutating
- `replaceind!` mutates only index metadata
- `replaceinds` supports pair notation
- empty replacement lists are no-ops for index collections
- missing indices are ignored
- dimension mismatch throws
- multi-replacement uses original positions rather than sequential cascading

## Documentation impact

Document the new exported tensor methods with concise docstrings in
`src/Core/Tensor.jl`. Since they live in an existing source file already
covered by API docs, no `@autodocs` page-list expansion should be needed.

## Recommended implementation scope

Keep this change narrowly focused on `Core`:

- modify `src/Core/Index.jl`
- modify `src/Core/Tensor.jl`
- update `src/Tensor4all.jl` exports
- extend `test/core/index.jl`
- extend `test/core/tensor.jl`

This closes issue #60 while also aligning the foundational replacement surface
for future ITensors-style API compatibility.
