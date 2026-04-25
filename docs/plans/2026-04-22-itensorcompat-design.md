# ITensorCompat Design

## Summary

BubbleTeaCI currently needs many local helper functions to use
`Tensor4all.TensorNetworks.TensorTrain` as a replacement for `ITensorMPS.MPS`.
Most of those helpers do not express BubbleTeaCI-specific logic. They bridge
small but pervasive API differences: flat versus grouped site indices,
non-mutating versus mutating chain operations, truncation keyword names, and
canonical-region invalidation after tensor replacement.

Add a new `Tensor4all.ITensorCompat` submodule that provides an
ITensors/ITensorMPS source-compatibility facade over the existing Tensor4all
object model. Inside this facade, ITensors naming and call syntax take
priority over Tensor4all-native naming. The facade must not replace
`TensorNetworks.TensorTrain` as the primary indexed chain type.

## Goals

- Reduce downstream compatibility glue needed by BubbleTeaCI and similar
  ITensors-based packages.
- Make `ITensorCompat` cutoff-only for truncating APIs. Tensor4all-native
  `threshold` / `svd_policy` controls remain available through
  `TensorNetworks`, not through the compatibility facade.
- Use payload-only diagonal/copy tensors for `delta`-like helpers once the
  Rust C API exposes structured tensor payload metadata and logical-axis
  equivalence classes.
- Preserve the existing public layer split described in `AGENTS.md`.
- Keep `TensorNetworks.TensorTrain` generic: MPS-like and MPO-like meaning
  remains structural at the TensorNetworks layer.
- Avoid new C API functions for compatibility-only behavior that Julia can own.
- Centralize `llim` / `rlim` invalidation inside Tensor4all.jl so downstream
  packages never need to mutate those fields directly.

## Non-goals

- Do not make Tensor4all.jl an ITensors.jl clone outside the explicit
  `ITensorCompat` facade.
- Do not change `TensorNetworks.siteinds(::TensorTrain)` to return a flat
  vector conditionally.
- Do not promise exact ITensors storage aliasing or view semantics.
- Do not add Rust kernels for index metadata rewriting, identity insertion, or
  other pure-Julia structural edits.
- Do not move `MPS` or `MPO` into `TensorNetworks` in the first iteration.

## Proposed module

Add:

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

`MPS` and `MPO` are semantic wrappers. They state how a generic
`TensorNetworks.TensorTrain` should be interpreted by ITensors-style code. The
wrapped `TensorTrain` remains the storage and backend execution boundary.

The top-level `Tensor4all` module should export the submodule name
`ITensorCompat`, not necessarily every compat symbol. Users can opt in with:

```julia
using Tensor4all.ITensorCompat
```

## Invariants

`MPS(tt::TensorTrain)` validates that every tensor has exactly one site-like
index according to the existing TensorNetworks site-index rules.

`MPO(tt::TensorTrain)` validates that every tensor has exactly two site-like
indices. The first iteration should preserve the current index order reported by
`TensorNetworks.siteinds(tt)`. A later extension may add explicit input/output
space metadata for stronger MPO orientation guarantees.

Validation should fail early with `ArgumentError` and include the tensor
position plus the actual number of site indices found.

## Core TensorTrain mutation API

Downstream packages should not touch `tt.data`, `tt.llim`, or `tt.rlim`
directly. Add or tighten public mutation operations on
`TensorNetworks.TensorTrain`:

```julia
Base.setindex!(tt::TensorTrain, tensor::Tensor, i::Int)
Base.insert!(tt::TensorTrain, i::Int, tensor::Tensor)
Base.deleteat!(tt::TensorTrain, i)
Base.push!(tt::TensorTrain, tensor::Tensor)
Base.pushfirst!(tt::TensorTrain, tensor::Tensor)
invalidate_canonical!(tt::TensorTrain)
```

All topology-changing operations and tensor replacement operations should
conservatively reset the canonical window:

```julia
tt.llim = 0
tt.rlim = length(tt) + 1
```

The current `setindex!` narrows or expands `llim` / `rlim` locally. For source
compatibility work, full invalidation is preferable: it is simpler, correct, and
prevents external packages from knowing Tensor4all's canonical-region
convention. Local invalidation can be reintroduced later as an internal
optimization if tests prove it is safe.

## MPS facade API

The first `ITensorCompat.MPS` surface should cover operations needed by
BubbleTeaCI's Tensor4all backend migration:

```julia
Base.length(m::MPS)
Base.iterate(m::MPS, state...)
Base.getindex(m::MPS, i::Int)
Base.setindex!(m::MPS, tensor::Tensor, i::Int)

siteinds(m::MPS)::Vector{Index}
linkinds(m::MPS)
linkdims(m::MPS)
rank(m::MPS)
Base.eltype(m::MPS)

inner(a::MPS, b::MPS)
dot(a::MPS, b::MPS)
norm(m::MPS)
add(a::MPS, b::MPS; cutoff=0.0, maxdim=0, kwargs...)
Base.:+(a::MPS, b::MPS)
Base.:*(alpha::Number, m::MPS)
Base.:*(m::MPS, alpha::Number)
Base.:/(m::MPS, alpha::Number)
dag(m::MPS)

orthogonalize!(m::MPS, site::Integer)
truncate!(m::MPS; cutoff=0.0, maxdim=0, kwargs...)
replace_siteinds!(m::MPS, oldsites, newsites)
replace_siteinds(m::MPS, oldsites, newsites)
to_dense(m::MPS)
evaluate(m::MPS, indices, values)
```

`siteinds(::MPS)` returns a flat `Vector{Index}`. This is the main reason for
the wrapper type: it avoids making `siteinds(::TensorTrain)` type-unstable or
context-dependent.

Mutating operations return the wrapper where practical, matching Julia's common
`!` convention and ITensorMPS-style call sites:

```julia
orthogonalize!(m, 1) === m
truncate!(m; cutoff=1e-10) === m
```

When an underlying TensorNetworks operation is non-mutating, the compat `!`
method should assign the returned train back into `m.tt`.

## MPO facade API

The first `ITensorCompat.MPO` surface should be narrower and focused on layout
and operator application needs:

```julia
Base.length(W::MPO)
Base.getindex(W::MPO, i::Int)
Base.setindex!(W::MPO, tensor::Tensor, i::Int)

siteinds(W::MPO)::Vector{Vector{Index}}
linkinds(W::MPO)
linkdims(W::MPO)
rank(W::MPO)
dag(W::MPO)
```

Do not overfit MPO compatibility until BubbleTeaCI needs concrete MPO
construction or application paths against `ITensorCompat.MPS`. The existing
`TensorNetworks.LinearOperator` and `TensorNetworks.apply` remain the generic
operator layer.

## Tensor and Index compatibility primitives

Some compatibility belongs in Core rather than in `ITensorCompat`, because the
operations are generally useful and match existing Tensor4all names:

```julia
Index(dim::Integer, tag::AbstractString; kwargs...)
Index(dim::Integer; tags::AbstractString, kwargs...)
const ITensor = Tensor
ITensor(data, inds::Index...)
ITensor(value::Number)
Tensor(data, inds::Index...)
replaceind(t::Tensor, old::Index, new::Index)
replaceind!(t::Tensor, old::Index, new::Index)
replaceinds(t::Tensor, oldinds, newinds)
replaceinds!(t::Tensor, oldinds, newinds)
commoninds(a::Tensor, b::Tensor)
uniqueinds(a::Tensor, b::Tensor)
hasinds(t::Tensor, query_inds...)
scalar(t::Tensor)
Base.eltype(t::Tensor)
Base.:*(a::Tensor, b::Tensor)
onehot(index_value::Pair{Index,<:Integer})
delta(i::Index, j::Index; T=Float64)
```

`*(a::Tensor, b::Tensor)` should delegate to `contract(a, b)`. This is an
ITensors convenience, but it is also a natural tensor algebra shorthand.

`replaceind!` and `replaceinds!` should mutate tensor index metadata only. The
current `Tensor` type is immutable as an object, but its `inds::Vector{Index}`
field is mutable. Implement these methods by rewriting the contents of
`t.inds`; do not touch dense data.

`onehot(i => n)` should be lazy. It represents a rank-1 basis tensor with
index `i` and a one at the 1-based value `n`, but it should not allocate a
dense vector just to participate in a contraction. `contract(t,
onehot(i => n))` and `contract(onehot(i => n), t)` should slice `t` along
index `i` when that index is present, removing `i` from the output indices.
Only explicit materialization, such as `Tensor(onehot(i => n))` or `Array`,
should allocate the dense vector. If the contracted tensor does not contain
`i`, the compatibility layer can fall back to ordinary outer-product
semantics by materializing the one-hot operand.

`delta(i, j)` returns the rank-2 identity tensor over equal dimension indices
`i` and `j`. It may use a dense fallback only until the Rust C API exposes
structured diagonal payloads. The intended long-term implementation is a
payload-only diagonal/copy tensor backed by `tensor4all-rs` structured storage:
compact payload data plus logical-axis equivalence classes. Upstream issue
https://github.com/tensor4all/tensor4all-rs/issues/434 tracks the required C
API for constructing these tensors and reading their structured metadata
without dense materialization.

The Julia side should not model diagonal tensors as merely dense arrays with
zeros. It needs enough Core API to preserve and inspect the structured state
when Rust provides it: logical dimensions, payload dimensions, payload strides,
payload length, scalar dtype, and axis equivalence classes. Public ITensors-like
helpers such as `delta` can remain simple, but their implementation should call
the storage-backed path when available. These helpers remove most downstream
hand-written diagonal and basis tensor construction.

Prefer ITensors syntax over Tensor4all-specific factory names. Do not add
`zeros_tensor`, `ones_tensor`, or `diag_tensor` to the first compatibility
surface. Dense constructors such as `ITensor(ones(T, dim(i)), i)` cover the
all-ones case without new names.

## Tensor factorization compatibility

Tensor4all already provides tensor QR and SVD in `src/Core/Tensor.jl`:

```julia
qr(t::Tensor, left_inds::Vector{Index})
svd(t::Tensor, left_inds::Vector{Index}; threshold=0.0, maxdim=0, svd_policy=nothing)
```

Add compatibility overloads and keyword aliases:

```julia
qr(t::Tensor, left_inds::Index...)
svd(t::Tensor, left_inds::Index...; cutoff=0.0, maxdim=0, kwargs...)
```

The `cutoff` keyword maps internally to Tensor4all's `threshold` with an
ITensors-like SVD policy. Do not accept `threshold` or `svd_policy` in
`ITensorCompat.svd`; callers that need Tensor4all-native truncation controls
should call `Tensor4all.svd` or `TensorNetworks` directly.

## Truncation compatibility

`TensorNetworks` intentionally uses `threshold` plus
`SvdTruncationPolicy`. ITensors migration code commonly passes `cutoff`.
`ITensorCompat` should accept only `cutoff` and translate it explicitly:

```julia
const ITENSORS_CUTOFF_POLICY = TensorNetworks.SvdTruncationPolicy(
    measure = :squared_value,
    rule = :discarded_tail_sum,
)
```

Compat methods should reject ambiguous calls that pass both `cutoff` and
Tensor4all-native truncation controls such as `threshold` or `svd_policy`.
The error should say that `ITensorCompat` is cutoff-only and direct the user to
`TensorNetworks` for native controls.

Default behavior:

- `cutoff = 0.0` means no threshold-based truncation.
- `maxdim = 0` means no hard rank cap.
- `cutoff > 0` maps to
  `threshold = cutoff, svd_policy = ITENSORS_CUTOFF_POLICY`.
- `threshold` and `svd_policy` are not part of the compatibility API.

## Constructors from raw site tensors

BubbleTeaCI often starts from raw site tensors produced by TCI. Provide
compat constructors that mirror the ITensorMPS entry points enough for
migration:

```julia
MPS(sitetensors::AbstractVector{<:Array{T,3}}, sites::Vector{Index}) where T
MPS(sitetensors::AbstractVector{<:Array{T,3}})
MPO(sitetensors::AbstractVector{<:Array{T,4}}, input_sites, output_sites)
```

Grid-aware constructors such as `MPS(tt, grid; addsites_inds=...)` should not
live in Tensor4all.jl unless Tensor4all.jl owns that grid type. BubbleTeaCI can
construct `sites` from its grid and pass them explicitly.

The constructor should create link indices tagged with the existing
TensorNetworks link tag convention (`"Link"` plus `"l=i"`).

## Error handling

Follow the repository's error policy:

- Use `ArgumentError` for invalid structure or ambiguous keyword usage.
- Use `DimensionMismatch` for shape and index-dimension mismatches.
- Include tensor position, expected site arity, actual site arity, and relevant
  index dimensions in messages.
- Validate in Julia before calling the C API.

## Documentation impact

Add public docstrings for exported `ITensorCompat` types and methods.

Add the new source file paths to `docs/src/api.md` `@autodocs` pages when the
implementation adds public files. This is required by `AGENTS.md` because
Documenter can otherwise pass while missing new docstrings.

The API reference should describe `ITensorCompat` as an opt-in migration facade,
not as the canonical Tensor4all modeling layer.

## Testing strategy

Add a focused `test/itensorcompat/` test group:

- `MPS(tt)` accepts one-site-index-per-tensor chains and rejects non-MPS chains.
- `MPO(tt)` accepts two-site-index-per-tensor chains and rejects non-MPO chains.
- `siteinds(::MPS)` returns a flat `Vector{Index}` in chain order.
- `siteinds(::MPO)` returns grouped site indices.
- `setindex!`, `insert!`, `deleteat!`, `push!`, and `pushfirst!` invalidate
  `llim` / `rlim` without downstream access to those fields.
- `orthogonalize!` and `truncate!` mutate the wrapper by replacing `m.tt` and
  return `m`.
- `truncate!(; cutoff=...)` maps to ITensors-like truncation policy.
- passing Tensor4all-native truncation keywords such as `threshold` or
  `svd_policy` through `ITensorCompat` throws `ArgumentError`.
- raw `Array{T,3}` constructors round-trip through `to_dense`.
- Tensor-level `ITensor`, `commoninds`, `uniqueinds`, `replaceind!`, `scalar`,
  `onehot`, `delta`, and `*` match ITensors-style usage.
- contracting a tensor with `onehot(i => n)` slices the tensor along `i`
  without requiring dense one-hot vector materialization.
- once tensor4all-rs#434 is available, `delta` / copy tensor construction uses
  structured payload storage and exposes payload/axis-class metadata without
  dense materialization.

For BubbleTeaCI readiness, add a small integration-style test that performs:

1. build an `MPS` from raw site tensors,
2. replace site indices,
3. evaluate,
4. add two MPS values,
5. truncate and orthogonalize,
6. materialize to dense.

This should cover the operations currently requiring helper functions in the
BubbleTeaCI Tensor4all backend branch.

## Recommended implementation order

1. Core compatibility primitives that are not controversial:
   `Index(dim, tag)`, `ITensor(data, inds...)`, tensor `commoninds` /
   `uniqueinds`, `replaceind!`, `scalar`, `eltype`, `onehot`, `delta`, and
   `*(Tensor, Tensor)`.
2. Rust-backed structured diagonal payload integration once
   https://github.com/tensor4all/tensor4all-rs/issues/434 is available.
3. TensorTrain canonical invalidation API:
   `invalidate_canonical!`, conservative `setindex!`, and topology-changing
   mutation methods.
4. `ITensorCompat` module skeleton and `MPS` wrapper with validation,
   `siteinds`, indexing, dense, evaluate, scalar arithmetic, and site
   replacement.
5. cutoff-only translation and mutating `truncate!` / `orthogonalize!`.
6. Raw-array `MPS` constructors.
7. Narrow `MPO` wrapper.
8. Documentation and BubbleTeaCI migration check.

This order gives BubbleTeaCI the highest-value compatibility surface before
spending time on broader MPO parity.
