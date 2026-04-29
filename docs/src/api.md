# API Reference

## Core

`Core` exposes the low-level wrappers that everything else builds on:

- `Tensor4all.Index`
- `Tensor4all.Tensor`, `Tensor4all.ITensor`
- `Tensor4all.dim`, `Tensor4all.id`, `Tensor4all.tags`, `Tensor4all.plev`
- `Tensor4all.hastag`, `Tensor4all.sim`, `Tensor4all.prime`,
  `Tensor4all.noprime`, `Tensor4all.setprime`
- `Tensor4all.replaceind`, `Tensor4all.replaceinds`
- `Tensor4all.replaceind!`, `Tensor4all.replaceinds!`
- `Tensor4all.commoninds`, `Tensor4all.uniqueinds`, `Tensor4all.hasinds`
- `Tensor4all.inds`, `Tensor4all.rank`, `Tensor4all.dims`,
  `Tensor4all.swapinds`
- `Tensor4all.scalar`, `Tensor4all.onehot`
- `Tensor4all.copy_data` — explicitly materialize a fresh dense copy of a
  `Tensor`, optionally in a requested `Index` order
- `Tensor4all.fixinds`, `Tensor4all.suminds`, `Tensor4all.projectinds`
- `Tensor4all.delta`, `Tensor4all.isdiag`,
  `Tensor4all.structured_storage_info`, `Tensor4all.structured_payload`
- `Tensor4all.dag` — pure Julia tensor conjugation
- `Array(t, inds...)` — dense tensor extraction in the requested index order
- `Tensor4all.contract` — backend tensor contraction via the
  `t4a_tensor_contract` C API
- `Tensor4all.svd`, `Tensor4all.qr` — backend tensor factorizations via the
  C API

`Tensor` owns a backend tensor handle plus Julia-side `Index` metadata. It does
not expose `.data` or cached `structured_storage` fields; dense materialization
is always an explicit copy through `copy_data` or `Array`.

```@docs
Tensor4all
```

```@autodocs
Modules = [Tensor4all]
Pages = ["Tensor4all.jl", "Core/Errors.jl", "Core/Backend.jl", "Core/Index.jl", "Core/Tensor.jl", "Core/IndexOps.jl"]
Private = false
Order = [:type, :function]
```

## TensorNetworks

The public chain wrapper is `Tensor4all.TensorNetworks.TensorTrain`.

```julia
mutable struct TensorTrain
    data::Vector{Tensor}
    llim::Int
    rlim::Int
end
```

Other chain-facing names in this layer include:

- `Tensor4all.TensorNetworks.LinearOperator`
- `Tensor4all.TensorNetworks.SvdTruncationPolicy` — SVD truncation strategy
  (`scale`, `measure`, `rule`), paired with a per-call `threshold` kwarg
  on every truncating function. See the Truncation Policy chapter.
- `Tensor4all.TensorNetworks.invalidate_canonical!`,
  `Tensor4all.TensorNetworks.replaceblock!`,
  `Tensor4all.TensorNetworks.insert_site!`, and
  `Tensor4all.TensorNetworks.delete_site!` — public mutation helpers that keep
  canonical bounds consistent
- `Tensor4all.TensorNetworks.default_svd_policy`,
  `Tensor4all.TensorNetworks.set_default_svd_policy!`,
  `Tensor4all.TensorNetworks.with_svd_policy` — process-wide and task-local
  default-policy registry.
- `Tensor4all.TensorNetworks.set_input_space!`
- `Tensor4all.TensorNetworks.set_output_space!`
- `Tensor4all.TensorNetworks.set_iospaces!`
- `Tensor4all.TensorNetworks.insert_operator_identity!`
- `Tensor4all.TensorNetworks.delete_operator_site!`
- `Tensor4all.TensorNetworks.delete_operator_sites!`
- `Tensor4all.TensorNetworks.permute_operator_sites!`
- `Tensor4all.TensorNetworks.replace_operator_input_indices!`
- `Tensor4all.TensorNetworks.replace_operator_output_indices!`
- `Tensor4all.TensorNetworks.apply`
- `Tensor4all.TensorNetworks.findsite`
- `Tensor4all.TensorNetworks.findsites`
- `Tensor4all.TensorNetworks.findallsiteinds_by_tag`
- `Tensor4all.TensorNetworks.findallsites_by_tag`
- `Tensor4all.TensorNetworks.replace_siteinds!`
- `Tensor4all.TensorNetworks.replace_siteinds`
- `Tensor4all.TensorNetworks.replace_siteinds_part!`
- `Tensor4all.TensorNetworks.fixinds`
- `Tensor4all.TensorNetworks.suminds`
- `Tensor4all.TensorNetworks.projectinds`
- `Tensor4all.TensorNetworks.identity_link_tensor`
- `Tensor4all.TensorNetworks.insert_identity!`
- `Tensor4all.TensorNetworks.rearrange_siteinds`
- `Tensor4all.TensorNetworks.makesitediagonal`
- `Tensor4all.TensorNetworks.extractdiagonal`
- `Tensor4all.TensorNetworks.matchsiteinds`
- `Tensor4all.TensorNetworks.save_as_mps`
- `Tensor4all.TensorNetworks.load_tt`
- `Tensor4all.TensorNetworks.dag`
- `Tensor4all.TensorNetworks.linkinds`
- `Tensor4all.TensorNetworks.linkdims`
- `Tensor4all.TensorNetworks.siteinds`
- `Tensor4all.TensorNetworks.orthogonalize`
- `Tensor4all.TensorNetworks.truncate`
- `Tensor4all.TensorNetworks.add`
- `Tensor4all.TensorNetworks.dot`, `Tensor4all.TensorNetworks.inner`
- `Tensor4all.TensorNetworks.dist`
- `Tensor4all.TensorNetworks.fuse_to`
- `Tensor4all.TensorNetworks.split_to`
- `Tensor4all.TensorNetworks.swap_site_indices`
- `Tensor4all.TensorNetworks.restructure_to`
- `Tensor4all.TensorNetworks.contract` — `TensorTrain × TensorTrain` chain
  contraction via the `t4a_treetn_contract` C API
- `Tensor4all.TensorNetworks.PartialContractionSpec`,
  `Tensor4all.TensorNetworks.partial_contract`, and
  `Tensor4all.TensorNetworks.elementwise_product` — explicit partial
  contraction and elementwise chain products through the TreeTN C API
- `Tensor4all.TensorNetworks.to_dense` — materialize a `TensorTrain` as a
  single dense `Tensor` over its site indices
- `Tensor4all.TensorNetworks.evaluate` — pointwise evaluation of a
  `TensorTrain` at one or more index value tuples
- `Tensor4all.TensorNetworks.TensorTrainEvaluator` and
  `Tensor4all.TensorNetworks.TensorTrainEvalWorkspace` — pure Julia dense
  chain snapshots for repeated scalar point evaluation without per-call
  Tensor/Index construction in the grouped hot path
- `Tensor4all.TensorNetworks.random_tt` — pure-Julia random `TensorTrain`
  built from Haar-distributed partial isometries (mirrors
  `ITensorMPS.random_mps`)

`TensorNetworks.TensorTrain` is the container that HDF5 compatibility works
against.

The current Julia implementation includes the full helper surface above.
`set_input_space!`, `set_output_space!`, and `set_iospaces!` accept explicit
`Vector{Index}` arguments only. `apply` is implemented for the current
chain-oriented backend path. `TensorTrain` also supports scalar arithmetic and
comparison on the current backend path through `+`, `-`, scalar `*`, scalar
`/`, `norm`, `dot`/`inner`, `isapprox`, `dist`, and `add`. `restructure_to`
is implemented in pure Julia and dispatches to `fuse_to` / `split_to` /
`swap_site_indices` based on the diff between the current and target site
groupings, including mixed split/swap/fuse regrouping. `rearrange_siteinds`
is a thin wrapper over `restructure_to`.

`LinearOperator` topology helpers mutate `op.mpo`, `op.input_indices`,
`op.output_indices`, `op.true_input`, and `op.true_output` together. Use them
when inserting identity operator sites, deleting/pruning operator sites,
permuting sites, or renaming internal input/output site indices; callers
should not update those metadata arrays manually.

For repeated point evaluation of one fixed chain, prefer
`TensorTrainEvaluator(tt)` plus `TensorTrainEvalWorkspace(ev)` and call
`evaluate!(ws, ev, grouped_values)`. A small benchmark should warm up once,
then measure the grouped hot path directly, for example:

```julia
ev = Tensor4all.TensorNetworks.TensorTrainEvaluator(tt)
ws = Tensor4all.TensorNetworks.TensorTrainEvalWorkspace(ev)
grouped_values = [[1], [2], [1]]
Tensor4all.TensorNetworks.evaluate!(ws, ev, grouped_values)
@btime Tensor4all.TensorNetworks.evaluate!($ws, $ev, $grouped_values)
```

```@autodocs
Modules = [Tensor4all.TensorNetworks]
Pages = ["TensorNetworks/types.jl", "TensorNetworks/operator_spaces.jl", "TensorNetworks/site_helpers.jl", "TensorNetworks/matchsiteinds.jl", "TensorNetworks/operator_mutations.jl", "TensorNetworks/transforms.jl", "TensorNetworks/index_ops.jl", "TensorNetworks/identity_helpers.jl", "TensorNetworks/truncation_policy.jl", "TensorNetworks/backend/apply.jl", "TensorNetworks/backend/treetn.jl", "TensorNetworks/backend/treetn_queries.jl", "TensorNetworks/backend/treetn_dense.jl", "TensorNetworks/backend/treetn_contract.jl", "TensorNetworks/backend/partial_contract.jl", "TensorNetworks/backend/treetn_evaluate.jl", "TensorNetworks/evaluator.jl", "TensorNetworks/backend/restructure/fuse_to.jl", "TensorNetworks/backend/restructure/split_to.jl", "TensorNetworks/backend/restructure/swap_site_indices.jl", "TensorNetworks/backend/restructure/restructure_to.jl", "TensorNetworks/backend/linsolve.jl", "TensorNetworks/random.jl", "TensorNetworks/bridge.jl", "TensorNetworks/deferred.jl"]
Private = false
Order = [:type, :function]
```

```@docs
Base.transpose(::Tensor4all.TensorNetworks.LinearOperator)
```

## ITensorCompat

`Tensor4all.ITensorCompat` is an opt-in migration facade for ITensors-style
code. It wraps `TensorNetworks.TensorTrain` values without replacing the
native TensorNetworks modeling layer.

The compatibility truncation surface is cutoff-only. Use
`ITensorCompat.truncate!(m; cutoff, maxdim)` for ITensors-style behavior, and
use `TensorNetworks.truncate` directly when you need native `threshold` or
`svd_policy` controls.

`fixinds`, `suminds`, `projectinds`, `delta`, and
`TensorNetworks.insert_identity!` are generic Tensor4all APIs. They are not
BubbleTeaCI-specific helpers.

Raw MPS blocks are read in `(left_link, site, right_link)` order. Raw MPO
blocks are read in `(left_link, input_site, output_site, right_link)` order.

```@autodocs
Modules = [Tensor4all.ITensorCompat]
Pages = ["ITensorCompat.jl"]
Private = false
Order = [:type, :function]
```

## SimpleTT

The raw-array TT layer is `Tensor4all.SimpleTT.TensorTrain{T,N}`.

```julia
mutable struct TensorTrain{T,N}
    sitetensors::Vector{Array{T,N}}
end
```

Its current public operations are:

- `Tensor4all.SimpleTT.compress!`
- `Tensor4all.SimpleTT.contract`

The important conventions are:

- `N=3` for MPS-like site tensors
- `N=4` for MPO-like site tensors
- `compress!` supports `:LU`, `:CI`, and `:SVD`
- `contract` supports `algorithm = :naive` and `algorithm = :zipup`

```@autodocs
Modules = [Tensor4all.SimpleTT]
Pages = ["SimpleTT.jl"]
Private = false
Order = [:type, :function]
```

```@docs
Tensor4all.SimpleTT.contract
```

## TensorCI

`Tensor4all.TensorCI.crossinterpolate2` is the interpolation boundary.

It returns `TensorCI2` for the supported multi-site path. Conversion into the
raw numerical TT layer happens through `Tensor4all.SimpleTT.TensorTrain(tci)`.

```@autodocs
Modules = [Tensor4all.TensorCI]
Pages = ["TensorCI.jl"]
Private = false
Order = [:type, :function]
```

## QuanticsTransform

`Tensor4all.QuanticsTransform` provides transform constructors such as:

- `shift_operator`
- `flip_operator`
- `phase_rotation_operator`
- `cumsum_operator`
- `fourier_operator`
- `affine_operator`
- `binaryop_operator`
- `unfuse_quantics_operator`

These constructors return `TensorNetworks.LinearOperator` values. The generic
operator type itself does not live in `QuanticsTransform`.

In the current branch, `shift_operator`, `shift_operator_multivar`,
`flip_operator`, `flip_operator_multivar`, `phase_rotation_operator`,
`phase_rotation_operator_multivar`, `cumsum_operator`, `fourier_operator`,
`affine_operator`, `affine_pullback_operator`,
`affine_pullback_operator_multivar`, `binaryop_operator`, and
`binaryop_operator_multivar` materialize real MPO-backed operators through the
C API.
`TensorNetworks.apply` owns execution of those materialized operators once the
I/O spaces are bound.

### QuanticsTransform Layout Contract

Univariate constructors such as `shift_operator`, `flip_operator`, and
`affine_operator` produce one MPO site per bit layer. Multivariable
constructors such as `shift_operator_multivar`, `flip_operator_multivar`,
`affine_pullback_operator_multivar`, and `binaryop_operator_multivar` use a
fused-QTT layout: there are `r` operator sites, and each input/output physical
index has dimension `2^D` for `D` variables.

Bit layers are ordered from `1:r` in the returned `op.input_indices` and
`op.output_indices`. Inside each fused physical index, variables are encoded in
Julia column-major order: with zero-based digits, `fused = v1 + 2*v2 + ...`.
The first variable is therefore the least-significant digit. The helper
`unfuse_quantics_operator(op, input_sites, output_sites; base=2)` expands that
layout into a flat site-major vector `[bit1_var1, bit1_var2, bit2_var1, ...]`
using caller-supplied per-variable input/output indices.

Dense operator tensors should be interpreted with output sites first and input
sites second: `Array(TensorNetworks.to_dense(op.mpo), op.output_indices...,
op.input_indices...)`. `transpose(op::TensorNetworks.LinearOperator)` swaps
input/output metadata and tensor axes, so affine forward operators and
pullback-style usage follow the usual matrix-transpose relation. Boundary
condition keywords (`:periodic` or `:open`) are constructor-specific and apply
to the transformed source variables documented by each constructor.

```@autodocs
Modules = [Tensor4all.QuanticsTransform]
Pages = ["QuanticsTransform/QuanticsTransform.jl", "QuanticsTransform/operators.jl", "QuanticsTransform/unfuse.jl"]
Private = false
Order = [:function]
```

## Adopted Modules

- `Tensor4all.QuanticsGrids` re-exports the public `QuanticsGrids.jl` surface
- `Tensor4all.QuanticsTCI` re-exports the public `QuanticsTCI.jl` surface

These modules are wrapper re-exports for discoverability. Tensor4all does not
take ownership of their APIs; see the upstream
[QuanticsGrids.jl](https://tensor4all.org/QuanticsGrids.jl/dev/apireference/)
and
[QuanticsTCI.jl](https://tensor4all.org/QuanticsTCI.jl/dev/apireference/)
docs for the complete surface.

```julia
const QG = Tensor4all.QuanticsGrids

grid = QG.DiscretizedGrid(3, 0.0, 1.0)
grid_index = QG.origcoord_to_grididx(grid, 0.5)
quantics_index = QG.grididx_to_quantics(grid, grid_index)
QG.quantics_to_origcoord(grid, quantics_index)  # 0.5
```

```julia
const QTCI = Tensor4all.QuanticsTCI

qtt, ranks, errors = QTCI.quanticscrossinterpolate(
    Float64,
    x -> sin(x),
    range(0, 1; length=8);
    tolerance=1e-8,
)

qtt(4)      # evaluate at an integer grid index
sum(qtt)    # upstream QuanticsTCI reduction
```

## HDF5 Compatibility

The HDF5 extension provides the persistence boundary for the restored chain
type:

- `save_as_mps` writes a `TensorNetworks.TensorTrain` using the `MPS` schema
- `load_tt` reads that schema back into `TensorNetworks.TensorTrain`
- the public docs assume a reduced, chain-oriented C API target on the Rust
  side
