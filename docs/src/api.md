# API Reference

## Core

`Core` exposes the low-level wrappers that everything else builds on:

- `Tensor4all.Index`
- `Tensor4all.Tensor`
- `Tensor4all.dim`, `Tensor4all.id`, `Tensor4all.tags`, `Tensor4all.plev`
- `Tensor4all.hastag`, `Tensor4all.sim`, `Tensor4all.prime`,
  `Tensor4all.noprime`, `Tensor4all.setprime`
- `Tensor4all.replaceind`, `Tensor4all.replaceinds`
- `Tensor4all.commoninds`, `Tensor4all.uniqueinds`
- `Tensor4all.inds`, `Tensor4all.rank`, `Tensor4all.dims`,
  `Tensor4all.swapinds`
- `Tensor4all.dag` — pure Julia tensor conjugation
- `Array(t, inds...)` — dense tensor extraction in the requested index order
- `Tensor4all.contract` — backend tensor contraction via the
  `t4a_tensor_contract` C API
- `Tensor4all.svd`, `Tensor4all.qr` — backend tensor factorizations via the
  C API

```@docs
Tensor4all
```

```@autodocs
Modules = [Tensor4all]
Pages = ["Tensor4all.jl", "Core/Errors.jl", "Core/Backend.jl", "Core/Index.jl", "Core/Tensor.jl"]
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
- `Tensor4all.TensorNetworks.set_input_space!`
- `Tensor4all.TensorNetworks.set_output_space!`
- `Tensor4all.TensorNetworks.set_iospaces!`
- `Tensor4all.TensorNetworks.apply`
- `Tensor4all.TensorNetworks.findsite`
- `Tensor4all.TensorNetworks.findsites`
- `Tensor4all.TensorNetworks.findallsiteinds_by_tag`
- `Tensor4all.TensorNetworks.findallsites_by_tag`
- `Tensor4all.TensorNetworks.replace_siteinds!`
- `Tensor4all.TensorNetworks.replace_siteinds`
- `Tensor4all.TensorNetworks.replace_siteinds_part!`
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

`TensorNetworks.TensorTrain` is the container that HDF5 compatibility works
against.

The current Julia implementation includes the full helper surface above.
`set_input_space!`, `set_output_space!`, and `set_iospaces!` accept explicit
`Vector{Index}` arguments only. `apply` is implemented for the current
chain-oriented backend path. `TensorTrain` also supports scalar arithmetic and
comparison on the current backend path through `+`, `-`, scalar `*`, scalar
`/`, `norm`, `dot`/`inner`, `isapprox`, `dist`, and `add`.

```@autodocs
Modules = [Tensor4all.TensorNetworks]
Pages = ["TensorNetworks/types.jl", "TensorNetworks/operator_spaces.jl", "TensorNetworks/site_helpers.jl", "TensorNetworks/matchsiteinds.jl", "TensorNetworks/transforms.jl", "TensorNetworks/backend/apply.jl", "TensorNetworks/backend/treetn.jl", "TensorNetworks/backend/treetn_queries.jl", "TensorNetworks/backend/treetn_dense.jl", "TensorNetworks/backend/treetn_contract.jl", "TensorNetworks/backend/treetn_evaluate.jl", "TensorNetworks/backend/restructure/fuse_to.jl", "TensorNetworks/backend/restructure/split_to.jl", "TensorNetworks/backend/restructure/swap_site_indices.jl", "TensorNetworks/backend/restructure/restructure_to.jl", "TensorNetworks/random.jl", "TensorNetworks/deferred.jl"]
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

These constructors return `TensorNetworks.LinearOperator` values. The generic
operator type itself does not live in `QuanticsTransform`.

In the current branch, `shift_operator`, `shift_operator_multivar`,
`flip_operator`, `flip_operator_multivar`, `phase_rotation_operator`,
`phase_rotation_operator_multivar`, `cumsum_operator`, `fourier_operator`, and
`affine_operator` materialize real MPO-backed operators through the C API.
`TensorNetworks.apply` owns execution of those materialized operators once the
I/O spaces are bound.

`affine_pullback_operator` and `binaryop_operator` remain deferred
metadata-only placeholders in this phase.

```@autodocs
Modules = [Tensor4all.QuanticsTransform]
Pages = ["QuanticsTransform/QuanticsTransform.jl", "QuanticsTransform/operators.jl"]
Private = false
Order = [:function]
```

## Adopted Modules

- `Tensor4all.QuanticsGrids` re-exports the public `QuanticsGrids.jl` surface
- `Tensor4all.QuanticsTCI` re-exports the public `QuanticsTCI.jl` surface

## HDF5 Compatibility

The HDF5 extension provides the persistence boundary for the restored chain
type:

- `save_as_mps` writes a `TensorNetworks.TensorTrain` using the `MPS` schema
- `load_tt` reads that schema back into `TensorNetworks.TensorTrain`
- the public docs assume a reduced, chain-oriented C API target on the Rust
  side
