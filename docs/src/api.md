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
  `Tensor4all.swapinds`, `Tensor4all.contract`

```@docs
Tensor4all
Tensor4all.SkeletonPhaseError
Tensor4all.SkeletonNotImplemented
Tensor4all.BackendUnavailableError
Tensor4all.backend_library_path
Tensor4all.require_backend
Tensor4all.Index
Tensor4all.dim
Tensor4all.id
Tensor4all.tags
Tensor4all.plev
Tensor4all.hastag
Tensor4all.sim
Tensor4all.prime
Tensor4all.noprime
Tensor4all.setprime
Tensor4all.replaceind
Tensor4all.replaceinds
Tensor4all.commoninds
Tensor4all.uniqueinds
Tensor4all.Tensor
Tensor4all.inds
Tensor4all.rank
Tensor4all.dims
Tensor4all.swapinds
Tensor4all.contract
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

`TensorNetworks.TensorTrain` is the container that HDF5 compatibility works
against.

The current Julia implementation includes the full helper surface above.
`set_input_space!`, `set_output_space!`, and `set_iospaces!` accept explicit
`Vector{Index}` arguments only. `apply` remains the main deferred execution
boundary in this layer.

```@docs
Tensor4all.TensorNetworks.TensorTrain
Tensor4all.TensorNetworks.LinearOperator
Tensor4all.TensorNetworks.set_input_space!
Tensor4all.TensorNetworks.set_output_space!
Tensor4all.TensorNetworks.set_iospaces!
Tensor4all.TensorNetworks.apply
Tensor4all.TensorNetworks.findsite
Tensor4all.TensorNetworks.findsites
Tensor4all.TensorNetworks.findallsiteinds_by_tag
Tensor4all.TensorNetworks.findallsites_by_tag
Tensor4all.TensorNetworks.replace_siteinds!
Tensor4all.TensorNetworks.replace_siteinds
Tensor4all.TensorNetworks.replace_siteinds_part!
Tensor4all.TensorNetworks.rearrange_siteinds
Tensor4all.TensorNetworks.makesitediagonal
Tensor4all.TensorNetworks.extractdiagonal
Tensor4all.TensorNetworks.matchsiteinds
Tensor4all.TensorNetworks.save_as_mps
Tensor4all.TensorNetworks.load_tt
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

```@docs
Tensor4all.SimpleTT.TensorTrain
Tensor4all.SimpleTT.contract
```

## TensorCI

`Tensor4all.TensorCI.crossinterpolate2` is the interpolation boundary.

It returns `TensorCI2` for the supported multi-site path. Conversion into the
raw numerical TT layer happens through `Tensor4all.SimpleTT.TensorTrain(tci)`.

```@docs
Tensor4all.TensorCI.TensorCI2
Tensor4all.TensorCI.crossinterpolate2
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

In the current branch these constructors are metadata-only descriptions of the
requested quantics operator. Real operator execution remains owned by
`TensorNetworks.apply`.

```@docs
Tensor4all.QuanticsTransform.shift_operator
Tensor4all.QuanticsTransform.shift_operator_multivar
Tensor4all.QuanticsTransform.flip_operator
Tensor4all.QuanticsTransform.flip_operator_multivar
Tensor4all.QuanticsTransform.phase_rotation_operator
Tensor4all.QuanticsTransform.phase_rotation_operator_multivar
Tensor4all.QuanticsTransform.cumsum_operator
Tensor4all.QuanticsTransform.fourier_operator
Tensor4all.QuanticsTransform.affine_operator
Tensor4all.QuanticsTransform.affine_pullback_operator
Tensor4all.QuanticsTransform.binaryop_operator
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
