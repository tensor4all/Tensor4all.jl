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

- `Tensor4all.vertices`
- `Tensor4all.neighbors`
- `Tensor4all.siteinds`
- `Tensor4all.linkind`
- `Tensor4all.is_chain`
- `Tensor4all.is_mps_like`
- `Tensor4all.is_mpo_like`

`TensorNetworks.TensorTrain` is the container that HDF5 compatibility works
against.

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

## TensorCI

`Tensor4all.TensorCI.crossinterpolate2` is the interpolation boundary.

It returns `SimpleTT.TensorTrain`, not a chain wrapper. That keeps
interpolation output on the raw numerical side of the architecture.

## QuanticsTransform

`Tensor4all.QuanticsTransform.LinearOperator` is the public operator boundary.

This layer is intentionally lightweight. The docs for this phase assume a
minimized chain-oriented backend ABI for materialization and apply kernels.

## HDF5 Compatibility

The HDF5 extension provides the persistence boundary for the restored chain
type:

- `save_as_mps` writes a `TensorNetworks.TensorTrain` using the `MPS` schema
- `load_tt` reads that schema back into `TensorNetworks.TensorTrain`
- the public docs assume a reduced, chain-oriented C API target on the Rust
  side

## Deferred TreeTN Surface

`TreeTensorNetwork` still exists in the repository, but it is secondary in this
branch. The restored public architecture is the `Core` → `TensorNetworks` →
`SimpleTT` → `TensorCI` → `QuanticsTransform` split above.
