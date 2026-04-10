# API Reference

```@docs
Tensor4all
```

## Core

```@docs
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

## TreeTN

```@docs
Tensor4all.TreeTensorNetwork
Tensor4all.TensorTrain
Tensor4all.MPS
Tensor4all.MPO
Tensor4all.vertices
Tensor4all.neighbors
Tensor4all.siteinds
Tensor4all.linkind
Tensor4all.is_chain
Tensor4all.is_mps_like
Tensor4all.is_mpo_like
Tensor4all.orthogonalize!
Tensor4all.truncate!
Tensor4all.inner
Tensor4all.norm
Tensor4all.to_dense
Tensor4all.evaluate
```

## Quantics

### Adopted `QuanticsGrids.jl` Surface

The following names are re-exported through `Tensor4all.jl` for single-import
usability, but their grid semantics and original documentation remain owned by
`QuanticsGrids.jl`:

- `DiscretizedGrid`
- `InherentDiscreteGrid`
- `quantics_to_grididx`
- `quantics_to_origcoord`
- `grididx_to_quantics`
- `grididx_to_origcoord`
- `origcoord_to_quantics`
- `origcoord_to_grididx`

```@docs
Tensor4all.QuanticsTransform
Tensor4all.affine_transform
Tensor4all.shift_transform
Tensor4all.flip_transform
Tensor4all.phase_rotation_transform
Tensor4all.cumsum_transform
Tensor4all.fourier_transform
Tensor4all.binaryop_transform
Tensor4all.materialize_transform
Tensor4all.QTCIOptions
Tensor4all.QTCIDiagnostics
Tensor4all.QTCIResultPlaceholder
```
