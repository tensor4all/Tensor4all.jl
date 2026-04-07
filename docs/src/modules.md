# [Module Architecture](@id Module-Architecture)

## Dependency Graph

```
                    ┌──────────────────────┐
                    │   Core Foundation     │
                    │  Index, Tensor,       │
                    │  Algorithm            │
                    └──────────┬───────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
         ┌──────────┐   ┌──────────┐   ┌──────────────┐
         │ SimpleTT │◄─►│  TreeTN  │   │QuanticsGrids │
         │          │   │ MPS/MPO  │   │              │
         └──────────┘   └────┬─────┘   └──────┬───────┘
                             │                 │
                    ┌────────┼────────┐        │
                    ▼        ▼        ▼        ▼
              ┌──────────┐ ┌──────┐ ┌────────────┐
              │Quantics  │ │Tree  │ │QuanticsTCI │
              │Transform │ │TCI   │ │            │
              └──────────┘ └──────┘ └────────────┘
```

## Data Flow

Tensor4all.jl modules form a pipeline where data flows from grid definition through interpolation to tensor network operations:

| Stage | Module | Input | Output |
|-------|--------|-------|--------|
| Grid definition | **QuanticsGrids** | Domain bounds, R (bits) | `DiscretizedGrid`, `InherentDiscreteGrid` |
| Function interpolation | **QuanticsTCI** | Function + Grid | `QuanticsTensorCI2` |
| Simple tensor train | **SimpleTT** | TCI result via `to_tensor_train()` | `SimpleTensorTrain` |
| Tensor network | **TreeTN** | SimpleTT via `MPS()` | `MPS` / `TreeTensorNetwork` |
| Operators | **QuanticsTransform** | MPS + operator | Transformed `MPS` |
| Tree interpolation | **TreeTCI** | Function + tree graph | `TreeTensorNetwork` |

## Typical Pipelines

### Quantics function interpolation
```
QuanticsGrids.DiscretizedGrid → QuanticsTCI.quanticscrossinterpolate
  → QuanticsTCI.to_tensor_train → SimpleTensorTrain
  → TreeTN.MPS → MPS operations (truncate!, orthogonalize!, inner, contract)
```

### Fourier analysis of interpolated function
```
QuanticsTCI result → MPS
  → QuanticsTransform.fourier_operator → apply → Fourier-space MPS
  → TreeTCI.evaluate (at specific k-points)
```

### Affine coordinate transform
```
MPS → QuanticsTransform.affine_pullback_operator → apply → transformed MPS
```

### Tree-structured interpolation
```
TreeTCI.crossinterpolate2 → TreeTensorNetwork
  → TreeTN operations (contract, truncate!, evaluate)
```

## Type Relationships

```julia
# Chain tensor networks
TensorTrain = TreeTensorNetwork{Int}  # Primary chain type
MPS = TensorTrain                      # Alias (no type distinction)
MPO = TensorTrain                      # Alias (no type distinction)

# MPS-like vs MPO-like is a runtime property:
is_mps_like(tt)  # each vertex has 1 site index
is_mpo_like(tt)  # each vertex has 2 site indices (unprimed + primed)
```
