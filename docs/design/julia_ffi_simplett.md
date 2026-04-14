# Julia Frontend SimpleTT Layer

## Purpose

This document defines the raw-array TT layer used for numerical work in the
restored Julia frontend.

## In Scope

- `SimpleTT.TensorTrain{T,N}`
- `compress!`
- `contract`
- the `:LU` / `:CI` / `:SVD` factorization choices used by `compress!`

This layer owns the raw numerical TT representation. It does not own the public
indexed chain wrapper or the interpolation boundary.

## `TensorTrain{T,N}`

```julia
mutable struct TensorTrain{T,N}
    sitetensors::Vector{Array{T,N}}
end
```

- `TensorTrain{T,3}` is the MPS-like raw-array representation.
- `TensorTrain{T,4}` is the MPO-like raw-array representation.
- `length(tt)` is the number of site tensors.
- All state lives in Julia arrays; the container is intentionally simple.

## Numerical Operations

### Compression

`compress!` performs left/right sweeps over the site tensors and supports:

- `:LU`
- `:CI`
- `:SVD`

The `:LU` and `:CI` branches reuse the cross-interpolation helper package for
the low-rank factorization step.

### MPO-MPO Contraction

`contract` on `TensorTrain{T,4}` is implemented in Julia and supports:

- `algorithm = :naive`
- `algorithm = :zipup`

This keeps the numerical layer independent from the old TreeTN-first story and
lets `TensorCI` and higher layers consume a pure Julia TT kernel.

## Boundary to TensorCI

`TensorCI` should return `SimpleTT.TensorTrain` objects, not `TensorNetworks`
containers. That keeps interpolation output on the raw numerical side of the
boundary.

## Open Questions

- Should additional factorization strategies be exposed here, or should the
  public surface stay limited to the three current options?
- Which parts of the TT numerics should eventually move back to Rust, if any?
