# Julia Frontend TT Support

## Purpose

This document covers backend tensor-train support in the Julia frontend. It is the layer between the low-level core primitives and the higher-level `BubbleTeaCI` function semantics.

## In Scope

- `TensorTrain` as a Julia-owned chain of `Tensor`
- TT-level contraction and compression
- TT arithmetic and structural operations
- TT-level transform operators

This document does not define `TTFunction` / `GriddedFunction` semantics. Those remain in [bubbleteaCI.md](./bubbleteaCI.md).

## `TensorTrain`

```julia
mutable struct TensorTrain
    data::Vector{Tensor}
    llim::Int
    rlim::Int
end
```

- Julia owns the chain structure.
- Individual site tensors are Rust-backed `Tensor` objects.
- The same structure can represent MPS- and MPO-style objects, depending on index layout.

## TT-Level Operations

### Core Rust-Backed Operations

- `t4a_contract` for TT-level contraction
- TT compression and reconstruction through arrays of `t4a_tensor` pointers
- whole-chain operations such as orthogonalization or truncation if the Rust backend exposes them

### Rust-Exposed TT Primitives to Surface in Julia

- TT add
- TT scale
- TT dot
- TT reverse
- full tensor export
- construction from site tensors
- any additional TT-level helpers already present in `tensor4all-rs`

## Transform Operators

Quantics transform operators are TT-level linear operators that the Julia frontend can expose as `TensorTrain`-like objects.

### Backend Flow

1. Construct a transform in Rust via `t4a_qtransform_*`
2. Obtain a `t4a_linop`
3. Extract site tensors with `t4a_linop_get_tensors`
4. Wrap the result as a Julia `TensorTrain`

### Existing Transform Constructors

- `t4a_qtransform_affine`
- `t4a_qtransform_shift`
- `t4a_qtransform_flip`
- `t4a_qtransform_phase_rotation`
- `t4a_qtransform_cumsum`
- `t4a_qtransform_fourier`
- `t4a_qtransform_binaryop`

## Relationship to BubbleTeaCI

- This is infrastructure for `BubbleTeaCI`, not a replacement for its high-level function semantics.
- `BubbleTeaCI` should build on this layer rather than duplicate TT backend functionality.

## Open Questions

- Should the public `TensorTrain` API remain thin and backend-shaped, or expose more Julia-side convenience methods?
- How much of the transform operator surface should be directly materialized as `TensorTrain` versus kept as backend handles?
