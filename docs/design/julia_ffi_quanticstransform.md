# Julia Frontend QuanticsTransform Layer

## Purpose

This document covers the Julia-side quantics-transform constructor layer that
sits on top of `TensorNetworks`.

## In Scope

- quantics-specific operator constructors
- construction-time metadata and validation
- the reduced chain-oriented C API assumption for transform kernels

This document does not own the generic operator type or `apply`.

## Ownership

- `TensorNetworks` owns `LinearOperator`.
- `TensorNetworks` owns `apply`, `set_input_space!`, `set_output_space!`, and
  `set_iospaces!`.
- `QuanticsTransform` only constructs `LinearOperator` values for quantics
  semantics.

## Kernel Boundary

The docs for this phase assume a minimized, chain-oriented backend ABI:

- materialization kernels for TT-like operators
- apply kernels for chain subsets
- no tree-specific public ABI promises in this branch

That keeps `QuanticsTransform` aligned with the reduced Julia frontend story.

## Relationship to the TT Layers

- `TensorNetworks` provides the public chain container.
- `TensorNetworks` also provides `LinearOperator` and `apply`.
- `SimpleTT` provides raw-array TT numerics.
- `TensorCI` provides interpolation output as `TensorCI2`.
- `QuanticsTransform` sits on top of those layers as a constructor boundary.

## Current Phase 1 Shape

The current skeleton constructors return `TensorNetworks.LinearOperator` values
whose `metadata` field records the requested transform.

Available constructor names are:

- `shift_operator`
- `shift_operator_multivar`
- `flip_operator`
- `flip_operator_multivar`
- `phase_rotation_operator`
- `phase_rotation_operator_multivar`
- `cumsum_operator`
- `fourier_operator`
- `affine_operator`
- `affine_pullback_operator`
- `binaryop_operator`

## Open Questions

- Which operator semantics should remain Julia-owned versus delegated to Rust
  kernels?
- How much layout information should be explicit in Julia, given the reduced
  C API assumption?
