# Issue #35 Skeleton API Alignment Status

> Status: completed on the current branch. Do not execute this file as an active work queue.

This plan was re-checked against the current repository on 2026-04-15.

## What Is Already Landed

- `Tensor4all` exposes the approved public submodules: `TensorNetworks`, `SimpleTT`, `TensorCI`, `QuanticsGrids`, `QuanticsTCI`, and `QuanticsTransform`.
- `Tensor4all` does not expose `TreeTensorNetwork`, `MPS`, `MPO`, `affine_transform`, or `shift_transform`.
- `TensorNetworks` owns `TensorTrain`, `LinearOperator`, and `apply`.
- `QuanticsTransform` constructs `TensorNetworks.LinearOperator` values and does not own the generic operator type.
- `TensorCI.crossinterpolate2` returns `TensorCI2`.
- The HDF5 boundary lives on `TensorNetworks.save_as_mps` / `TensorNetworks.load_tt`.

## Remaining Follow-On Work

The alignment work itself is done. The remaining work now lives in the active implementation plan:

- refresh the stale Julia manifest
- backend-enable `Index` and `Tensor`
- convert selected `TensorNetworks` helpers from stubs to real Julia behavior
- clean up stale README/generated-doc wording that still reflects pre-alignment text

Use [2026-04-10-tensor4all-rework-followup.md](./2026-04-10-tensor4all-rework-followup.md) as the single implementation handoff document.