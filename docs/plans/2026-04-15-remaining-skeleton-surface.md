# Remaining Skeleton Surface Status

> Status: completed as a skeleton-surface plan. Do not execute this file as written.

This plan was re-checked against the current repository on 2026-04-15.

## What Is Already Landed

- `TensorNetworks` exports the remaining issue-listed helper names.
- The missing helper APIs exist and currently throw explicit placeholder errors.
- `TensorNetworks.save_as_mps` and `TensorNetworks.load_tt` are the declared persistence entry points.
- `Tensor4allHDF5Ext` attaches real HDF5 round-trip methods to those entry points.
- The current test suite already includes `test/tensornetworks/skeleton_surface.jl` and `test/extensions/hdf5_roundtrip.jl`.

## What Moves Forward From Here

The remaining work is no longer `add the names`. It is:

- keep the current surface stable while `Index` and `Tensor` move to backend handles
- convert only the unambiguous helper APIs from stubs to real Julia behavior
- keep the ambiguous chain-helper APIs deferred until the package has an explicit site-index convention
- preserve the HDF5 round-trip while adapting it away from direct `Tensor` field access

Use [2026-04-10-tensor4all-rework-followup.md](./2026-04-10-tensor4all-rework-followup.md) as the single implementation handoff document.