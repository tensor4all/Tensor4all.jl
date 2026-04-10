# Tensor4all Rework Follow-Up Plan

Date: 2026-04-10

## Purpose

This document records the work intentionally deferred after the phase-0 reset.
Phase 0 removes stale implementation state, keeps the package loadable, and
switches the docs site into review-first mode. The items below are not part of
that reset.

## Deferred Work

### Core skeleton

- introduce the reviewed skeleton for `Index` and `Tensor`
- define ownership and lifecycle behavior at the Julia level
- add the first backend-facing smoke tests

### TreeTN-general layer

- introduce `TreeTensorNetwork{V}` as the general type
- add `TensorTrain = TreeTensorNetwork{Int}`
- add `MPS` and `MPO` as runtime aliases / conventions
- add topology predicates and chain-specific runtime checks

### Quantics and transforms

- add quantics grid and layout types
- add transform-constructor stubs and reviewed naming
- add QTCI-facing placeholder types and result containers

### Extensions

- reintroduce ITensors compatibility as an extension-only layer
- reintroduce HDF5 interoperability as extension glue
- add extension smoke tests once the core stubs exist

### Review cadence

- land the rework bottom-up in small commits
- use docs previews as the main human-review artifact after each layer
- avoid restoring numerics until API shape and ownership boundaries are approved

### Testing and docs expansion

- expand the smoke suite once real layer stubs exist
- restore API-reference pages after the reviewed skeletons land
- add later tutorial/onboarding material only after the package surface stabilizes

## Suggested Order

1. Core skeleton
2. TreeTN-general network layer
3. Quantics layer
4. Extensions
5. Expanded tests and API docs

## Non-Goals For Phase 0

- no real backend numerics
- no restored old module structure
- no `TTFunction` logic inside `Tensor4all.jl`
- no resurrection of the previous tutorials or integration-heavy tests
