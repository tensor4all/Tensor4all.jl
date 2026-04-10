# Julia Frontend Extensions

## Purpose

This document defines the compatibility and package-extension boundary for the Julia frontend.

## In Scope

- ITensors compatibility
- HDF5 interoperability
- bidirectional conversion rules
- Julia package extension wiring

This document does not own core tensor primitives, TT backend support, quantics semantics, or `TTFunction` logic.

## ITensors Compatibility

Interoperability with `ITensors.jl` remains a hard requirement for the Julia ecosystem.

### Conversion Rules

- `Index` ↔ `ITensors.Index{Int}`
- `Tensor` ↔ `ITensors.ITensor`

### Data Exchange

- support ITensors-style HDF5 save/load
- preserve round-trips between existing Julia workflows and the new frontend

## Package Extension Boundary

- keep ITensors/HDF5 conversion in a Julia extension layer
- avoid pushing compatibility glue into the low-level core
- let core types stay focused on ownership and backend-facing behavior

## Open Questions

- Should tag data be cached to reduce conversion overhead?
- Should metadata caching be handled by the core types or by the extension layer?
- Are there any extra conversion paths that should be first-class from day one?
