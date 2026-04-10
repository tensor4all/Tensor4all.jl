# Architecture Status

## Current Phase

`Tensor4all.jl` is in a reset-only phase. The package now exists mainly as a
review surface for the imported design documents and for the next staged
implementation plan.

## Intentional Removals

- The previous `src/` code has been removed.
- The previous integration-heavy test suite has been removed.
- The old API reference and tutorial docs have been removed.

This was done to avoid a mixed state where outdated behavior appears to be
supported while the architecture is being reworked.

## Planned Layering

| Layer | Planned responsibility | Status |
|------|-------------------------|--------|
| Core | low-level Julia wrappers and ownership model | deferred |
| TreeTN | TreeTN-general tensor network layer, including chain aliases | deferred |
| Quantics | adopted `QuanticsGrids.jl` grid layer plus `Tensor4all.jl`-owned transforms and integration | deferred |
| Extensions | ITensors/HDF5 compatibility glue | deferred |
| BubbleTeaCI | `TTFunction` and high-level function workflows | out of scope for this repo |

## Ownership Boundary

- `tensor4all-rs` owns kernels, storage, and numerically heavy backend behavior.
- `Tensor4all.jl` will own Julia-side wrappers and TreeTN-general abstractions.
- `QuanticsGrids.jl` owns quantics grid semantics and coordinate conversion; `Tensor4all.jl` is expected to adopt and re-export that layer rather than reimplement it.
- `BubbleTeaCI` remains the home of `TTFunction` / `GriddedFunction` logic.

## Next Review Questions

- Is the TreeTN-first public model the right base for the Julia package?
- Is the adopted-dependency and re-export policy for the quantics layer clean enough?
- Is the downstream `BubbleTeaCI` contract explicit enough before implementation starts?
- Is the staged implementation order acceptable before backend-facing APIs reappear?
