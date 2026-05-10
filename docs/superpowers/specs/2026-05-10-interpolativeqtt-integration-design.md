# InterpolativeQTT.jl Integration Design

**Date:** 2026-05-10
**Status:** Approved

## Goal

Integrate [InterpolativeQTT.jl](https://github.com/tensor4all/InterpolativeQTT.jl) into
Tensor4all.jl as a re-export module (`Tensor4all.InterpolativeQTT`), following the same
pattern as `Tensor4all.QuanticsGrids` and `Tensor4all.QuanticsTCI`.

## Background

InterpolativeQTT.jl implements multiscale polynomial (Chebyshev) interpolation for
quantics tensor trains (QTTs), based on Lindsey arXiv:2311.12554. It currently has no
`export` statements, so the standard `names()`-based re-export loop used elsewhere in
Tensor4all.jl would produce nothing without upstream changes.

## Chosen Approach

**Approach A — Curated exports upstream + standard re-export loop.**

Add exactly eight `export` statements to InterpolativeQTT.jl covering the user-facing
QTT interpolation surface. The Tensor4all wrapper then uses a `names()` loop identical
to QuanticsGrids.jl. Authority over the public surface lives in the upstream package.

Rejected alternatives:
- **Approach B** (no upstream exports, explicit list in wrapper): keeps upstream
  unchanged but duplicates the symbol list in Tensor4all and makes the wrapper diverge
  from the established pattern.
- **Approach C** (export everything upstream, filter in wrapper): risks leaking internal
  symbols if the upstream package grows.

## Exported Surface

The following eight symbols are exported from InterpolativeQTT.jl and re-exported from
`Tensor4all.InterpolativeQTT`:

| Symbol | Kind |
|--------|------|
| `LagrangePolynomials` | type |
| `getChebyshevGrid` | function |
| `interpolatesinglescale` | function |
| `interpolatemultiscale` | function |
| `interpolateadaptive` | function |
| `interpolatesinglescale_sparse` | function |
| `invertqtt` | function |
| `estimate_interpolation_error` | function |

Not re-exported: `Interval`, `NInterval`, `midpoint`, `split`, `intervallength`,
`angular_local_lagrange` (internal interval machinery and implementation helpers).

## Implementation Plan

### Step 1 — InterpolativeQTT.jl (upstream PR, merged first)

Add to `src/InterpolativeQTT.jl` before the `include` calls:

```julia
export LagrangePolynomials, getChebyshevGrid
export interpolatesinglescale, interpolatemultiscale, interpolateadaptive
export interpolatesinglescale_sparse
export invertqtt, estimate_interpolation_error
```

Merge this PR and note the commit hash. Register InterpolativeQTT.jl in the Julia
General Registry. Pin the registered version in Tensor4all.jl only after registration
is complete.

### Step 2 — New file `src/InterpolativeQTT.jl` in Tensor4all.jl

```julia
module InterpolativeQTT

import ..UpstreamInterpolativeQTT

function _reexportable_symbols()
    return filter(names(UpstreamInterpolativeQTT)) do sym
        sym !== nameof(UpstreamInterpolativeQTT) &&
            Base.isidentifier(sym) &&
            sym ∉ (:eval, :include)
    end
end

for sym in _reexportable_symbols()
    @eval const $(sym) = getfield(UpstreamInterpolativeQTT, $(QuoteNode(sym)))
    @eval export $(sym)
end

end
```

### Step 3 — `Project.toml`

- Add `InterpolativeQTT` under `[deps]` with its UUID (`87f1ea11-1d4d-47cb-b1d1-07788fc25290`).
- Add `InterpolativeQTT = "0.1"` under `[compat]`.

### Step 4 — `src/Tensor4all.jl`

- Add `import InterpolativeQTT as UpstreamInterpolativeQTT` alongside the existing
  upstream imports.
- Add `include("InterpolativeQTT.jl")` alongside the other submodule includes.
- Add `InterpolativeQTT` to the `export` list at the bottom.

### Step 5 — `docs/src/api.md`

Append `"InterpolativeQTT.jl"` to the `Pages = [...]` list in the relevant `@autodocs`
block so `scripts/check_autodocs_coverage.jl` stays green.

## Cross-repo Sequencing

Per AGENTS.md cross-repo development rules:

1. Develop and test the InterpolativeQTT.jl export PR locally.
2. Merge the InterpolativeQTT.jl PR.
3. Register InterpolativeQTT.jl in the General Registry; wait for registration.
4. Update `[compat]` pin in Tensor4all.jl to the registered version.
5. Open the Tensor4all.jl PR.

## Out of Scope

- No custom method wrappers or overrides at the Tensor4all boundary.
- No re-export of interval types or Base method extensions (`in`, `split`).
- No changes to `TensorNetworks`, `SimpleTT`, or any other Tensor4all submodule.
