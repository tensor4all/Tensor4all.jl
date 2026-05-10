# InterpolativeQTT Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate InterpolativeQTT.jl into Tensor4all.jl as `Tensor4all.InterpolativeQTT`, re-exporting the eight QTT interpolation symbols using the same pattern as `Tensor4all.QuanticsGrids`.

**Architecture:** Add curated `export` statements to InterpolativeQTT.jl upstream, register it in the General Registry, then add it as a dependency of Tensor4all.jl and create a thin re-export wrapper module — identical in structure to `src/QuanticsGrids.jl`.

**Tech Stack:** Julia 1.9+, TensorCrossInterpolation.jl (already a shared dependency), Documenter.jl.

---

## Cross-repo Sequencing

Tasks 1–2 target the **InterpolativeQTT.jl** repo. Tasks 3–7 target **Tensor4all.jl**. Do not open the Tensor4all.jl PR until InterpolativeQTT.jl is merged **and** registered in the General Registry.

## File Map

| File | Action |
|------|--------|
| `InterpolativeQTT.jl/src/InterpolativeQTT.jl` | Modify — add 8 export statements |
| `Tensor4all.jl/src/InterpolativeQTT.jl` | Create — re-export wrapper module |
| `Tensor4all.jl/src/Tensor4all.jl` | Modify — import, include, export |
| `Tensor4all.jl/Project.toml` | Modify — add dep + compat entry |
| `Tensor4all.jl/test/interpolativeqtt/surface.jl` | Create — surface + functional tests |
| `Tensor4all.jl/test/runtests.jl` | Modify — include new test file |
| `Tensor4all.jl/docs/src/api.md` | Modify — add InterpolativeQTT section |

---

## Task 1: Add exports to InterpolativeQTT.jl

**Repo:** `InterpolativeQTT.jl`

**Files:**
- Modify: `src/InterpolativeQTT.jl`

- [ ] **Step 1.1: Write the failing export test**

Create `test/exports.jl` in the InterpolativeQTT.jl repo:

```julia
using Test
import InterpolativeQTT

@testset "exported symbols" begin
    exported = names(InterpolativeQTT)
    for sym in [
        :LagrangePolynomials, :getChebyshevGrid,
        :interpolatesinglescale, :interpolatemultiscale, :interpolateadaptive,
        :interpolatesinglescale_sparse, :invertqtt, :estimate_interpolation_error,
    ]
        @test sym ∈ exported
    end
    for sym in [:Interval, :NInterval, :midpoint, :intervallength, :angular_local_lagrange]
        @test sym ∉ exported
    end
end
```

- [ ] **Step 1.2: Run test to confirm it fails**

```bash
cd path/to/InterpolativeQTT.jl
env JULIA_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    BLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
    julia --startup-file=no --project=. test/exports.jl
```

Expected: test failures — none of the 8 symbols are in `names(InterpolativeQTT)`.

- [ ] **Step 1.3: Add exports to `src/InterpolativeQTT.jl`**

Open `src/InterpolativeQTT.jl`. Insert these lines immediately after the `module InterpolativeQTT` line, before the first `import`/`using`:

```julia
export LagrangePolynomials, getChebyshevGrid
export interpolatesinglescale, interpolatemultiscale, interpolateadaptive
export interpolatesinglescale_sparse
export invertqtt, estimate_interpolation_error
```

The file should now begin:

```julia
module InterpolativeQTT

export LagrangePolynomials, getChebyshevGrid
export interpolatesinglescale, interpolatemultiscale, interpolateadaptive
export interpolatesinglescale_sparse
export invertqtt, estimate_interpolation_error

import TensorCrossInterpolation as TCI
using LinearAlgebra

import Base: in

include("interval.jl")
include("interpolation.jl")
include("angular_lagrange.jl")

end
```

- [ ] **Step 1.4: Run test to confirm it passes**

```bash
env JULIA_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    BLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
    julia --startup-file=no --project=. test/exports.jl
```

Expected: all tests pass.

- [ ] **Step 1.5: Commit and open PR**

```bash
git add src/InterpolativeQTT.jl test/exports.jl
git commit -m "feat: export public QTT interpolation surface"
```

Open a PR in the InterpolativeQTT.jl repo. Merge it. Then register the package in the Julia General Registry (open a PR to [JuliaRegistries/General](https://github.com/JuliaRegistries/General) using the `Registrator.jl` bot or `LocalRegistry`). **Wait for registration to complete before proceeding to Task 2.**

---

## Task 2: Register InterpolativeQTT.jl as a dependency in Tensor4all.jl

**Repo:** `Tensor4all.jl`

**Files:**
- Modify: `Project.toml`

Once InterpolativeQTT.jl v0.1.1 is registered in the General Registry, add it to Tensor4all.jl:

- [ ] **Step 2.1: Add the dependency**

```bash
cd path/to/Tensor4all.jl
env JULIA_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    BLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
    julia --startup-file=no --project=. -e "using Pkg; Pkg.add(\"InterpolativeQTT\")"
```

> **During local development before registration:** replace the above with
> `Pkg.develop(path="path/to/InterpolativeQTT.jl")` to use a local checkout.
> Switch to the registered version (`Pkg.add`) once registration is complete.

- [ ] **Step 2.2: Verify `Project.toml` now contains**

```toml
[deps]
# ... existing deps ...
InterpolativeQTT = "87f1ea11-1d4d-47cb-b1d1-07788fc25290"

[compat]
# ... existing compat ...
InterpolativeQTT = "0.1"
```

If the `Pkg.add` did not add a `[compat]` entry, add it manually.

- [ ] **Step 2.3: Commit**

```bash
git add Project.toml Manifest.toml
git commit -m "deps: add InterpolativeQTT.jl dependency"
```

---

## Task 3: Write the failing surface and functional tests

**Files:**
- Create: `test/interpolativeqtt/surface.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 3.1: Create `test/interpolativeqtt/surface.jl`**

```julia
using Test
using Tensor4all
import TensorCrossInterpolation as TCI

@testset "InterpolativeQTT surface" begin
    @test isdefined(Tensor4all, :InterpolativeQTT)
    @test :InterpolativeQTT ∉ Tensor4all.InterpolativeQTT._reexportable_symbols()

    for sym in [
        :LagrangePolynomials, :getChebyshevGrid,
        :interpolatesinglescale, :interpolatemultiscale, :interpolateadaptive,
        :interpolatesinglescale_sparse, :invertqtt, :estimate_interpolation_error,
    ]
        @test isdefined(Tensor4all.InterpolativeQTT, sym)
    end

    for sym in [:Interval, :NInterval, :angular_local_lagrange]
        @test !isdefined(Tensor4all.InterpolativeQTT, sym)
    end
end

@testset "InterpolativeQTT functional" begin
    IQTT = Tensor4all.InterpolativeQTT

    f = x -> exp(-x^2)
    tt = IQTT.interpolatesinglescale(f, -2.0, 2.0, 8, 20)
    @test tt isa TCI.TensorTrain{Float64, 3}
    @test length(tt) == 8

    P = IQTT.getChebyshevGrid(5)
    @test P isa IQTT.LagrangePolynomials{Float64}
    @test length(P.grid) == 6
end
```

- [ ] **Step 3.2: Add the include to `test/runtests.jl`**

Add this line at the end of `test/runtests.jl`, before the HDF5 block:

```julia
include("interpolativeqtt/surface.jl")
```

- [ ] **Step 3.3: Run tests to confirm they fail**

```bash
env JULIA_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    BLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
    julia --startup-file=no --project=. test/runtests.jl 2>&1 | grep -A5 "InterpolativeQTT"
```

Expected: `UndefVarError: InterpolativeQTT not defined` or similar.

---

## Task 4: Create the re-export wrapper and wire it in

**Files:**
- Create: `src/InterpolativeQTT.jl`
- Modify: `src/Tensor4all.jl`

- [ ] **Step 4.1: Create `src/InterpolativeQTT.jl`**

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

- [ ] **Step 4.2: Modify `src/Tensor4all.jl`**

Add the upstream import alongside the existing ones (after `import QuanticsTCI as UpstreamQuanticsTCI`):

```julia
import InterpolativeQTT as UpstreamInterpolativeQTT
```

Add the include alongside the other submodule includes (after `include("QuanticsTCI.jl")`):

```julia
include("InterpolativeQTT.jl")
```

Add `InterpolativeQTT` to the export list at the bottom of the file (alongside `QuanticsGrids`, `QuanticsTCI`, etc.):

```julia
export TensorNetworks, ITensorCompat, SimpleTT, TensorCI, QuanticsGrids, QuanticsTCI, QuanticsTransform, InterpolativeQTT
```

- [ ] **Step 4.3: Run the full test suite to confirm the new tests pass**

```bash
env JULIA_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    BLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
    julia --startup-file=no --project=. test/runtests.jl
```

Expected: all tests pass, including the two new `InterpolativeQTT` testsets.

- [ ] **Step 4.4: Commit**

```bash
git add src/InterpolativeQTT.jl src/Tensor4all.jl test/interpolativeqtt/surface.jl test/runtests.jl
git commit -m "feat: add Tensor4all.InterpolativeQTT re-export module"
```

---

## Task 5: Update `docs/src/api.md`

**Files:**
- Modify: `docs/src/api.md`

- [ ] **Step 5.1: Extend the Adopted Modules section**

Find the `## Adopted Modules` section in `docs/src/api.md`. It currently covers QuanticsGrids and QuanticsTCI. Add the InterpolativeQTT entry to the opening bullet list and append an `@autodocs` block.

Replace the existing `## Adopted Modules` section with:

```markdown
## Adopted Modules

- `Tensor4all.QuanticsGrids` re-exports the public `QuanticsGrids.jl` surface
- `Tensor4all.QuanticsTCI` re-exports the public `QuanticsTCI.jl` surface
- `Tensor4all.InterpolativeQTT` re-exports the public `InterpolativeQTT.jl` surface

These modules are wrapper re-exports for discoverability. Tensor4all does not
take ownership of their APIs; see the upstream docs for the complete surface.

[QuanticsGrids.jl docs](https://tensor4all.org/QuanticsGrids.jl/dev/apireference/)
[QuanticsTCI.jl docs](https://tensor4all.org/QuanticsTCI.jl/dev/apireference/)
[InterpolativeQTT.jl docs](https://tensor4all.github.io/InterpolativeQTT.jl/dev)

```julia
const QG = Tensor4all.QuanticsGrids

grid = QG.DiscretizedGrid(3, 0.0, 1.0)
grid_index = QG.origcoord_to_grididx(grid, 0.5)
quantics_index = QG.grididx_to_quantics(grid, grid_index)
QG.quantics_to_origcoord(grid, quantics_index)  # 0.5
```

```julia
const QTCI = Tensor4all.QuanticsTCI

qtt, ranks, errors = QTCI.quanticscrossinterpolate(
    Float64,
    x -> sin(x),
    range(0, 1; length=8);
    tolerance=1e-8,
)

qtt(4)      # evaluate at an integer grid index
sum(qtt)    # upstream QuanticsTCI reduction
```

```julia
const IQTT = Tensor4all.InterpolativeQTT

tt = IQTT.interpolatesinglescale(x -> exp(-x^2), -2.0, 2.0, 8, 20)
```

```@autodocs
Modules = [Tensor4all.InterpolativeQTT]
Pages = ["InterpolativeQTT.jl"]
Private = false
Order = [:type, :function]
```
```

- [ ] **Step 5.2: Verify the docs build cleanly**

```bash
env JULIA_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    BLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
    julia --startup-file=no --project=docs docs/make.jl
```

Expected: no errors, no warnings about missing docstrings.

- [ ] **Step 5.3: Run autodocs coverage check**

```bash
env JULIA_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    BLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
    julia --startup-file=no --project=. scripts/check_autodocs_coverage.jl
```

Expected: exits 0 with no missing pages reported.

- [ ] **Step 5.4: Commit**

```bash
git add docs/src/api.md
git commit -m "docs: add InterpolativeQTT to api.md Adopted Modules section"
```
