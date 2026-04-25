# ITensorCompat Extension Design

## Goal

Extend `Tensor4all.jl`'s `ITensorCompat` module with the missing MPS/MPO operations needed to migrate `BubbleTeaCI.jl` and `QuanticsNEGF.jl` away from `ITensors.jl`/`ITensorMPS.jl`.

## Added API Surface

### Layer 1: MPS/MPO constructors and site-index operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `MPO(tensors::Vector{Tensor})` | constructor | Build MPO from tensor list; each tensor must have exactly 2 site indices |
| `prime(m::MPS)` | `MPS` | Return new MPS with siteinds' prime level +1 (tensor data shared) |
| `prime(m::MPO)` | `MPO` | Return new MPO with siteinds' prime level +1 |
| `prime!(m::MPS)` | `MPS` | In-place siteinds prime +1 |
| `prime!(m::MPO)` | `MPO` | In-place siteinds prime +1 |
| `replaceprime(m::MPS, pairs...)` | `MPS` | Replace prime levels in siteinds: `replaceprime(m, 1=>0)` |
| `replaceprime(m::MPO, pairs...)` | `MPO` | Same for MPO |
| `sim(siteinds, m::MPS)` | `Vector{Index}` | Return cloned siteinds with fresh IDs |
| `sim(siteinds, m::MPO)` | `Vector{Vector{Index}}` | Same for MPO (per-site index list) |

### Layer 2: Index `'` operator

| Function | Description |
|----------|-------------|
| `Base.adjoint(idx::Tensor4all.Index)` | `idx'` returns `prime(idx)` |

### Layer 3: Utility accessors

| Function | Description |
|----------|-------------|
| `maxlinkdim(m::MPS)` | = `rank(m)` alias |
| `maxlinkdim(m::MPO)` | = `rank(m)` alias |
| `data(m::MPS)` | `m.tt.data` — raw tensor vector |
| `data(w::MPO)` | `w.tt.data` — raw tensor vector |
| `data(tt::TensorTrain)` | `tt.data` — raw tensor vector |

## Design Decisions

### `prime(mps)` returns MPS, not Index vector
Matches ITensorMPS behavior: returns a new MPS wrapping the same internal tensors but with siteinds' prime levels incremented.

### `replaceprime` semantics
Scans siteinds and replaces any index whose `plev` matches the `old` value to have `new` value. Applied to all site groups (for MPO, applied to all indices in each site group).

### `Base.adjoint` on Index
Julia's `'` calls `adjoint()`. Index is not a linear operator, so defining `adjoint` as `prime` has no mathematical conflict. This pattern matches ITensors.jl's own `idx'` convention.

### `data()` accessor
Provides a uniform way to access the underlying `Vector{Tensor}` for inspection/debugging, matching `ITensors.data()` which accesses the raw storage.
