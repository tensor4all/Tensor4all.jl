# Tensor4all.jl API Expansion Design

## Goal

Expand Tensor4all.jl to wrap the new C API functions (#393), unify indexing conventions to 1-indexed, align naming with Pure Julia ecosystem (TensorCrossInterpolation.jl, QuanticsTCI.jl), and add type conversion between SimpleTT and TreeTN.

## Principles

- All Julia-facing APIs use **1-indexed** (Julia convention). Internal C API calls convert to 0-indexed.
- Function names match Pure Julia libraries where equivalent functionality exists.
- No backward compatibility aliases. Breaking changes are acceptable.
- Explicit type conversions — no implicit magic.

---

## A. Indexing Convention: 1-indexed everywhere

### SimpleTT

| Function | Current | Change |
|---|---|---|
| `evaluate(tt, indices)` | 0-indexed | 1-indexed; subtract 1 internally |
| `sitetensor(tt, site)` | 0-indexed | 1-indexed; subtract 1 internally |
| Callable `tt(indices...)` | 0-indexed | 1-indexed |

### TreeTCI

| Function | Current | Change |
|---|---|---|
| `initialpivots` in `crossinterpolate2` | 0-indexed | 1-indexed; subtract 1 internally |
| `evaluate(ttn, indices)` | 0-indexed | 1-indexed; subtract 1 internally |
| Callback `f(batch)` | 0-indexed | 1-indexed; add 1 when passing to Julia callback |

### TreeTN

No change — already 1-indexed.

---

## B. Naming Alignment

Rename to match Pure Julia convention (no underscores):

| Current | New | Module |
|---|---|---|
| `link_dims` | `linkdims` | SimpleTT |
| `site_dims` | `sitedims` | SimpleTT |
| `site_tensor` | `sitetensor` | SimpleTT |
| `local_dimensions` | `localdimensions` | QuanticsGrids |
| `link_dims` | `linkdims` | QuanticsTCI |

No deprecated aliases. Direct rename.

---

## C. SimpleTT Basic Operations

New functions in `src/SimpleTT.jl`. All support Float64 and ComplexF64.

### Arithmetic (operator overloading)

```julia
Base.:+(a::SimpleTensorTrain{T}, b::SimpleTensorTrain{T}) where T
    → SimpleTensorTrain{T}   # calls t4a_simplett_{f64,c64}_add

Base.:-(a::SimpleTensorTrain{T}, b::SimpleTensorTrain{T}) where T
    → SimpleTensorTrain{T}   # add(a, scaled(b, -1))

Base.:*(α::Number, tt::SimpleTensorTrain{T}) where T
    → SimpleTensorTrain{T}   # clone + scale

Base.:*(tt::SimpleTensorTrain{T}, α::Number) where T
    → SimpleTensorTrain{T}   # α * tt

LinearAlgebra.dot(a::SimpleTensorTrain{T}, b::SimpleTensorTrain{T}) where T
    → T                      # calls t4a_simplett_{f64,c64}_dot
```

### In-place

```julia
scale!(tt::SimpleTensorTrain, α::Number) → tt
    # calls t4a_simplett_{f64,c64}_scale
```

### Other operations

```julia
Base.reverse(tt::SimpleTensorTrain{T}) where T → SimpleTensorTrain{T}
    # calls t4a_simplett_{f64,c64}_reverse

fulltensor(tt::SimpleTensorTrain{T}) where T → Array{T}
    # calls t4a_simplett_{f64,c64}_fulltensor
    # returns Array with dimensions = sitedims(tt)
```

### Constructor from site tensors

```julia
SimpleTensorTrain(site_tensors::Vector{<:AbstractArray{T,3}}) where T<:Union{Float64,ComplexF64}
    # calls t4a_simplett_{f64,c64}_from_site_tensors
    # site_tensors[i] has shape (left_dim, site_dim, right_dim)
```

---

## D. SimpleTT ↔ TreeTN.MPS Conversion

### In TreeTN.jl

```julia
function MPS(tt::SimpleTensorTrain{T}) where T
    # Extract site tensors from SimpleTT (1-indexed after B)
    # Create Index objects for sites and links
    # Build Tensor objects and call MPS(tensors::Vector{Tensor})
end
```

### In SimpleTT.jl

```julia
function SimpleTensorTrain(mps::TreeTensorNetwork{Int})
    # For each site 1..n, extract tensor data via TreeTN accessors
    # Build 3D arrays (left, site, right)
    # Call SimpleTensorTrain(site_tensors)
end
```

Both conversions are pure Julia — no new C API needed. They compose existing accessors.

---

## E. QuanticsTCI Update

### Type parameterization

```julia
mutable struct QuanticsTensorCI2{V}
    ptr::Ptr{Cvoid}
    # V determines which C API functions to call (f64 vs c64)
end
```

### Updated main function

```julia
function quanticscrossinterpolate(
    ::Type{V}, f, grid::DiscretizedGrid;
    tolerance=1e-8, maxbonddim=0, maxiter=200,
    initialpivots=nothing, nrandominitpivot=5,
    verbosity=0, unfoldingscheme=:interleaved,
    nsearchglobalpivot=5, nsearch=100,
    normalize_error=true
) where V <: Union{Float64, ComplexF64}
    → (QuanticsTensorCI2{V}, Vector{Int}, Vector{Float64})
end
```

### Overloads

```julia
# Discrete domain (size tuple)
quanticscrossinterpolate(::Type{V}, f, size::NTuple{N,Int}; kwargs...)

# From coordinate arrays
quanticscrossinterpolate(::Type{V}, f, xvals::Vector{Vector{Float64}}; kwargs...)

# From dense array
quanticscrossinterpolate(F::Array{V}; kwargs...)
```

### New accessors

```julia
max_bond_error(qtci::QuanticsTensorCI2) → Float64
max_rank(qtci::QuanticsTensorCI2) → Int
```

### Internal: QtciOptions management

Julia kwargs → create `t4a_qtci_options` handle → pass to C API → release after call. The options handle is ephemeral (not stored).

---

## F. TCI Conversion Bridge

New file: `ext/Tensor4allTCIExt.jl`

Weak dependency on `TensorCrossInterpolation`.

```julia
function SimpleTensorTrain(tt::TCI.TensorTrain{T}) where T
    # Extract tt.sitetensors::Vector{Array{T,3}}
    # Call SimpleTensorTrain(site_tensors)
end

function TCI.TensorTrain(stt::SimpleTensorTrain{T}) where T
    # Extract site tensors via sitetensor(stt, i) for i in 1:length(stt)
    # Build TCI.TensorTrain from Vector{Array{T,3}}
end
```

---

## Scope Exclusions

- QuanticsTransform extensions (multivar operators, etc.)
- TreeTCI Julia wrapper overhaul (beyond indexing fix)
- ITensorsExt updates
- QuanticsGrids naming beyond `localdimensions`
- `cachedata`, `quanticsfouriermpo`
- `compress!` for SimpleTT (already exists via C API)
