# Tensor4all.jl API Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand Tensor4all.jl to wrap new C API functions, unify to 1-indexed, align naming with Pure Julia ecosystem, add ComplexF64 support, and add type conversions.

**Architecture:** Add ccall bindings in C_API.jl for ~40 new Rust C API functions. Refactor SimpleTT.jl for ComplexF64 support via type dispatch (_f64/_c64). Rename functions to match Pure Julia convention (no underscores). Convert all user-facing indices to 1-based. Add operator overloading. Update QuanticsTCI for ComplexF64 and tuple returns. Add SimpleTT↔TreeTN conversion and TCI extension.

**Tech Stack:** Julia, ccall FFI, Tensor4all.jl modules (SimpleTT, TreeTN, QuanticsTCI, QuanticsGrids, C_API)

**Spec:** `docs/specs/2026-03-31-api-expansion-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/C_API.jl` | Modify | Add ccall bindings for ~40 new C API functions |
| `src/SimpleTT.jl` | Rewrite | ComplexF64, 1-indexed, renamed functions, new operations |
| `src/QuanticsTCI.jl` | Rewrite | ComplexF64, tuple return, QtciOptions, new accessors |
| `src/QuanticsGrids.jl` | Modify | Rename `local_dimensions` → `localdimensions` |
| `src/TreeTN.jl` | Modify | Add `MPS(tt::SimpleTensorTrain)` constructor |
| `src/TreeTCI.jl` | Modify | 1-indexed pivots/evaluate/callbacks |
| `ext/Tensor4allTCIExt.jl` | Create | TCI.TensorTrain ↔ SimpleTensorTrain conversion |
| `Project.toml` | Modify | Add TensorCrossInterpolation weak dep |
| `test/test_simplett.jl` | Rewrite | Tests for all SimpleTT changes |
| `test/test_quanticstci.jl` | Create | Tests for QuanticsTCI updates |
| `test/test_treetci.jl` | Create | Tests for TreeTCI 1-indexed |
| `test/test_conversions.jl` | Create | Tests for SimpleTT↔TreeTN conversion |
| `test/runtests.jl` | Modify | Include new test files |

---

## Task 1: C_API.jl — Add SimpleTT new function bindings

**Files:**
- Modify: `src/C_API.jl`

Add ccall wrappers for all new SimpleTT C API functions. Follow existing patterns in C_API.jl (look at `t4a_simplett_f64_evaluate` etc. for reference).

- [ ] **Step 1: Add f64 operation bindings**

Add to C_API.jl after existing simplett_f64 functions:

```julia
# SimpleTT f64 new operations
function t4a_simplett_f64_from_site_tensors(n_sites, left_dims, site_dims, right_dims, data, data_len, out_ptr)
    ccall(_sym(:t4a_simplett_f64_from_site_tensors), Cint,
        (Csize_t, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Cdouble}, Csize_t, Ptr{Ptr{Cvoid}}),
        n_sites, left_dims, site_dims, right_dims, data, data_len, out_ptr)
end

function t4a_simplett_f64_add(a, b, out)
    ccall(_sym(:t4a_simplett_f64_add), Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), a, b, out)
end

function t4a_simplett_f64_scale(ptr, factor)
    ccall(_sym(:t4a_simplett_f64_scale), Cint,
        (Ptr{Cvoid}, Cdouble), ptr, factor)
end

function t4a_simplett_f64_dot(a, b, out_value)
    ccall(_sym(:t4a_simplett_f64_dot), Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cdouble}), a, b, out_value)
end

function t4a_simplett_f64_reverse(ptr, out)
    ccall(_sym(:t4a_simplett_f64_reverse), Cint,
        (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), ptr, out)
end

function t4a_simplett_f64_fulltensor(ptr, out_data, buf_len, out_data_len)
    ccall(_sym(:t4a_simplett_f64_fulltensor), Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ptr{Csize_t}), ptr, out_data, buf_len, out_data_len)
end
```

- [ ] **Step 2: Add c64 SimpleTT bindings (all functions including existing equivalents)**

Add full c64 SimpleTT API. c64 needs release, clone, len, sitedims, linkdims, rank, evaluate, sum, norm, site_tensor, compress, partial_sum, PLUS the new operations:

```julia
# SimpleTT c64 lifecycle
function t4a_simplett_c64_release(ptr)
    ccall(_sym(:t4a_simplett_c64_release), Cvoid, (Ptr{Cvoid},), ptr)
end

function t4a_simplett_c64_clone(ptr)
    ccall(_sym(:t4a_simplett_c64_clone), Ptr{Cvoid}, (Ptr{Cvoid},), ptr)
end

function t4a_simplett_c64_constant(site_dims, value_re, value_im)
    ccall(_sym(:t4a_simplett_c64_constant), Ptr{Cvoid},
        (Ptr{Csize_t}, Cdouble, Cdouble), site_dims, value_re, value_im)
end

function t4a_simplett_c64_zeros(site_dims)
    ccall(_sym(:t4a_simplett_c64_zeros), Ptr{Cvoid}, (Ptr{Csize_t},), site_dims)
end

function t4a_simplett_c64_len(ptr, out_len)
    ccall(_sym(:t4a_simplett_c64_len), Cint, (Ptr{Cvoid}, Ptr{Csize_t}), ptr, out_len)
end

function t4a_simplett_c64_site_dims(ptr, out_dims)
    ccall(_sym(:t4a_simplett_c64_site_dims), Cint, (Ptr{Cvoid}, Ptr{Csize_t}), ptr, out_dims)
end

function t4a_simplett_c64_link_dims(ptr, out_dims)
    ccall(_sym(:t4a_simplett_c64_link_dims), Cint, (Ptr{Cvoid}, Ptr{Csize_t}), ptr, out_dims)
end

function t4a_simplett_c64_rank(ptr, out_rank)
    ccall(_sym(:t4a_simplett_c64_rank), Cint, (Ptr{Cvoid}, Ptr{Csize_t}), ptr, out_rank)
end

function t4a_simplett_c64_evaluate(ptr, indices, out_re, out_im)
    ccall(_sym(:t4a_simplett_c64_evaluate), Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Cdouble}, Ptr{Cdouble}), ptr, indices, out_re, out_im)
end

function t4a_simplett_c64_sum(ptr, out_re, out_im)
    ccall(_sym(:t4a_simplett_c64_sum), Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}), ptr, out_re, out_im)
end

function t4a_simplett_c64_norm(ptr, out_value)
    ccall(_sym(:t4a_simplett_c64_norm), Cint, (Ptr{Cvoid}, Ptr{Cdouble}), ptr, out_value)
end

function t4a_simplett_c64_site_tensor(ptr, site, out_data, buf_len, out_left, out_site, out_right)
    ccall(_sym(:t4a_simplett_c64_site_tensor), Cint,
        (Ptr{Cvoid}, Csize_t, Ptr{Cdouble}, Csize_t, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}),
        ptr, site, out_data, buf_len, out_left, out_site, out_right)
end

function t4a_simplett_c64_compress(ptr, method, tolerance, max_bonddim)
    ccall(_sym(:t4a_simplett_c64_compress), Cint,
        (Ptr{Cvoid}, Cint, Cdouble, Csize_t), ptr, method, tolerance, max_bonddim)
end

function t4a_simplett_c64_partial_sum(ptr, dims, out)
    ccall(_sym(:t4a_simplett_c64_partial_sum), Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Ptr{Cvoid}}), ptr, dims, out)
end

# c64 new operations
function t4a_simplett_c64_from_site_tensors(n_sites, left_dims, site_dims, right_dims, data, data_len, out_ptr)
    ccall(_sym(:t4a_simplett_c64_from_site_tensors), Cint,
        (Csize_t, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Cdouble}, Csize_t, Ptr{Ptr{Cvoid}}),
        n_sites, left_dims, site_dims, right_dims, data, data_len, out_ptr)
end

function t4a_simplett_c64_add(a, b, out)
    ccall(_sym(:t4a_simplett_c64_add), Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), a, b, out)
end

function t4a_simplett_c64_scale(ptr, factor_re, factor_im)
    ccall(_sym(:t4a_simplett_c64_scale), Cint,
        (Ptr{Cvoid}, Cdouble, Cdouble), ptr, factor_re, factor_im)
end

function t4a_simplett_c64_dot(a, b, out_re, out_im)
    ccall(_sym(:t4a_simplett_c64_dot), Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}), a, b, out_re, out_im)
end

function t4a_simplett_c64_reverse(ptr, out)
    ccall(_sym(:t4a_simplett_c64_reverse), Cint,
        (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), ptr, out)
end

function t4a_simplett_c64_fulltensor(ptr, out_data, buf_len, out_data_len)
    ccall(_sym(:t4a_simplett_c64_fulltensor), Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ptr{Csize_t}), ptr, out_data, buf_len, out_data_len)
end
```

- [ ] **Step 3: Verify C_API loads**

Run: `julia --startup-file=no -e "using Pkg; Pkg.activate(\".\"); include(\"src/C_API.jl\")"`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add src/C_API.jl
git commit -m "feat(C_API): add ccall bindings for SimpleTT new operations (f64+c64)"
```

---

## Task 2: C_API.jl — Add QuanticsTCI new function bindings

**Files:**
- Modify: `src/C_API.jl`

- [ ] **Step 1: Add QtciOptions bindings**

```julia
# QtciOptions lifecycle
function t4a_qtci_options_default()
    ccall(_sym(:t4a_qtci_options_default), Ptr{Cvoid}, ())
end

function t4a_qtci_options_release(ptr)
    ccall(_sym(:t4a_qtci_options_release), Cvoid, (Ptr{Cvoid},), ptr)
end

function t4a_qtci_options_clone(ptr)
    ccall(_sym(:t4a_qtci_options_clone), Ptr{Cvoid}, (Ptr{Cvoid},), ptr)
end

# QtciOptions setters
function t4a_qtci_options_set_tolerance(ptr, tol)
    ccall(_sym(:t4a_qtci_options_set_tolerance), Cint, (Ptr{Cvoid}, Cdouble), ptr, tol)
end

function t4a_qtci_options_set_maxbonddim(ptr, dim)
    ccall(_sym(:t4a_qtci_options_set_maxbonddim), Cint, (Ptr{Cvoid}, Csize_t), ptr, dim)
end

function t4a_qtci_options_set_maxiter(ptr, iter)
    ccall(_sym(:t4a_qtci_options_set_maxiter), Cint, (Ptr{Cvoid}, Csize_t), ptr, iter)
end

function t4a_qtci_options_set_nrandominitpivot(ptr, n)
    ccall(_sym(:t4a_qtci_options_set_nrandominitpivot), Cint, (Ptr{Cvoid}, Csize_t), ptr, n)
end

function t4a_qtci_options_set_unfoldingscheme(ptr, scheme)
    ccall(_sym(:t4a_qtci_options_set_unfoldingscheme), Cint, (Ptr{Cvoid}, Cint), ptr, scheme)
end

function t4a_qtci_options_set_normalize_error(ptr, flag)
    ccall(_sym(:t4a_qtci_options_set_normalize_error), Cint, (Ptr{Cvoid}, Cint), ptr, flag)
end

function t4a_qtci_options_set_verbosity(ptr, level)
    ccall(_sym(:t4a_qtci_options_set_verbosity), Cint, (Ptr{Cvoid}, Csize_t), ptr, level)
end

function t4a_qtci_options_set_nsearchglobalpivot(ptr, n)
    ccall(_sym(:t4a_qtci_options_set_nsearchglobalpivot), Cint, (Ptr{Cvoid}, Csize_t), ptr, n)
end

function t4a_qtci_options_set_nsearch(ptr, n)
    ccall(_sym(:t4a_qtci_options_set_nsearch), Cint, (Ptr{Cvoid}, Csize_t), ptr, n)
end
```

- [ ] **Step 2: Add QuanticsTCI c64 + updated f64 bindings**

Note: The existing `t4a_quanticscrossinterpolate_f64` signature has CHANGED (breaking). It now takes options, initial_pivots, and convergence output buffers. Update the existing binding AND add c64 variants.

```julia
# Updated f64 crossinterpolate (BREAKING: new signature)
function t4a_quanticscrossinterpolate_f64(grid, eval_fn, user_data, options,
        tolerance, max_bonddim, max_iter, initial_pivots, n_pivots,
        out_qtci, out_ranks, out_errors, out_n_iters)
    ccall(_sym(:t4a_quanticscrossinterpolate_f64), Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
         Cdouble, Csize_t, Csize_t, Ptr{Int64}, Csize_t,
         Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Cdouble}, Ptr{Csize_t}),
        grid, eval_fn, user_data, options,
        tolerance, max_bonddim, max_iter, initial_pivots, n_pivots,
        out_qtci, out_ranks, out_errors, out_n_iters)
end

# Updated discrete f64 (BREAKING: new signature)
function t4a_quanticscrossinterpolate_discrete_f64(sizes, ndims, eval_fn, user_data, options,
        tolerance, max_bonddim, max_iter, unfoldingscheme, initial_pivots, n_pivots,
        out_qtci, out_ranks, out_errors, out_n_iters)
    ccall(_sym(:t4a_quanticscrossinterpolate_discrete_f64), Cint,
        (Ptr{Csize_t}, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
         Cdouble, Csize_t, Csize_t, Cint, Ptr{Int64}, Csize_t,
         Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Cdouble}, Ptr{Csize_t}),
        sizes, ndims, eval_fn, user_data, options,
        tolerance, max_bonddim, max_iter, unfoldingscheme, initial_pivots, n_pivots,
        out_qtci, out_ranks, out_errors, out_n_iters)
end

# c64 crossinterpolate
function t4a_quanticscrossinterpolate_c64(grid, eval_fn, user_data, options,
        tolerance, max_bonddim, max_iter, initial_pivots, n_pivots,
        out_qtci, out_ranks, out_errors, out_n_iters)
    ccall(_sym(:t4a_quanticscrossinterpolate_c64), Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
         Cdouble, Csize_t, Csize_t, Ptr{Int64}, Csize_t,
         Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Cdouble}, Ptr{Csize_t}),
        grid, eval_fn, user_data, options,
        tolerance, max_bonddim, max_iter, initial_pivots, n_pivots,
        out_qtci, out_ranks, out_errors, out_n_iters)
end

function t4a_quanticscrossinterpolate_discrete_c64(sizes, ndims, eval_fn, user_data, options,
        tolerance, max_bonddim, max_iter, unfoldingscheme, initial_pivots, n_pivots,
        out_qtci, out_ranks, out_errors, out_n_iters)
    ccall(_sym(:t4a_quanticscrossinterpolate_discrete_c64), Cint,
        (Ptr{Csize_t}, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
         Cdouble, Csize_t, Csize_t, Cint, Ptr{Int64}, Csize_t,
         Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Cdouble}, Ptr{Csize_t}),
        sizes, ndims, eval_fn, user_data, options,
        tolerance, max_bonddim, max_iter, unfoldingscheme, initial_pivots, n_pivots,
        out_qtci, out_ranks, out_errors, out_n_iters)
end

# c64 QTCI lifecycle + accessors
function t4a_qtci_c64_release(ptr)
    ccall(_sym(:t4a_qtci_c64_release), Cvoid, (Ptr{Cvoid},), ptr)
end

function t4a_qtci_c64_clone(ptr)
    ccall(_sym(:t4a_qtci_c64_clone), Ptr{Cvoid}, (Ptr{Cvoid},), ptr)
end

function t4a_qtci_c64_rank(ptr, out_rank)
    ccall(_sym(:t4a_qtci_c64_rank), Cint, (Ptr{Cvoid}, Ptr{Csize_t}), ptr, out_rank)
end

function t4a_qtci_c64_link_dims(ptr, out_dims, buf_len)
    ccall(_sym(:t4a_qtci_c64_link_dims), Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t), ptr, out_dims, buf_len)
end

function t4a_qtci_c64_evaluate(ptr, indices, n_indices, out_re, out_im)
    ccall(_sym(:t4a_qtci_c64_evaluate), Cint,
        (Ptr{Cvoid}, Ptr{Int64}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}), ptr, indices, n_indices, out_re, out_im)
end

function t4a_qtci_c64_sum(ptr, out_re, out_im)
    ccall(_sym(:t4a_qtci_c64_sum), Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}), ptr, out_re, out_im)
end

function t4a_qtci_c64_integral(ptr, out_re, out_im)
    ccall(_sym(:t4a_qtci_c64_integral), Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}), ptr, out_re, out_im)
end

function t4a_qtci_c64_to_tensor_train(ptr)
    ccall(_sym(:t4a_qtci_c64_to_tensor_train), Ptr{Cvoid}, (Ptr{Cvoid},), ptr)
end

# TreeTCI2 state accessors (f64 + c64)
function t4a_qtci_f64_max_bond_error(ptr, out_value)
    ccall(_sym(:t4a_qtci_f64_max_bond_error), Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}), ptr, out_value)
end

function t4a_qtci_f64_max_rank(ptr, out_rank)
    ccall(_sym(:t4a_qtci_f64_max_rank), Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}), ptr, out_rank)
end

function t4a_qtci_c64_max_bond_error(ptr, out_value)
    ccall(_sym(:t4a_qtci_c64_max_bond_error), Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}), ptr, out_value)
end

function t4a_qtci_c64_max_rank(ptr, out_rank)
    ccall(_sym(:t4a_qtci_c64_max_rank), Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}), ptr, out_rank)
end
```

- [ ] **Step 3: Commit**

```bash
git add src/C_API.jl
git commit -m "feat(C_API): add ccall bindings for QtciOptions, QuanticsTCI c64, state accessors"
```

---

## Task 3: Rewrite SimpleTT.jl

Complete rewrite of `src/SimpleTT.jl` to support ComplexF64, 1-indexed, renamed functions, and new operations. The implementer should read the current file first, then replace it entirely.

**Files:**
- Modify: `src/SimpleTT.jl`

**Key design decisions for the implementer:**

1. **Type dispatch pattern:** Use `_suffix(::Type{Float64}) = "f64"` and `_api(T, name) = getfield(C_API, Symbol("t4a_simplett_", _suffix(T), "_", name))` pattern (already exists in current code) for dispatching f64/c64 calls.

2. **ComplexF64 type parameter:** `SimpleTensorTrain{T}` where `T <: Union{Float64, ComplexF64}`. The `_SimpleTTScalar = Union{Float64, ComplexF64}`.

3. **1-indexed:** All user-facing functions use 1-based indices. Internal C API calls subtract 1.

4. **Renamed functions (no aliases):**
   - `site_dims` → `sitedims`
   - `link_dims` → `linkdims`
   - `site_tensor` → `sitetensor`

5. **New operations to add:**
   - `Base.:+(a, b)` — calls `_add` C API
   - `Base.:-(a, b)` — `a + (-1 * b)`
   - `Base.:*(α, tt)` and `Base.:*(tt, α)` — clone + scale
   - `LinearAlgebra.dot(a, b)` — calls `_dot` C API
   - `scale!(tt, α)` — in-place, calls `_scale` C API
   - `Base.reverse(tt)` — calls `_reverse` C API
   - `fulltensor(tt)` — calls `_fulltensor` C API
   - `SimpleTensorTrain(site_tensors::Vector{<:AbstractArray{T,3}})` — calls `_from_site_tensors`

6. **ComplexF64 specifics:**
   - `evaluate` returns `ComplexF64` via (out_re, out_im)
   - `sum` returns `ComplexF64` via (out_re, out_im)
   - `sitetensor` returns `Array{ComplexF64,3}` by reinterpreting interleaved doubles
   - `scale!(tt, α::Complex)` passes (re, im)
   - `dot` returns `ComplexF64` via (out_re, out_im)
   - `from_site_tensors` converts ComplexF64 data to interleaved doubles
   - `fulltensor` reinterprets interleaved doubles to ComplexF64

- [ ] **Step 1: Rewrite SimpleTT.jl**

The implementer should write the complete new SimpleTT.jl following the design decisions above. Reference the existing code for patterns, but produce a complete rewrite. All functions must work for both Float64 and ComplexF64 via type dispatch.

- [ ] **Step 2: Verify it compiles**

Run: `julia --startup-file=no -e "using Pkg; Pkg.activate(\".\"); Pkg.instantiate(); using Tensor4all; using Tensor4all.SimpleTT"`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add src/SimpleTT.jl
git commit -m "feat(SimpleTT): rewrite with ComplexF64, 1-indexed, renamed functions, new operations"
```

---

## Task 4: Rewrite test_simplett.jl

**Files:**
- Modify: `test/test_simplett.jl`
- Modify: `test/runtests.jl` — add `include("test_simplett.jl")` if not present

- [ ] **Step 1: Write comprehensive tests**

Tests must cover for BOTH Float64 and ComplexF64:
- Construction: constant, zeros, from site tensors
- Accessors: length, sitedims, linkdims, rank
- Evaluation: 1-indexed evaluate, callable interface `tt(1, 2, 3)`
- Site tensor: 1-indexed sitetensor
- Arithmetic: `+`, `-`, `*` (scalar), `dot`
- In-place: `scale!`
- Other: `reverse`, `fulltensor`, `copy`, `norm`, `sum`
- Verify 1-indexing: `evaluate(tt, [1,1,1])` works, `evaluate(tt, [0,0,0])` errors

- [ ] **Step 2: Run tests**

Run: `julia --startup-file=no --project=. -e "using Pkg; Pkg.test()"`
Or: `julia --startup-file=no --project=. test/test_simplett.jl`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add test/test_simplett.jl test/runtests.jl
git commit -m "test(SimpleTT): comprehensive tests for rewritten module"
```

---

## Task 5: Rename QuanticsGrids.localdimensions

**Files:**
- Modify: `src/QuanticsGrids.jl`

- [ ] **Step 1: Rename function**

In `src/QuanticsGrids.jl`, rename `local_dimensions` → `localdimensions` for both `DiscretizedGrid` and `InherentDiscreteGrid`. Update the export list.

- [ ] **Step 2: Update callers**

Search for `local_dimensions` in all src/ files and update to `localdimensions`. Key callers: `src/QuanticsTCI.jl`.

- [ ] **Step 3: Commit**

```bash
git add src/QuanticsGrids.jl src/QuanticsTCI.jl
git commit -m "refactor: rename local_dimensions to localdimensions"
```

---

## Task 6: Rewrite QuanticsTCI.jl

**Files:**
- Modify: `src/QuanticsTCI.jl`

**Key changes:**
1. `QuanticsTensorCI2{V}` — type-parameterized, V = Float64 or ComplexF64
2. Return `(qtci, ranks, errors)` tuple from `quanticscrossinterpolate`
3. First argument is `::Type{V}` (matching Pure Julia API)
4. Build `t4a_qtci_options` handle from kwargs, pass to C API, release after
5. Pass `initial_pivots` if provided
6. Receive `out_ranks`, `out_errors`, `out_n_iters` from C API
7. Add `max_bond_error(qtci)` and `max_rank(qtci)` accessors
8. Add c64 callback trampolines (result writes [re, im])
9. `evaluate` for c64 returns ComplexF64
10. `sum`, `integral` for c64 return ComplexF64
11. `to_tensor_train` for c64 returns `SimpleTensorTrain{ComplexF64}`
12. Add overloads: size tuple, xvals, Array (internally construct grid then call main function)

- [ ] **Step 1: Rewrite QuanticsTCI.jl**

Complete rewrite. Reference existing code for callback trampoline patterns.

- [ ] **Step 2: Commit**

```bash
git add src/QuanticsTCI.jl
git commit -m "feat(QuanticsTCI): rewrite with ComplexF64, tuple return, QtciOptions, overloads"
```

---

## Task 7: Write QuanticsTCI tests

**Files:**
- Create: `test/test_quanticstci.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write tests**

Cover:
- `quanticscrossinterpolate(Float64, f, grid)` returns (qtci, ranks, errors)
- `quanticscrossinterpolate(ComplexF64, f, grid)` works
- Discrete variant with size tuple
- `evaluate`, `sum`, `integral` for both Float64 and ComplexF64
- `max_bond_error`, `max_rank`
- `to_tensor_train` returns correct type
- kwargs: tolerance, maxbonddim, maxiter, verbosity

- [ ] **Step 2: Run tests and commit**

```bash
git add test/test_quanticstci.jl test/runtests.jl
git commit -m "test(QuanticsTCI): comprehensive tests for rewritten module"
```

---

## Task 8: TreeTCI 1-indexed

**Files:**
- Modify: `src/TreeTCI.jl`

- [ ] **Step 1: Update indexing**

Changes needed:
- `crossinterpolate2`: `initialpivots` default changes from `[zeros(Int, n)]` to `[ones(Int, n)]`. Before passing to C API, subtract 1 from each pivot.
- `evaluate(ttn, indices)`: subtract 1 before passing to C API
- `evaluate(ttn, batch::Matrix)`: subtract 1 from all indices
- Callback wrapper: when Rust passes 0-indexed batch to Julia callback, add 1 before calling user function

- [ ] **Step 2: Write tests**

Create `test/test_treetci.jl` with basic crossinterpolate2 test using 1-indexed pivots and evaluation.

- [ ] **Step 3: Commit**

```bash
git add src/TreeTCI.jl test/test_treetci.jl test/runtests.jl
git commit -m "feat(TreeTCI): convert to 1-indexed user API"
```

---

## Task 9: SimpleTT ↔ TreeTN.MPS conversion

**Files:**
- Modify: `src/TreeTN.jl` — add `MPS(tt::SimpleTensorTrain)`
- Modify: `src/SimpleTT.jl` — add `SimpleTensorTrain(mps::TreeTensorNetwork{Int})`

- [ ] **Step 1: Implement MPS(tt::SimpleTensorTrain)**

In `src/TreeTN.jl`:
```julia
function MPS(tt::SimpleTensorTrain{T}) where T
    n = length(tt)
    tensors = Tensor[]
    links = Index[]

    # Create site and link indices
    for i in 1:n
        st = sitetensor(tt, i)  # 1-indexed, shape (left, site, right)
        left_dim, site_dim, right_dim = size(st)

        site_idx = Index(site_dim)
        inds = Index[]

        if i > 1
            push!(inds, links[end])  # left link
        end
        push!(inds, site_idx)
        if i < n
            link = Index(right_dim; tags="Link,l=$i")
            push!(links, link)
            push!(inds, link)
        end

        push!(tensors, Tensor(inds, st))
    end

    return MPS(tensors)
end
```

- [ ] **Step 2: Implement SimpleTensorTrain(mps::TreeTensorNetwork{Int})**

In `src/SimpleTT.jl`, import TreeTN types and add:
```julia
function SimpleTensorTrain(mps::TreeTN.TreeTensorNetwork{Int})
    n = TreeTN.nv(mps)
    site_tensors = Array{Float64,3}[]  # or ComplexF64 based on storage

    for i in 1:n
        tensor = mps[i]
        # Extract data and determine (left, site, right) ordering
        # ... (implementation depends on TreeTN tensor index ordering)
    end

    return SimpleTensorTrain(site_tensors)
end
```

- [ ] **Step 3: Write round-trip tests**

Create `test/test_conversions.jl`:
- Create SimpleTT → convert to MPS → convert back → compare values
- Create MPS (via random_mps) → convert to SimpleTT → convert back → compare

- [ ] **Step 4: Commit**

```bash
git add src/TreeTN.jl src/SimpleTT.jl test/test_conversions.jl test/runtests.jl
git commit -m "feat: add SimpleTT ↔ TreeTN.MPS bidirectional conversion"
```

---

## Task 10: TCI conversion bridge extension

**Files:**
- Create: `ext/Tensor4allTCIExt.jl`
- Modify: `Project.toml` — add TensorCrossInterpolation weak dep

- [ ] **Step 1: Update Project.toml**

Add to `[weakdeps]`:
```toml
TensorCrossInterpolation = "<uuid>"
```

Add to `[extensions]`:
```toml
Tensor4allTCIExt = ["TensorCrossInterpolation"]
```

Get UUID from `../TensorCrossInterpolation.jl/Project.toml`.

- [ ] **Step 2: Create extension**

```julia
module Tensor4allTCIExt

using Tensor4all
using TensorCrossInterpolation
import Tensor4all.SimpleTT: SimpleTensorTrain

function SimpleTensorTrain(tt::TensorCrossInterpolation.TensorTrain{T}) where T
    return SimpleTensorTrain(tt.sitetensors)
end

function TensorCrossInterpolation.TensorTrain(stt::SimpleTensorTrain{T}) where T
    n = length(stt)
    site_tensors = [Tensor4all.SimpleTT.sitetensor(stt, i) for i in 1:n]
    return TensorCrossInterpolation.TensorTrain(site_tensors)
end

end # module
```

- [ ] **Step 3: Write tests (if TCI is available in test deps)**

- [ ] **Step 4: Commit**

```bash
git add ext/Tensor4allTCIExt.jl Project.toml
git commit -m "feat: add TCI.TensorTrain ↔ SimpleTensorTrain conversion extension"
```

---

## Implementation Notes

### Rust library must be rebuilt

After pulling the latest tensor4all-rs (with #393), the shared library `libtensor4all_capi.so` must be rebuilt. Run:
```bash
TENSOR4ALL_RS_PATH=/home/shinaoka/tensor4all/tensor4all-rs julia --startup-file=no deps/build.jl
```

### Testing approach

Tests should NOT use ITensors (to verify the "ITensors-free" goal). Use only Tensor4all modules directly. ITensors tests are separate in `itensors_ext_test.jl`.

### Breaking changes

This plan introduces breaking changes to:
- SimpleTT: 0→1 indexed, function renames
- QuanticsTCI: return type (single → tuple), argument order
- TreeTCI: 0→1 indexed
- QuanticsGrids: `local_dimensions` → `localdimensions`

No backward compatibility aliases are provided.
