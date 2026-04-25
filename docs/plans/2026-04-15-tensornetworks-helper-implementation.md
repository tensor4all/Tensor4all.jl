# TensorNetworks Helper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the remaining `TensorNetworks` `SkeletonNotImplemented` surface with pure Julia helper implementations and a correctness-first `LinearOperator` application path, then port the decisive `tensor4all-rs` / `Quantics.jl` tests that should lock behavior.

**Architecture:** Keep `TensorNetworks.TensorTrain` as the indexed chain type and implement the remaining helper surface in Julia on top of `Tensor` / `Index` metadata plus dense array operations. Pure metadata helpers should not call the C API. Numeric chain helpers should use `reshape`, `permutedims`, and `LinearAlgebra` factorization, not new C-API entry points. Use `Quantics.jl` as the primary semantic reference for chain helper behavior and `ITensorMPS.jl` as the natural Julia API reference for `findsite`, `findsites`, `replace_siteinds!`, and `apply`. If `src/TensorNetworks.jl` stops being comfortably reviewable, split it into `src/TensorNetworks/` with a soft cap of roughly 250-350 lines per file.

**Tech Stack:** Julia, `LinearAlgebra`, existing `Tensor4all.Index` / `Tensor4all.Tensor` / `Tensor4all.SimpleTT`, Documenter.jl, reference checkouts `../Quantics.jl`, `../ITensorMPS.jl`, and `../tensor4all-rs`

---

## Reference checkpoints

- `../ITensorMPS.jl/src/mps.jl:519-527`:
  `replace_siteinds!` / `replace_siteinds` shape and mutating-vs-copy contract.
- `../ITensorMPS.jl/src/abstractmps.jl:538-595`:
  `findsite` / `findsites` public semantics.
- `../ITensorMPS.jl/src/abstractmps.jl:653-680` and `../ITensorMPS.jl/src/mpo.jl:254-283`:
  best available analogs for space/index ownership, but not a direct `set_*space!` equivalent.
- `../ITensorMPS.jl/test/base/test_mps.jl:1218-1251`:
  decisive `findsite` / `findsites` expectations for MPS and MPO.
- `../ITensorMPS.jl/src/mpo.jl:598-614`:
  natural Julia-facing `apply(A, ψ)` contract for MPO-on-MPS application.
- `../Quantics.jl/src/tag.jl:14-63`:
  tag-based site lookup semantics.
- `../Quantics.jl/src/util.jl:263-327`:
  `matchsiteinds`.
- `../Quantics.jl/src/util.jl:436-556`:
  `replace_siteinds_part!`, `rearrange_siteinds`, `makesitediagonal`, `extractdiagonal`.
- `../Quantics.jl/test/util_tests.jl:89-301`:
  minimal numerical checks worth porting.
- `../tensor4all-rs/crates/tensor4all-quanticstransform/tests/integration_test.rs:367-2822`:
  dense-matrix and basis-state operator tests worth translating to Julia.
- `../tensor4all-rs/crates/tensor4all-quanticstransform/src/cumsum/tests/mod.rs:1-70`:
  cumsum structure and validation checks.
- `../tensor4all-rs/crates/tensor4all-quanticstransform/src/fourier/tests/mod.rs:1-111`:
  fourier structure and validation checks.
- `../tensor4all-rs/crates/tensor4all-quanticstransform/src/affine/tests/mod.rs:612-1038`:
  affine correctness cases that should become Julia tests once `apply` is real.

### Scope decisions

- Do not add HDF5 C-API usage back. The pure Julia HDF5 extension stays as-is.
- Do not add new C-API entry points for helper behavior that can be expressed with Julia tensor metadata and dense array algebra.
- Every newly exported type and function touched by this plan needs a concise docstring.
- If implementing these tasks pushes a single submodule file past the soft cap above, create a subdirectory and split by responsibility before adding more logic.
- Tensor4all does not assign semantic meaning to prime level for operator I/O.
- Therefore the public `set_input_space!`, `set_output_space!`, and `set_iospaces!`
  APIs should accept explicit `Vector{Index}` arguments only, not `TensorTrain`.
- If an operator is built from an MPO-like chain, the constructor or helper that
  wraps that chain must receive explicit `input_indices` and `output_indices`.

### Task 1: Lock the metadata helper semantics with focused tests

**Files:**
- Create: `test/tensornetworks/index_queries.jl`
- Modify: `test/runtests.jl`

**Step 1: Write the failing test**

Create two small fixtures:

- one MPS-like `TensorNetworks.TensorTrain` with one site index per tensor
- one MPO-like `TensorNetworks.TensorTrain` with one unprimed and one primed site index per tensor

Add checks modeled on `ITensorMPS.jl` and `Quantics.jl`:

```julia
@testset "TensorNetworks index queries" begin
    TN = Tensor4all.TensorNetworks
    ψ, sites, links = mps_fixture()
    M, mpo_sites, mpo_links = mpo_fixture()

    @test TN.findsite(ψ, sites[2]) == 2
    @test TN.findsites(ψ, links[1]) == [1, 2]
    @test TN.findallsites_by_tag(ψ; tag="x") == [1, 2, 3]
    @test TN.findallsiteinds_by_tag(ψ; tag="x") == sites

    @test TN.findsite(M, mpo_sites[2][1]) == 2
    @test TN.findsite(M, mpo_sites[2][2]) == 2
    @test TN.findsites(M, (mpo_sites[2][1], mpo_sites[3][2])) == [2, 3]

    replaced = TN.replace_siteinds(ψ, sites, sim.(sites))
    @test Tensor4all.inds(replaced[1])[1] == sim(sites[1])
    @test Tensor4all.inds(ψ[1])[1] == sites[1]

    ψcopy = deepcopy_tt(ψ)
    TN.replace_siteinds!(ψcopy, sites, sim.(sites))
    @test Tensor4all.inds(ψcopy[2])[1] == sim(sites[2])
end
```

Also add argument-error cases:

- invalid tag containing `=`
- replacement length mismatch
- lookup target not found

**Step 2: Run test to verify it fails**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: FAIL because the current methods still throw `SkeletonNotImplemented`.

**Step 3: Write minimal implementation**

No implementation in this task.

**Step 4: Run test to confirm the failure is the intended one**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: FAIL only on the new helper expectations.

**Step 5: Commit**

```bash
git add test/tensornetworks/index_queries.jl test/runtests.jl
git commit -m "test: lock tensornetworks metadata helper semantics"
```

### Task 2: Implement structural introspection and metadata helpers

**Files:**
- Modify: `src/TensorNetworks.jl`
- Or split into:
  - Create: `src/TensorNetworks/types.jl`
  - Create: `src/TensorNetworks/index_queries.jl`
  - Create: `src/TensorNetworks/siteops.jl`
  - Modify: `src/TensorNetworks.jl`

**Step 1: Write the failing test**

Use the tests from Task 1.

**Step 2: Run test to verify it fails**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: FAIL on `findsite`, `findsites`, `findallsiteinds_by_tag`, `findallsites_by_tag`, `replace_siteinds!`, `replace_siteinds`, and `replace_siteinds_part!`.

**Step 3: Write minimal implementation**

Implement internal utilities first:

- `_siteinds_per_tensor(tt)`:
  return the physical indices for each tensor, excluding chain links.
- `_linkinds(tt)`:
  infer adjacent shared indices between neighboring tensors.
- `_is_mps_like(tt)` / `_is_mpo_like(tt)`:
  classify tensors by physical-site count per core.
- `_find_tensor_sites(tt, target)`:
  match a single `Index`, tuple of `Index`, or `Tensor`.

Then implement:

- `findsite`
- `findsites`
- `findallsiteinds_by_tag`
- `findallsites_by_tag`
- `replace_siteinds!`
- `replace_siteinds`
- `replace_siteinds_part!`

Required semantics:

- follow the `ITensorMPS.jl` mutating/non-mutating split
- preserve tensor data unless the index metadata itself changes
- validate replacement lengths and not-found cases with actionable messages
- treat tags the way `Quantics.jl/src/tag.jl` does:
  `x=1`, `x=2`, ... search in order, error on duplicates, empty result allowed

While touching exports, add concise docstrings for each exported helper.

**Step 4: Run test to verify it passes**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: PASS for the new metadata-helper tests.

**Step 5: Commit**

```bash
git add src/TensorNetworks.jl src/TensorNetworks test/tensornetworks/index_queries.jl
git commit -m "feat: implement tensornetworks metadata helpers"
```

### Task 3: Port and implement `matchsiteinds`

**Files:**
- Create: `test/tensornetworks/matchsiteinds.jl`
- Modify: `test/runtests.jl`
- Modify: `src/TensorNetworks.jl`
- Or create: `src/TensorNetworks/matchsiteinds.jl`

**Step 1: Write the failing test**

Translate the decisive `Quantics.jl/test/util_tests.jl:89-134` cases:

```julia
@testset "TensorNetworks.matchsiteinds" begin
    ψ_sparse, sparse_sites, full_sites = sparse_mps_fixture()
    ψ_full = Tensor4all.TensorNetworks.matchsiteinds(ψ_sparse, full_sites)

    @test dense_mps_values(ψ_full, full_sites) ==
        dense_mps_values_with_inserted_singletons(ψ_sparse, sparse_sites, full_sites)

    M_sparse, sparse_sites, full_sites = sparse_mpo_fixture()
    M_full = Tensor4all.TensorNetworks.matchsiteinds(M_sparse, full_sites)

    @test dense_mpo_matrix(M_full, full_sites) == dense_mpo_matrix_with_inserted_identity(
        M_sparse,
        sparse_sites,
        full_sites,
    )
end
```

Keep the fixtures tiny, for example 2-3 sites only.

**Step 2: Run test to verify it fails**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: FAIL because `matchsiteinds` still throws.

**Step 3: Write minimal implementation**

Port the `Quantics.jl` algorithm, but rewrite it for `Tensor4all.Tensor`:

- normalize candidate sites with `noprime`
- compute the current site positions in the requested ordering
- reject non-ascending order unless reversing is explicitly handled
- synthesize missing identity or basis tensors for gaps
- rebuild the chain with consistent link dimensions

Prefer tiny internal array helpers over generic tensor-contraction abstractions. Keep the first implementation correctness-first, not optimized.

**Step 4: Run test to verify it passes**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: PASS for the `matchsiteinds` tests.

**Step 5: Commit**

```bash
git add src/TensorNetworks.jl src/TensorNetworks test/tensornetworks/matchsiteinds.jl test/runtests.jl
git commit -m "feat: implement tensornetworks matchsiteinds"
```

### Task 4: Port and implement `rearrange_siteinds`, `makesitediagonal`, and `extractdiagonal`

**Files:**
- Create: `test/tensornetworks/site_transforms.jl`
- Modify: `test/runtests.jl`
- Modify: `src/TensorNetworks.jl`
- Or create:
  - `src/TensorNetworks/dense_chain_ops.jl`
  - `src/TensorNetworks/site_transforms.jl`

**Step 1: Write the failing test**

Translate the small but decisive `Quantics.jl/test/util_tests.jl:177-301` cases:

- rearrange `[[x1], [y1], [x2], [y2]] -> [[x1, y1], [x2, y2]]`
- rearrange back and verify exact roundtrip
- make a diagonal MPO from a tiny MPS and verify only diagonal entries survive
- extract the diagonal back and verify exact recovery

Representative checks:

```julia
@testset "TensorNetworks site transforms" begin
    ψ, sites = xy_fixture()

    fused_sites = [[sites.x[1], sites.y[1]], [sites.x[2], sites.y[2]]]
    ψ_fused = Tensor4all.TensorNetworks.rearrange_siteinds(ψ, fused_sites)
    ψ_back = Tensor4all.TensorNetworks.rearrange_siteinds(ψ_fused, [[s] for s in vcat(sites.x, sites.y)])
    @test dense_tt(ψ) == dense_tt(ψ_back)

    Mdiag = Tensor4all.TensorNetworks.makesitediagonal(ψ, "x")
    @test dense_diagonal_entries_match(ψ, Mdiag, "x")
    @test Tensor4all.TensorNetworks.extractdiagonal(Mdiag, "x") == ψ
end
```

**Step 2: Run test to verify it fails**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: FAIL because these helpers still throw.

**Step 3: Write minimal implementation**

Implement dense internal helpers:

- `_contract_dense_chain_prefix!`
- `_factorize_tensor_to_chain`
- `_tensor_as_diagonal`
- `_extract_dense_diagonal`

Then implement:

- `rearrange_siteinds`
- `makesitediagonal`
- `extractdiagonal`

Recommended approach:

- contract only as much of the chain as needed
- factorize back with `qr` or `svd` from `LinearAlgebra`
- preserve site ordering exactly as requested
- keep dimensions and index IDs stable unless the API explicitly creates primed copies

If this logic pushes the module past the soft cap, split it before continuing.

**Step 4: Run test to verify it passes**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: PASS for the new site-transform tests.

**Step 5: Commit**

```bash
git add src/TensorNetworks.jl src/TensorNetworks test/tensornetworks/site_transforms.jl test/runtests.jl
git commit -m "feat: implement tensornetworks site transforms"
```

### Task 5: Keep `set_*space!` explicit and implement exact `apply`

**Files:**
- Create: `test/tensornetworks/apply.jl`
- Modify: `test/runtests.jl`
- Modify: `src/TensorNetworks.jl`
- Or create: `src/TensorNetworks/operators.jl`

**Step 1: Write the failing test**

Add three layers of tests:

1. space-binding behavior
2. exact application of a hand-written identity or diagonal MPO to a tiny MPS
3. helpful argument errors

Representative tests:

```julia
@testset "TensorNetworks operator binding and apply" begin
    TN = Tensor4all.TensorNetworks
    ψ, sites = basis_state_fixture(0b10)
    op = identity_linear_operator(sites)

    TN.set_iospaces!(op, sites)
    @test op.true_input == Vector{Union{Tensor4all.Index, Nothing}}(sites)
    @test op.true_output == Vector{Union{Tensor4all.Index, Nothing}}(sites)

    ϕ = TN.apply(op, ψ)
    @test dense_tt(ϕ) == dense_tt(ψ)

    bad = mpo_like_fixture_with_wrong_length()
    @test_throws DimensionMismatch TN.apply(op, bad)
end
```

Keep the first implementation scope narrow:

- support MPO-like operator on MPS-like state
- exact or `:naive` algorithm only
- no truncation heuristics yet

**Step 2: Run test to verify it fails**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: FAIL because TT overloads still throw and `apply` is unimplemented.
Expected: FAIL because `apply` is unimplemented and the explicit binding path is not yet fully exercised.

**Step 3: Write minimal implementation**

For the public binding API:

- keep only the explicit index-vector methods:
  - `set_input_space!(op, input_indices::Vector{Index})`
  - `set_output_space!(op, output_indices::Vector{Index})`
  - `set_iospaces!(op, input_indices::Vector{Index}, output_indices::Vector{Index}=input_indices)`
- do not add public `TensorTrain` overloads for these setters
- if wrapping an MPO-like chain as a `LinearOperator`, require explicit
  `input_indices` and `output_indices` at construction time
- document clearly that prime level is not used to infer operator domain/codomain

For `apply`:

- validate operator/state lengths and site compatibility
- require explicit `true_input` and `true_output` binding before execution
- if either side is unset, throw an actionable `ArgumentError` telling the user to call `set_input_space!`, `set_output_space!`, or `set_iospaces!`
- never infer input/output role from prime levels

For `apply`:

- validate operator/state lengths and site compatibility
- contract each MPO/MPS site pair exactly using dense array algebra
- factorize back to an MPS-like chain
- return a `TensorTrain` with user-facing output indices and the same chain limits as the input

Use `ITensorMPS.jl/src/mpo.jl:598-614` as the API reference and the existing `SimpleTT` contraction code as a shape/algebra reference when useful.

Add concise docstrings to the overloads and to `apply`.

**Step 4: Run test to verify it passes**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: PASS for the new operator-binding and exact-apply tests.

**Step 5: Commit**

```bash
git add src/TensorNetworks.jl src/TensorNetworks test/tensornetworks/apply.jl test/runtests.jl
git commit -m "feat: implement tensornetworks linear operator apply"
```

### Task 6: Port the decisive `tensor4all-rs` `QuanticsTransform` tests in phases

**Files:**
- Create: `test/quanticstransform/numerics.jl`
- Modify: `test/runtests.jl`
- Modify: `src/QuanticsTransform.jl`
- Modify: `src/TensorNetworks.jl`

**Step 1: Write the failing test**

Start with the smallest correctness locks from Rust:

- shift all values:
  `integration_test.rs:903`
- flip all values:
  `integration_test.rs:827`
- phase rotation on basis states:
  `integration_test.rs:1513-1667`
- open-boundary shift / flip edge cases:
  `integration_test.rs:1277-1412`
- cumsum and fourier structure/validation:
  `src/cumsum/tests/mod.rs:1-70`
  `src/fourier/tests/mod.rs:1-111`

Only after those pass, add affine cases:

- affine identity / shift / rectangular / light-cone:
  `src/affine/tests/mod.rs:612-686`
- affine unitarity and matrix-match checks:
  `src/affine/tests/mod.rs:694-1038`

Representative test shape:

```julia
@testset "QuanticsTransform numerics" begin
    ψ = basis_state_tt(0b001)

    shift = Tensor4all.QuanticsTransform.shift_operator(3, 1)
    Tensor4all.TensorNetworks.set_iospaces!(shift, ψ)
    @test basis_index_of(Tensor4all.TensorNetworks.apply(shift, ψ)) == 0b010

    flip = Tensor4all.QuanticsTransform.flip_operator(3)
    Tensor4all.TensorNetworks.set_iospaces!(flip, ψ)
    @test basis_index_of(Tensor4all.TensorNetworks.apply(flip, ψ)) == 0b111
end
```

**Step 2: Run test to verify it fails**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: FAIL because the operators are still metadata-only placeholders.

**Step 3: Write minimal implementation**

Implement real operator construction incrementally:

- shift
- flip
- phase rotation
- cumsum
- fourier
- affine
- binaryop

Do not jump to affine first. The natural order is:

1. basis-preserving permutation operators
2. diagonal/phase operators
3. cumulative/triangular operators
4. fourier structure and inverse checks
5. affine and binaryop

Each constructor should return a real `TensorNetworks.LinearOperator` with a concrete MPO and binding metadata, not just a `metadata` placeholder.

**Step 4: Run test to verify it passes**

Run:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

Expected: PASS for the translated Rust test subset now covered by Julia.

**Step 5: Commit**

```bash
git add src/QuanticsTransform.jl src/TensorNetworks.jl test/quanticstransform/numerics.jl test/runtests.jl
git commit -m "feat: implement quantics transform operators with translated tests"
```

### Task 7: Align docs, exported docstrings, and public status notes

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/src/index.md`
- Modify: `docs/src/modules.md`
- Modify: `docs/src/api.md`
- Modify: any touched `src/*.jl` files

**Step 1: Write the failing test**

No new unit test. Verification is docs and package-loading.

**Step 2: Run verification to identify gaps**

Run:

```bash
rg -n "SkeletonNotImplemented|placeholder|review phase|Phase 2 helper surface" README.md AGENTS.md docs/src src
```

Expected: find stale wording that still implies the helper surface is unimplemented after Tasks 2-6.

**Step 3: Write minimal implementation**

Update docs so they say:

- which helper APIs are now implemented
- which operator families are implemented and tested
- what still remains intentionally unimplemented

While touching exported functions and structs, add concise docstrings:

- one-sentence summary
- current implemented behavior
- short example only when deterministic and cheap

Do not write expansive narrative docstrings.

**Step 4: Run verification to verify it passes**

Run:

```bash
julia --startup-file=no --project=docs docs/make.jl
julia --startup-file=no --project=. test/runtests.jl
julia --startup-file=no --project=. -e 'using Tensor4all'
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add README.md AGENTS.md docs/src src
git commit -m "docs: align helper and operator docs with implementation"
```

### Task 8: Final verification and branch hygiene

**Files:**
- Test: `test/runtests.jl`

**Step 1: Run the full package test suite**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: PASS.

**Step 2: Run docs build**

Run:

```bash
julia --startup-file=no --project=docs docs/make.jl
```

Expected: PASS.

**Step 3: Inspect the diff**

Run:

```bash
git status --short
git diff --stat
```

Expected: only the intended helper, operator, test, and docs files are modified.

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: implement remaining tensornetworks helpers and operator path"
```
