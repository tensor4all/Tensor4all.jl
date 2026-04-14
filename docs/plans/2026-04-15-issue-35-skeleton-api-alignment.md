# Issue #35 Skeleton API Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align the restored Julia-facing skeleton with the approved old-design architecture, prioritizing API shape and module ownership over deep implementation completeness.

**Architecture:** Remove the TreeTN-first and internal-Quantics leftovers, make `TensorNetworks` the owner of `TensorTrain` and generic operator/application APIs, turn `QuanticsTransform` into a transform-constructor layer, and narrow `TensorCI` to the intended boundary. Tests should only lock the intended public API and major ownership decisions.

**Tech Stack:** Julia, Documenter.jl, Git worktree workflow, tensor4all-rs C backend (only where already required by existing code)

---

### Task 1: Add minimal failing API-shape tests

**Files:**
- Modify: `test/runtests.jl`
- Create: `test/api/skeleton_alignment.jl`

**Step 1: Write the failing test**

```julia
@testset "Skeleton API alignment" begin
    @test !isdefined(Tensor4all, :TreeTensorNetwork)
    @test !isdefined(Tensor4all, :MPS)
    @test !isdefined(Tensor4all, :MPO)
    @test isdefined(Tensor4all.TensorNetworks, :TensorTrain)
    @test isdefined(Tensor4all.TensorNetworks, :LinearOperator)
    @test isdefined(Tensor4all.TensorNetworks, :apply)
    @test !isdefined(Tensor4all.QuanticsTransform, :LinearOperator)
end
```

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: FAIL because current exports and module ownership still follow the TreeTN-first / old-Quantics skeleton.

**Step 3: Write minimal implementation**

Update exports, includes, and module definitions so the tested ownership and visibility become true.

**Step 4: Run test to verify it passes**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: PASS for the new API-shape test.

**Step 5: Commit**

```bash
git add test/runtests.jl test/api/skeleton_alignment.jl
git commit -m "test: lock skeleton api alignment"
```

### Task 2: Remove TreeTN-first and internal Quantics leftovers from the source tree

**Files:**
- Modify: `src/Tensor4all.jl`
- Delete: `src/TreeTN/TreeTensorNetwork.jl`
- Delete: `src/Quantics/Transforms.jl`
- Delete: `src/Quantics/QuanticsGridsBridge.jl`

**Step 1: Write the failing test**

Extend the API-shape test to assert the removed modules are no longer loaded through top-level exports.

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: FAIL because the files are still included and the exports still exist.

**Step 3: Write minimal implementation**

Remove the obsolete includes/exports and keep only the intended public submodules in the top-level package file.

**Step 4: Run test to verify it passes**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: PASS with no TreeTN-first exports remaining.

**Step 5: Commit**

```bash
git add src/Tensor4all.jl src/TreeTN/TreeTensorNetwork.jl src/Quantics/Transforms.jl src/Quantics/QuanticsGridsBridge.jl
git commit -m "refactor: remove obsolete treetn and quantics leftovers"
```

### Task 3: Move generic operator ownership into TensorNetworks

**Files:**
- Modify: `src/TensorNetworks.jl`
- Modify: `src/QuanticsTransform.jl`
- Test: `test/api/skeleton_alignment.jl`

**Step 1: Write the failing test**

Add checks that `TensorNetworks.LinearOperator` and `TensorNetworks.apply` exist, and that `QuanticsTransform` only exposes transform-constructor placeholders rather than the core operator type.

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: FAIL because `LinearOperator` is currently defined in `QuanticsTransform`.

**Step 3: Write minimal implementation**

Define a lightweight `LinearOperator` skeleton and `apply` stub in `TensorNetworks`; simplify `QuanticsTransform` into a constructor/factory layer that references the generic operator type.

**Step 4: Run test to verify it passes**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: PASS with the new ownership model.

**Step 5: Commit**

```bash
git add src/TensorNetworks.jl src/QuanticsTransform.jl test/api/skeleton_alignment.jl
git commit -m "refactor: move generic operator api into tensornetworks"
```

### Task 4: Narrow TensorCI to the intended boundary

**Files:**
- Modify: `src/TensorCI.jl`
- Test: `test/api/skeleton_alignment.jl`

**Step 1: Write the failing test**

Add checks that `TensorCI` re-exports `TensorCrossInterpolation` public entry points and that `crossinterpolate2` returns `TensorCI2`.

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: FAIL because `crossinterpolate2` currently returns `SimpleTT.TensorTrain`.

**Step 3: Write minimal implementation**

Re-export the public upstream surface needed for now and change `crossinterpolate2` to return the upstream `TensorCI2` object directly; keep only the local conversion entry point to `SimpleTT.TensorTrain`.

**Step 4: Run test to verify it passes**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: PASS for the tightened TensorCI boundary.

**Step 5: Commit**

```bash
git add src/TensorCI.jl test/api/skeleton_alignment.jl
git commit -m "refactor: narrow tensorci to approved boundary"
```

### Task 5: Align docs and AGENTS with the implemented skeleton

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/src/index.md`
- Modify: `docs/src/modules.md`
- Modify: `docs/src/api.md`
- Modify: `docs/design/julia_ffi_tci.md`
- Modify: `docs/design/julia_ffi_quantics.md`

**Step 1: Write the failing test**

No extra unit test. The verification is doc consistency and package loading.

**Step 2: Run verification to identify mismatches**

Run: `rg -n "TreeTensorNetwork|TreeTN|crossinterpolate2 returns|QuanticsTransform\\.LinearOperator|save_as_mps|load_tt" AGENTS.md docs/src docs/design`
Expected: find contradictory old wording that no longer matches the intended skeleton.

**Step 3: Write minimal implementation**

Rewrite only the contradictory sections so design docs, module docs, and AGENTS match the implemented skeleton and current phase boundaries.

**Step 4: Run verification to verify it passes**

Run: `julia --startup-file=no --project=. -e 'using Tensor4all'`
Expected: PASS

Run: `rg -n "TreeTensorNetwork|TreeTN-first|QuanticsTransform\\.LinearOperator" AGENTS.md docs/src docs/design`
Expected: only intentional historical references remain.

**Step 5: Commit**

```bash
git add AGENTS.md docs/src/index.md docs/src/modules.md docs/src/api.md docs/design/julia_ffi_tci.md docs/design/julia_ffi_quantics.md
git commit -m "docs: align design docs with skeleton api"
```

### Task 6: Final targeted verification

**Files:**
- Test: `test/runtests.jl`

**Step 1: Run the focused test suite**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: PASS

**Step 2: Run a package-load smoke test**

Run: `julia --startup-file=no --project=. -e 'using Tensor4all'`
Expected: PASS

**Step 3: Inspect git diff**

Run: `git status --short && git diff --stat`
Expected: only the intended API-alignment changes are present.

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: align skeleton api with approved design"
```
