# Remaining Skeleton Surface Implementation Plan

> Historical note: this plan described the temporary stub-only alignment pass.
> The current branch has moved on to real Julia implementations for the
> `TensorNetworks` helper surface, so treat the details below as superseded
> implementation history rather than current guidance.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the approved Julia-side skeleton surface from issue #35 without adding real backend behavior.

**Architecture:** Keep Phase 1 ownership unchanged, but add the still-missing API names as explicit `TensorNetworks` skeletons so the package surface matches the design documents. Prefer lightweight signatures that validate obvious arguments or throw `SkeletonNotImplemented`, and keep tests focused on API presence and stub behavior rather than real numerics.

**Tech Stack:** Julia, package extensions, `Test`, existing `Tensor4all` skeleton types

---

### Task 1: Lock the missing skeleton surface with failing tests

**Files:**
- Create: `test/tensornetworks/skeleton_surface.jl`
- Modify: `test/runtests.jl`

**Step 1: Write the failing test**

Add tests that assert `TensorNetworks` exports the remaining issue-listed skeleton names:

- `findsite`
- `findsites`
- `findallsiteinds_by_tag`
- `findallsites_by_tag`
- `replace_siteinds!`
- `replace_siteinds`
- `replace_siteinds_part!`
- `rearrange_siteinds`
- `makesitediagonal`
- `extractdiagonal`
- `matchsiteinds`
- `save_as_mps`
- `load_tt`

Also assert that the stubbed methods throw `SkeletonNotImplemented` where appropriate.

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: FAIL because several names do not exist yet.

**Step 3: Write minimal implementation**

Add just enough surface to make the tests pass.

**Step 4: Run test to verify it passes**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: PASS for the new skeleton-surface tests.

**Step 5: Commit**

```bash
git add test/runtests.jl test/tensornetworks/skeleton_surface.jl
git commit -m "test: lock remaining skeleton surface"
```

### Task 2: Add the remaining `TensorNetworks` skeleton APIs

**Files:**
- Modify: `src/TensorNetworks.jl`

**Step 1: Write the failing test**

Use the tests from Task 1.

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: FAIL because the symbols are still missing.

**Step 3: Write minimal implementation**

Add exported skeleton definitions for:

- site search helpers
- tag-based site search helpers
- site-index replacement helpers
- rearrangement / diagonal helpers
- optional `matchsiteinds`
- HDF5 boundary declarations `save_as_mps` / `load_tt`

Keep behavior minimal: simple argument validation where easy, otherwise `SkeletonNotImplemented`.

**Step 4: Run test to verify it passes**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: PASS for the new surface.

**Step 5: Commit**

```bash
git add src/TensorNetworks.jl test/tensornetworks/skeleton_surface.jl
git commit -m "feat: add remaining tensornetworks skeleton apis"
```

### Task 3: Align the HDF5 extension with the declared skeleton boundary

**Files:**
- Modify: `ext/Tensor4allHDF5Ext.jl`
- Test: `test/extensions/hdf5_roundtrip.jl`

**Step 1: Write the failing test**

Add checks that the extension methods attach to `TensorNetworks.save_as_mps` and `TensorNetworks.load_tt` rather than existing only as extension-local names.

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: FAIL because the core declarations do not yet have extension-backed methods.

**Step 3: Write minimal implementation**

Wire the extension methods onto the declared `TensorNetworks` entry points while preserving current HDF5 behavior.

**Step 4: Run test to verify it passes**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: PASS with HDF5 roundtrip still working.

**Step 5: Commit**

```bash
git add ext/Tensor4allHDF5Ext.jl test/extensions/hdf5_roundtrip.jl
git commit -m "refactor: align hdf5 extension with skeleton boundary"
```

### Task 4: Update docs to reflect the completed skeleton surface

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/src/api.md`
- Modify: `docs/src/modules.md`
- Modify: `docs/design/julia_ffi_tensornetworks.md`

**Step 1: Write the failing test**

No new unit test. Use grep verification for missing names in docs.

**Step 2: Run verification to identify gaps**

Run: `rg -n "findsite|replace_siteinds|rearrange_siteinds|makesitediagonal|extractdiagonal|matchsiteinds|save_as_mps|load_tt" AGENTS.md docs/src docs/design`
Expected: missing or incomplete documentation for part of the new skeleton surface.

**Step 3: Write minimal implementation**

Document the names as Phase 2 skeleton APIs owned by `TensorNetworks`, without implying that backend behavior exists yet.

**Step 4: Run verification to verify it passes**

Run: `julia --startup-file=no --project=docs docs/make.jl`
Expected: PASS

**Step 5: Commit**

```bash
git add AGENTS.md docs/src/api.md docs/src/modules.md docs/design/julia_ffi_tensornetworks.md
git commit -m "docs: describe completed skeleton surface"
```

### Task 5: Final verification

**Files:**
- Test: `test/runtests.jl`

**Step 1: Run the focused test suite**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: PASS

**Step 2: Run docs build**

Run: `julia --startup-file=no --project=docs docs/make.jl`
Expected: PASS

**Step 3: Run package-load smoke test**

Run: `julia --startup-file=no --project=. -e 'using Tensor4all'`
Expected: PASS

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: complete approved skeleton surface"
```
