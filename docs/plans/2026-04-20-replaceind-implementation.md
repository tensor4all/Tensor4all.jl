# Replaceind Compatibility Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add ITensors-style `replaceind` / `replaceinds` APIs for `Tensor4all.Tensor` and align the underlying `Index`-collection replacement semantics with the same contract.

**Architecture:** Keep all replacement semantics Julia-owned in `Core`. Centralize replacement normalization and validation in `src/Core/Index.jl`, then have `src/Core/Tensor.jl` build non-mutating and mutating tensor APIs on top of that shared logic so the tensor methods only manage metadata updates and object construction.

**Tech Stack:** Julia, `Test`, existing `Tensor4all.Index` and `Tensor4all.Tensor` core types

---

### Task 1: Lock the desired index-collection replacement contract with failing tests

**Files:**
- Modify: `test/core/index.jl`

**Step 1: Write the failing test**

Add a new `@testset` that covers:

- `replaceind([i, j], i => ip)` returns `[ip, j]`
- `replaceinds([i, j], i => ip, j => jp)` returns `[ip, jp]`
- `replaceinds([i, j]) == [i, j]`
- `replaceinds([i, j], ()) == [i, j]`
- `replaceinds([i, j], [i], [ip]) == [ip, j]`
- `replaceinds([i, j], [missing_index], [ip]) == [i, j]`
- `replaceinds([i, j], [i], [bad_dim_index])` throws `ArgumentError`
- `replaceinds([i, j], i => j, j => ip)` is resolved against the original
  indices, so the result is `[j, ip]` rather than a sequentially cascaded
  rewrite

Use concrete indices such as:

```julia
i = Index(2; tags=["i"])
j = Index(3; tags=["j"])
ip = Index(2; tags=["ip"])
jp = Index(3; tags=["jp"])
bad = Index(5; tags=["bad"])
missing_index = Index(2; tags=["missing"])
```

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. test/core/index.jl`
Expected: FAIL because pair notation, empty replacement no-ops, length-checked
collection replacement, and non-sequential multi-replacement are not all
implemented yet.

**Step 3: Write minimal implementation**

Do not change implementation yet. Use the failing test output to confirm the
contract is locked.

**Step 4: Run test to verify it still fails for the expected reason**

Run: `julia --startup-file=no --project=. test/core/index.jl`
Expected: FAIL with assertion mismatches or missing methods in the new
replacement testset, not unrelated package-load errors.

**Step 5: Commit**

```bash
git add test/core/index.jl
git commit -m "test: lock index replacement compatibility contract"
```

### Task 2: Implement shared index replacement normalization and validation

**Files:**
- Modify: `src/Core/Index.jl`
- Test: `test/core/index.jl`

**Step 1: Write the failing test**

Use the tests from Task 1.

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. test/core/index.jl`
Expected: FAIL because `src/Core/Index.jl` still uses sequential pair
replacement and lacks the new compatibility entry points.

**Step 3: Write minimal implementation**

Add a shared helper layer in `src/Core/Index.jl`:

```julia
function _replaceinds_impl(xs::AbstractVector{Index}, olds::AbstractVector{Index}, news::AbstractVector{Index})
    length(olds) == length(news) || throw(DimensionMismatch(...))
    ys = collect(xs)
    for (pos, current) in pairs(xs)
        match = findfirst(==(current), olds)
        isnothing(match) && continue
        replacement = news[match]
        dim(current) == dim(replacement) || throw(ArgumentError(...))
        ys[pos] = replacement
    end
    return ys
end
```

Then expose these public methods:

```julia
replaceinds(xs::AbstractVector{Index}) = collect(xs)
replaceinds(xs::AbstractVector{Index}, replacements::Tuple{}) = collect(xs)
replaceind(xs::AbstractVector{Index}, old::Index, new::Index) = _replaceinds_impl(xs, (old,), (new,))
replaceind(xs::AbstractVector{Index}, replacement::Pair{Index,Index}) = replaceind(xs, first(replacement), last(replacement))
replaceinds(xs::AbstractVector{Index}, replacements::Pair{Index,Index}...) = _replaceinds_impl(xs, first.(replacements), last.(replacements))
replaceinds(xs::AbstractVector{Index}, olds, news) = _replaceinds_impl(xs, collect(olds), collect(news))
```

Keep the dimension check conditional on the old index actually appearing in
`xs`, so missing indices remain no-ops.

**Step 4: Run test to verify it passes**

Run: `julia --startup-file=no --project=. test/core/index.jl`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Core/Index.jl test/core/index.jl
git commit -m "feat: align core index replacement semantics with ITensors"
```

### Task 3: Lock tensor-level replacement APIs with failing tests

**Files:**
- Modify: `test/core/tensor.jl`

**Step 1: Write the failing test**

Add a new `@testset` covering:

- `replaceind(tensor, i, ip)` returns a new tensor with `inds == [ip, j]`
- `replaceind(tensor, i => ip)` also works
- the original tensor remains `[i, j]`
- `replaceinds(tensor, (i, j), (ip, jp))` returns `[ip, jp]`
- `replaceinds(tensor, i => ip, j => jp)` returns `[ip, jp]`
- `replaceind!(tensor, i, ip)` mutates the original tensor indices in place
- `replaceinds!(tensor, (i, j), (ip, jp))` mutates in place
- replacing a missing index is a no-op
- replacing a present index with different `dim` throws `ArgumentError`
- non-mutating methods preserve the dense data contents and `backend_handle`

Construct one tensor with an explicit handle to verify preservation:

```julia
handle = Ptr{Cvoid}(0x1)
tensor = Tensor(data, [i, j]; backend_handle=handle)
```

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. test/core/tensor.jl`
Expected: FAIL because tensor-level replacement methods do not exist yet.

**Step 3: Write minimal implementation**

Do not implement yet. Use the failing output to verify the tests are targeting
missing tensor APIs rather than unrelated tensor behavior.

**Step 4: Run test to verify it still fails for the expected reason**

Run: `julia --startup-file=no --project=. test/core/tensor.jl`
Expected: FAIL with missing method errors or failed assertions in the new
replacement tests.

**Step 5: Commit**

```bash
git add test/core/tensor.jl
git commit -m "test: lock tensor replaceind compatibility"
```

### Task 4: Implement tensor-level replace APIs

**Files:**
- Modify: `src/Core/Tensor.jl`
- Modify: `src/Tensor4all.jl`
- Test: `test/core/tensor.jl`

**Step 1: Write the failing test**

Use the tests from Task 3.

**Step 2: Run test to verify it fails**

Run: `julia --startup-file=no --project=. test/core/tensor.jl`
Expected: FAIL because `Tensor` has no `replaceind` or `replaceinds` methods.

**Step 3: Write minimal implementation**

Add concise docstrings and the following method family to `src/Core/Tensor.jl`:

```julia
function replaceind(t::Tensor, old::Index, new::Index)
    return Tensor(copy(t.data), replaceind(inds(t), old, new); backend_handle=t.backend_handle)
end

replaceind(t::Tensor, replacement::Pair{Index,Index}) = replaceind(t, first(replacement), last(replacement))

function replaceinds(t::Tensor, olds, news)
    return Tensor(copy(t.data), replaceinds(inds(t), olds, news); backend_handle=t.backend_handle)
end

function replaceinds(t::Tensor, replacements::Pair{Index,Index}...)
    return Tensor(copy(t.data), replaceinds(inds(t), replacements...); backend_handle=t.backend_handle)
end

function replaceind!(t::Tensor, old::Index, new::Index)
    t.inds .= replaceind(t.inds, old, new)
    return t
end
```

Also add the matching `Pair` and multi-replacement mutating methods, and export
`replaceind!` / `replaceinds!` from `src/Tensor4all.jl`.

**Step 4: Run test to verify it passes**

Run: `julia --startup-file=no --project=. test/core/tensor.jl`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Core/Tensor.jl src/Tensor4all.jl test/core/tensor.jl
git commit -m "feat: add tensor replaceind compatibility apis"
```

### Task 5: Run focused regression coverage

**Files:**
- Test: `test/core/index.jl`
- Test: `test/core/tensor.jl`

**Step 1: Run the focused tests**

Run: `julia --startup-file=no --project=. test/core/index.jl`
Expected: PASS

**Step 2: Run the tensor tests**

Run: `julia --startup-file=no --project=. test/core/tensor.jl`
Expected: PASS

**Step 3: Run package-level smoke coverage**

Run: `julia --startup-file=no --project=. test/runtests.jl`
Expected: PASS

**Step 4: Commit**

```bash
git add -A
git commit -m "test: verify replaceind compatibility coverage"
```

### Task 6: Verify docs and API load path

**Files:**
- Modify: `docs/src/api.md` only if autodocs coverage needs adjustment
- Test: `docs/make.jl`

**Step 1: Check whether docs wiring needs updates**

Run: `rg -n "Core/Tensor.jl|Core/Index.jl" docs/src/api.md`
Expected: Existing autodocs coverage already includes these files. If not,
update the relevant `Pages = [...]` list.

**Step 2: Build docs**

Run: `julia --startup-file=no --project=docs docs/make.jl`
Expected: PASS

**Step 3: Run package load smoke test**

Run: `julia --startup-file=no --project=. -e 'using Tensor4all; println("ok")'`
Expected: PASS with output `ok`

**Step 4: Commit**

```bash
git add docs/src/api.md
git commit -m "docs: keep replaceind api coverage in sync"
```
