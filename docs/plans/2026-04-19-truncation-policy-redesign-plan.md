# Implementation Plan — Truncation Policy Redesign

## Overview

Breaking redesign of the SVD truncation API:

- **Strategy vs amount separation**: `SvdTruncationPolicy` keeps only the three
  strategy fields (`scale`, `measure`, `rule`). The numeric amount moves to a
  per-call `threshold` kwarg.
- **Legacy surface removed**: `rtol`, `cutoff`, and `form` kwargs are
  deleted from every truncating entry point (previous release deprecated
  `form=:lu`; this release drops `form` entirely).
- **Global + scoped defaults**: add a two-layer registry — a process-wide
  default (lock-guarded) and a task-local scoped override based on
  `ScopedValues`. Users configure once with `set_default_svd_policy!`, or
  temporarily override with `with_svd_policy`.
- **New doc chapter**: add a dedicated "Truncation Policy" section to the
  Documenter site covering the 8 decision-rule combinations, all configuration
  patterns, and the ITensors.jl-compatible preset.

This plan builds on PR #54 and supersedes that PR's public surface. Existing
callers that used `rtol=`/`cutoff=`/`form=` must migrate.

## Scope

Breaking:
- `SvdTruncationPolicy.threshold` field removed; type becomes strategy-only.
- `rtol`, `cutoff`, `form` kwargs removed from `truncate`, `add`, `contract`,
  `apply`, `linsolve`, `split_to`, `restructure_to` (split/final pairs), and
  `Core.svd`.
- `_resolve_svd_policy` internal helper rewritten.

Additive:
- `threshold::Real` kwarg on every truncating function.
- `set_default_svd_policy!(p)`, `default_svd_policy()`, `with_svd_policy(f, p)`
  exported from `TensorNetworks`.
- `ScopedValues.jl` dependency (compat package; 1.10+ supported). Julia 1.11+
  has `Base.ScopedValues` but we import via the compat package for uniformity.
- New docs chapter `docs/src/truncation_policy.md`.

## Task 1 — refactor `SvdTruncationPolicy` into strategy-only

**Affected files:** `src/TensorNetworks/truncation_policy.jl`.

**What to do:**
- Remove the `threshold::Float64` field from `SvdTruncationPolicy`. New shape:
  ```julia
  struct SvdTruncationPolicy
      scale::Symbol
      measure::Symbol
      rule::Symbol
  end
  ```
- Update the keyword constructor defaults (`scale=:relative, measure=:value,
  rule=:per_value`) and the Symbol validation.
- Rename `_to_c_policy(p)` to `_to_c_policy(p, threshold)` — the FFI struct
  still needs `threshold` at the C boundary. The helper carries it as an
  explicit argument.
- Update the docstring: drop the `threshold` field, explain strategy-only
  role, reference `threshold` kwarg on call sites.

**Tests:** update `test/tensornetworks/truncation_policy.jl`:
- `SvdTruncationPolicy()` has only 3 fields.
- `hasfield(SvdTruncationPolicy, :threshold) == false`.

## Task 2 — global + scoped default registry

**Affected files:** `src/TensorNetworks/truncation_policy.jl`,
`src/TensorNetworks/TensorNetworks.jl` (include + export), `Project.toml`
(add dep).

**What to do:**
- Add `ScopedValues` as a dependency in `Project.toml` and `Manifest.toml`.
  Confirm the compat package covers Julia 1.10.
- Introduce two storage slots:
  ```julia
  const _PROCESS_DEFAULT_LOCK = ReentrantLock()
  const _PROCESS_DEFAULT = Ref(SvdTruncationPolicy())
  const _SCOPED_OVERRIDE =
      ScopedValue{Union{Nothing, SvdTruncationPolicy}}(nothing)
  ```
- Implement resolver:
  ```julia
  function default_svd_policy()
      s = _SCOPED_OVERRIDE[]
      s !== nothing && return s
      return lock(_PROCESS_DEFAULT_LOCK) do
          _PROCESS_DEFAULT[]
      end
  end
  ```
- `set_default_svd_policy!(p)` — lock-guarded assignment; returns `nothing`.
- `with_svd_policy(f, p) = @with _SCOPED_OVERRIDE => p f()`.
- Export `SvdTruncationPolicy`, `set_default_svd_policy!`,
  `default_svd_policy`, `with_svd_policy`.

**Tests:** new `test/tensornetworks/truncation_policy_registry.jl` (or extend
existing file):
- `default_svd_policy()` returns the built-in default initially.
- `set_default_svd_policy!` changes the process-wide default.
- `with_svd_policy` scopes an override; reverts on exit.
- Nested `with_svd_policy` restores the outer policy on exit.
- Task isolation: `@sync` block with two `Threads.@spawn` tasks each under
  their own `with_svd_policy`; assert each task sees its own policy.
- Resolution priority: per-call `svd_policy=` > scope > process > built-in.

## Task 3 — rewrite `_resolve_svd_policy`

**Affected files:** `src/TensorNetworks/truncation_policy.jl`.

**What to do:**
- Replace the old resolver with:
  ```julia
  function _resolve_svd_policy(; threshold::Real, svd_policy)
      threshold >= 0 || throw(ArgumentError("threshold must be nonnegative"))
      threshold == 0 && return nothing       # no SVD truncation
      p = svd_policy === nothing ? default_svd_policy() : svd_policy
      return _to_c_policy(p, Float64(threshold))
  end
  ```
- No ambiguity check needed (no `rtol`/`cutoff` any more).
- Callers pass `threshold` and optionally `svd_policy` (defaulting to
  `nothing` meaning "fall back to default").

**Tests:** update `_resolve_svd_policy` tests in
`test/tensornetworks/truncation_policy.jl`:
- `_resolve_svd_policy(threshold=0, svd_policy=nothing)` returns `nothing`.
- Non-zero threshold with no policy uses `default_svd_policy()`.
- Explicit `svd_policy` overrides the default.
- Negative threshold → `ArgumentError`.
- Scoped override via `with_svd_policy` is observed by the resolver.

## Tasks 4–11 — migrate each truncating function

For each of:

- Task 4: `Core.svd` (`src/Core/Tensor.jl`)
- Task 5: `truncate` (`src/TensorNetworks/backend/treetn.jl`)
- Task 6: `add` (same file)
- Task 7: `contract` (`src/TensorNetworks/backend/treetn_contract.jl`)
- Task 8: `apply` (`src/TensorNetworks/backend/apply.jl`)
- Task 9: `linsolve` (`src/TensorNetworks/backend/linsolve.jl`)
- Task 10: `split_to` (`src/TensorNetworks/backend/restructure/split_to.jl`)
- Task 11: `restructure_to` (same dir, `restructure_to.jl`)

**Common pattern — what to do:**

- Remove kwargs: `rtol`, `cutoff`, `form`. Delete their validation lines.
- Add kwarg: `threshold::Real = 0.0`.
- Keep kwarg: `maxdim::Integer = 0`.
- Keep kwarg: `svd_policy::Union{Nothing, SvdTruncationPolicy} = nothing`.
- For `restructure_to`: the split/final pairs become
  `split_threshold` / `split_svd_policy`, `final_threshold` /
  `final_svd_policy`. No more `split_rtol`, `split_cutoff`, `final_rtol`,
  `final_cutoff`, `split_form`, `final_form`.
- Call the new `_resolve_svd_policy(; threshold, svd_policy)`.
- The `_canonical_form_code(:unitary)` call sites that fed the old `form`
  ccall argument go away alongside the `form` parameter.
- `truncate` requires at least one of `threshold > 0` or `maxdim > 0`; reject
  otherwise with `ArgumentError` (already done; just update the message).
- Update the docstring: list the new kwargs, remove `rtol`/`cutoff`/`form`,
  add a one-line cross-reference to the "Truncation Policy" doc chapter.

**Tests:** each of the 8 function test files needs kwarg renames:
- `rtol=ε` → `threshold=ε`.
- `cutoff=ε²` → `threshold=ε` (apply sqrt at the call site).
- Delete any `form=:lu` → `ArgumentError` case (no more `form`).
- Keep the convenience-vs-policy equivalence test in `tensor_factorize.jl`
  but rephrase: `svd(...; threshold=1e-6)` using the default policy equals
  `svd(...; threshold=1e-6, svd_policy=SvdTruncationPolicy())`.
- Keep the `discarded_tail_sum` / `squared_value` reachability tests.

## Task 12 — update `_canonical_form_code` usage

**Affected files:** `src/TensorNetworks/backend/treetn.jl`.

`_canonical_form_code` still exists for `orthogonalize`, which retains the
`form::Symbol` kwarg (`t4a_treetn_orthogonalize` is unchanged by this
redesign). Keep the helper. Only its callers in `truncate`, `linsolve`, and
`split_to` disappear.

**Tests:** `orthogonalize` tests unchanged.

## Task 13 — new docs chapter "Truncation Policy"

**Affected files:** new `docs/src/truncation_policy.md`, updates to
`docs/make.jl` (navigation), `docs/src/api.md` (cross-references), and
removal/merge of the existing `docs/src/truncation.md` (Truncation Contract
page).

**What to include in `docs/src/truncation_policy.md`:**

1. **Overview** — 2 paragraphs: what SVD truncation is in this library, what
   the policy object means, and the "threshold is per-call, strategy is
   policy-level" split.
2. **Quick start** — 5-line example using defaults:
   ```julia
   truncate(tt; threshold=1e-8)
   truncate(tt; threshold=1e-8, maxdim=32)
   ```
3. **The `SvdTruncationPolicy` type** — reference its three fields with the
   allowed Symbol values and the default.
4. **Decision rules (8 combinations table)** — reproduce the formula table
   from the design discussion (relative/absolute × value/squared × per-value
   /tail-sum).
5. **Setting the default** — three sections, matching the pseudo-code
   patterns:
   - **Process-wide default**: `set_default_svd_policy!(...)` called once at
     program / `__init__` time. Include the **ITensors.jl-compatible**
     preset:
     ```julia
     set_default_svd_policy!(SvdTruncationPolicy(
         measure = :squared_value,
         rule    = :discarded_tail_sum,
     ))
     ```
     and a note that `threshold=ε` then matches `ITensors.truncate!(...; cutoff=ε)`.
   - **Scoped override**: `with_svd_policy(policy) do ... end`. Show nested
     and parallel-task examples. Note that `ScopedValues` makes child tasks
     inherit the enclosing policy while remaining isolated from siblings.
   - **Per-call override**: pass `svd_policy=` on any truncating entry point
     when you need a one-off.
6. **Resolution priority diagram**:
   ```
   svd_policy kwarg   (explicit per-call)
        ↑ overrides
   with_svd_policy    (task-local scope)
        ↑ overrides
   set_default_svd_policy!  (process-wide)
        ↑ fallback
   built-in (:relative, :value, :per_value)
   ```
7. **Functions accepting the contract** — list of the 8 entry points plus
   their `threshold` / `svd_policy` / `maxdim` signatures.
8. **Preset recipes** — short subsection:
   - ITensors.jl compat (squared value + discarded tail sum)
   - Absolute Frobenius² error bound (absolute + squared + tail sum)
   - "Classical rtol" (default)
9. **Thread-safety notes** — one short paragraph: `with_svd_policy` is
   task-local via `ScopedValues`; child tasks inherit. `set_default_svd_policy!`
   is lock-guarded for write/read safety. Sibling tasks with their own
   `with_svd_policy` are independent.

**Doc site plumbing:**
- Add the new page to `docs/make.jl` pages list between the Truncation
  Contract (now removed) and the API Reference.
- Delete `docs/src/truncation.md` or keep as a stub that redirects to the
  new page; prefer deletion to avoid drift.
- Update `docs/src/api.md` to cross-reference the new chapter on each
  truncating function's docstring summary. The `@autodocs` block already
  picks up the new registry functions once they are added to
  `src/TensorNetworks/truncation_policy.jl`.

## Task 14 — export and API coverage

**Affected files:** `src/TensorNetworks.jl`, `docs/src/api.md`,
`scripts/check_autodocs_coverage.jl` run.

**What to do:**
- `export SvdTruncationPolicy, set_default_svd_policy!, default_svd_policy, with_svd_policy`.
- Confirm `src/TensorNetworks/truncation_policy.jl` is already in the
  `@autodocs Pages = [...]` list for the TensorNetworks section (added in
  PR #54).
- Add a bullet in `docs/src/api.md`'s TensorNetworks prose list for the
  three new registry functions.
- Run `julia --startup-file=no --project=docs scripts/check_autodocs_coverage.jl`;
  expect green.

## Task 15 — migrate existing tests and fixtures

**Affected files:** every test file that uses the old kwargs. Grep the tree
for `rtol=` / `cutoff=` / `form=:lu`.

**What to do:**
- Replace `rtol=ε` with `threshold=ε`.
- Replace `cutoff=ε²` with `threshold=ε` (apply sqrt literally).
- Delete `form=:lu` error test cases.
- Update any internal helpers that threaded `rtol`/`cutoff` through a chain
  (e.g. `_validate_truncation_kwargs` signature in
  `restructure/helpers.jl`).
- Preserve test coverage: every prior test of `rtol` or `cutoff` now tests
  `threshold` with the appropriate default policy.

## Task 16 — run full suite + docs build

**What to do:**
- `T4A_SKIP_HDF5_TESTS=1 julia --startup-file=no --project=. test/runtests.jl` — expect all green.
- `julia --startup-file=no --project=docs scripts/check_autodocs_coverage.jl` — expect green.
- `julia --project=docs docs/make.jl` — expect clean build, new chapter
  appears in the navigation.

## Task ordering

```
Task 1  (strategy-only type)                    [foundation]
Task 2  (registry + ScopedValue)                [depends on 1]
Task 3  (resolver rewrite)                      [depends on 1, 2]

Tasks 4-11 (per-function migration)             [depend on 1-3; can parallel]

Task 12 (canonical form code cleanup)           [after 5, 9, 10]

Task 13 (new docs chapter)                      [after 2 exports exist;
                                                 can parallel with 4-11]
Task 14 (api.md + exports)                      [after 2]

Task 15 (test migration)                        [after 1-12; can parallel
                                                 with 13/14]

Task 16 (final verification)                    [last]
```

Single PR. Because this is a breaking redesign of an API that just landed
in PR #54, splitting across PRs would leave the codebase in an awkward
mid-migration state.

## Test strategy

1. **Existing tests stay the oracle**: after kwarg renames, every prior test
   continues to exercise the code path it did before. A drop in test count
   indicates accidental coverage loss.
2. **Registry tests** (Task 2) are the new first-class tests: default
   resolution, `set_default_svd_policy!`, `with_svd_policy` nesting, task
   isolation, per-call override priority.
3. **FFI struct layout** test (carried over from PR #54) stays green —
   `_SvdTruncationPolicyC` is unchanged.
4. **Docs build** is part of verification: the new chapter must render and
   every code example in it must be valid Julia (use `jldoctest` blocks
   where practical).
5. **Thread-safety smoke test**: a small `@sync`/`@spawn` test with
   divergent `with_svd_policy` scopes confirms isolation.

## Decisions

Backward compatibility with the PR #54 API is **not** required. All resolved:

- `default_svd_policy` (noun form) as the getter name.
- `docs/src/truncation.md` deleted outright; no redirect stub.
- `ScopedValues.jl` compat package added unconditionally (uniform across 1.10+).
- No in-repo migration guide and no special release-note migration wording.
