# Phase 2 BubbleTeaCI Migration Follow-up Plan (Tensor4all.jl)

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task once tensor4all-rs#423 lands.
>
> **Status:** Blocked on `tensor4all/tensor4all-rs#423` (Expose missing backend
> options and add TreeTN restructuring API). Do not begin implementation until
> the upstream issue is merged and the pin in `deps/build.jl` is updated.

**Goal:** After tensor4all-rs#423 merges and ships the new C API surface, land
all Tensor4all.jl wrappers and consumer changes in one coordinated PR series so
that BubbleTeaCI Phase 2 (`rearrange_siteinds`, real Fit-method `apply`, full
`truncate` controls, etc.) becomes available in a single visible step.

**Architecture:** Wrap the new and previously unwrapped C API additions through
the existing `TensorNetworks` / `Core` Julia layers. Implement
`rearrange_siteinds` on top of the new `restructure_to` Rust orchestrator
rather than reproducing the fuse / split / swap pipeline in Julia. Surface the
backend option additions (Fit method controls, truncation `form`,
canonicalization `force`, QR `rtol`) on the existing Julia signatures.

**Tech Stack:** Julia, the existing `Tensor4all.*` modules, Documenter.jl.

---

## Reference checkpoints

- tensor4all/tensor4all-rs#423 — upstream C API expansion this plan consumes.
- `tensor4all-rs:crates/tensor4all-treetn/src/treetn/transform.rs` — `fuse_to`,
  `split_to`, and the `restructure_to` to be added by #423.
- `tensor4all-rs:crates/tensor4all-treetn/src/treetn/mod.rs` —
  `swap_site_indices`.
- Local Quantics.jl reference for `rearrange_siteinds` semantics —
  `../Quantics.jl/src/util.jl:436-556`.
- `Tensor4all.jl:src/TensorNetworks/backend/` — current Julia wrapping pattern
  for C API entries.
- `Tensor4all.jl:docs/src/api.md` — `@autodocs` Pages list to update for every
  new file.
- `Tensor4all.jl:scripts/check_autodocs_coverage.jl` — runs in CI and will
  fail if new files defining public symbols are not added to the Pages list.

---

## Pre-flight (do once, before any task below)

1. Confirm tensor4all-rs#423 is merged and a release / commit hash is available.
2. Update `deps/build.jl::TENSOR4ALL_RS_FALLBACK_COMMIT` to that hash.
3. Update the matching commit in `.github/workflows/CI.yml` and
   `.github/workflows/docs.yml` (the hard-coded `git checkout` lines).
4. Run `Pkg.build("Tensor4all")` locally to confirm the new C symbols load.

---

## Task 1: Wrap previously-existing-but-unwrapped C API

These C API entries already exist on `5082aca` and need no Rust change.
Wrapping them is independent of #423 in principle, but landing them in the
same wave keeps the PR series coherent.

**Files (create):**
- `src/TensorNetworks/backend/treetn_dense.jl` — `Array(tt::TensorTrain)` /
  `to_dense(tt)` via `t4a_treetn_to_dense`.
- `src/TensorNetworks/backend/treetn_contract.jl` —
  `TensorNetworks.contract(a::TensorTrain, b::TensorTrain; method, rtol,
  cutoff, maxdim)` via `t4a_treetn_contract`.
- `src/TensorNetworks/backend/treetn_evaluate.jl` —
  `evaluate(tt, indices, values)` via `t4a_treetn_evaluate`.

**Files (modify):**
- `src/TensorNetworks/backend/treetn_queries.jl` — replace the Julia-side
  `siteinds` / `linkinds` derivation with calls to `t4a_treetn_siteinds`,
  `t4a_treetn_linkind`, `t4a_treetn_neighbors`. Confirm round-trip parity
  against the current Julia derivation before deleting the Julia path.
- `src/TensorNetworks/backend/treetn.jl` — wrap `t4a_treetn_set_tensor` (used
  by `setindex!` to keep the Rust-side handle in sync, if/when we add
  long-lived backend handles).
- `src/QuanticsTransform/operators.jl` — wrap
  `t4a_qtransform_binaryop_materialize` for `binaryop_operator`. Replace the
  current metadata-only placeholder.
- `docs/src/api.md` — add the new files to the corresponding `@autodocs`
  `Pages = [...]` lists. Confirm `scripts/check_autodocs_coverage.jl` passes.

**Tests:** new test files mirror the file split. Reuse the existing
`test/tensornetworks/arithmetic.jl` patterns for fixtures.

---

## Task 2: Surface A1–A5 option additions on existing Julia signatures

Each backend option exposed by tensor4all-rs#423 should appear as a keyword
argument on the corresponding Julia function. Default values must match the
backend default so existing call sites keep their behavior.

**A1 — `apply` (`src/TensorNetworks/backend/apply.jl`):**
- Add keyword args `nfullsweeps::Integer = 1` and `convergence_tol::Real = 0.0`
  (sentinel `0.0` = unset).
- Validate: `nfullsweeps >= 1`, `convergence_tol >= 0`.
- Update docstring to enumerate `method ∈ {:zipup, :fit, :naive}` and explain
  every option (including the existing `rtol` / `cutoff` / `maxdim` sentinel
  meaning that is currently undocumented).
- Tests: at least one `method=:fit` case with `nfullsweeps >= 2` that
  visibly converges better than `nfullsweeps = 1`.

**A2 — `truncate` (`src/TensorNetworks/backend/treetn.jl`):**
- Add `form::Symbol = :unitary`. Map `:unitary | :lu | :ci` to
  `t4a_canonical_form` (reuse `_canonical_form_code`).
- Update docstring (same gap as `apply`).
- Tests: cover at least `form=:lu`.

**A3 — `orthogonalize` (`src/TensorNetworks/backend/treetn.jl`):**
- Add `force::Bool = true` (preserves existing behavior).
- Tests: `force=false` on already-canonical input is observably faster or at
  least correct.

**A4 — `TensorNetworks.contract` (introduced in Task 1):**
- Mirror A1's options (`nfullsweeps`, `convergence_tol`, plus
  `factorize_alg::Symbol = :svd` mapped to a new `t4a_factorize_alg` enum).

**A5 — `Tensor4all.qr` (`src/Core/Tensor.jl`):**
- Add `rtol::Real = 0.0` (sentinel = no truncation).
- Tests: rank-revealing case where truncation actually reduces the output rank.

**Docstrings:** every modified function gets a complete docstring covering all
keyword arguments, sentinel meanings (`0.0` / `0` = "not set"), and the
existing semantics that are currently missing (return type, reference
formula for `dist`, `Vector{Index}` shape for `siteinds`, etc.). This is
covered separately by tensor4all/Tensor4all.jl#TODO (file a docs-only issue
once this plan starts).

---

## Task 3: Wrap `restructure_to` and the three TreeTN restructuring primitives

**File (create):** `src/TensorNetworks/backend/treetn_restructure.jl`

Wrap each of:
- `t4a_treetn_restructure_to`
- `t4a_treetn_fuse_to`
- `t4a_treetn_split_to`
- `t4a_treetn_swap_site_indices`

Public Julia names exposed at the `TensorNetworks` module boundary:
- `restructure_to(tt, target; ...)`
- `fuse_to(tt, target)`
- `split_to(tt, target; ...)`
- `swap_site_indices(tt, target_assignment; ...)`

The Julia-side wrapper for the target `SiteIndexNetwork` should accept a
plain `Vector{Vector{Index}}` (one inner vector per target node, in target
node order) and convert it to the C representation chosen by #423.

**Tests:** new file `test/tensornetworks/restructure.jl` covering fuse-only,
split-only, swap-only, and the orchestrator on a mixed case.

---

## Task 4: Implement `rearrange_siteinds` on top of `restructure_to`

**File (modify):** `src/TensorNetworks/site_helpers.jl` (or split out if it
crosses the soft 250–300 line cap from `AGENTS.md`).

Replace the current "not yet implemented" stub for
`TensorNetworks.rearrange_siteinds`:

```julia
function rearrange_siteinds(tt::TensorTrain, target_groups::Vector{Vector{Index}})
    target_net = _build_target_site_network(target_groups)
    return restructure_to(tt, target_net)
end
```

Tests should mirror the Quantics.jl reference suite for `rearrange_siteinds`,
covering: fused → interleaved, interleaved → fused, partial reorder.

Update `docs/src/api.md` to remove the "not yet implemented" annotation.

---

## Task 5: Wrap `t4a_treetn_linsolve` (B1 from tensor4all-rs#423)

**File (create):** `src/TensorNetworks/backend/linsolve.jl`

Public Julia name and signature to be confirmed once the upstream C API
shape is finalized in tensor4all-rs#423. Likely
`linsolve(A::LinearOperator, b::TensorTrain; tol, maxiter, ...) -> TensorTrain`.

This task may slip to a separate PR if linsolve in #423 is split out from the
main A1–B2 wave.

---

## Task 6: Documentation refresh

After Tasks 1–5:
- Update `docs/src/api.md` for every new file (`scripts/check_autodocs_coverage.jl`
  must pass).
- Update `docs/src/index.md` and `docs/src/modules.md` to reflect the new
  surface (`restructure_to`, `linsolve`, `rearrange_siteinds`).
- Run `julia --project=docs docs/make.jl` and confirm clean build.

---

## Acceptance criteria

- [ ] tensor4all-rs#423 merged and pin updated.
- [ ] Task 1 — previously-existing C API wrapped, tests pass.
- [ ] Task 2 — A1–A5 surfaced as Julia keyword args, tests pass.
- [ ] Task 3 — `restructure_to` + 3 primitives wrapped, tests pass.
- [ ] Task 4 — `rearrange_siteinds` works against the Quantics.jl reference
  suite.
- [ ] Task 5 — `linsolve` wrapper landed (or scoped out into follow-up).
- [ ] Task 6 — `docs/make.jl` clean, `scripts/check_autodocs_coverage.jl`
  green.
- [ ] BubbleTeaCI `scripts/examples/basic_operations.jl` runs end-to-end on
  the new backend (workspace-level milestone from `../AGENTS.md`).
