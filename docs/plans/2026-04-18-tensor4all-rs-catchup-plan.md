# Implementation Plan — catch up to tensor4all-rs `fd7180c`

## Overview

Catch up `Tensor4all.jl` to `tensor4all-rs` `origin/main` (`fd7180c`) by

- migrating every SVD-based C entry point to the new `t4a_svd_truncation_policy` struct (PR #429 / commit `bb2e04c`),
- renumbering the `t4a_qtt_layout_kind` enum and removing the `Grouped` variant (PR #430 / commit `ae14107`),
- re-implementing `binaryop_operator` / `binaryop_operator_multivar` on top of `affine_pullback_operator` + tensor SVD, since the `t4a_qtransform_binaryop_materialize` C symbol is gone.

**Hard constraints:**

1. The existing Julia-facing public API (`rtol`, `cutoff`, `maxdim`, `form`, `binaryop_operator`, `binaryop_operator_multivar`, re-exports, etc.) must keep its current keyword surface. User-visible kwarg names, defaults, and numerical semantics stay the same.
2. The full Rust SVD truncation strategy (threshold + scale + measure + rule) must be reachable from Julia — not just the `rtol` / `cutoff` convenience subset. This is done by introducing a public `SvdTruncationPolicy` Julia type and a new `svd_policy` kwarg that bypasses the convenience conversion.

## Scope

Two upstream PRs between the current pin (`0d7b954`, = `tensor4all-rs #427`) and `origin/main`:

1. **`bb2e04c` (#429)** — redesign of SVD truncation policy.
   - Introduces `t4a_svd_truncation_policy` struct (`threshold`, `scale`, `measure`, `rule`) plus three enums: `t4a_threshold_scale`, `t4a_singular_value_measure`, `t4a_truncation_rule`.
   - Every SVD-based C function gains a `const t4a_svd_truncation_policy *policy` parameter (passing `NULL` disables truncation). Some also gain `maxdim` as an explicit parameter.
   - Functions affected: `t4a_tensor_svd`, `t4a_treetn_add`, `t4a_treetn_truncate`, `t4a_treetn_contract`, `t4a_treetn_apply_operator_chain`, `t4a_treetn_linsolve`, `t4a_treetn_split_to`, `t4a_treetn_restructure_to`.
   - `t4a_treetn_truncate` **loses** its `form` parameter — it is now SVD-based only.
   - `t4a_treetn_linsolve` **loses** its `form` parameter.
   - `t4a_treetn_split_to` **loses** its `form` parameter.
   - `t4a_treetn_contract` **gains** `qr_rtol` as an explicit parameter (already exposed as `factorize_alg = :qr` on the Julia side).

2. **`ae14107` (#430)** — legacy cleanup.
   - `T4A_QTT_LAYOUT_KIND_GROUPED` removed.
   - Enum renumbered: `INTERLEAVED` 1→0, `FUSED` 2→1.
   - `t4a_qtransform_binaryop_materialize` removed.

## Task 1 — public `SvdTruncationPolicy` type and the FFI helper layer

**Affected files:** `src/TensorNetworks/backend/capi.jl` (internal enums, FFI struct, helpers); new file `src/TensorNetworks/truncation_policy.jl` (public Julia type and convenience constructor); `src/TensorNetworks/TensorNetworks.jl` (`include` + `export`).

**What to do:**

### 1a. Internal C mirrors (`backend/capi.jl`)

- Add Julia mirrors of the new C enums:
  ```
  const _T4A_THRESHOLD_SCALE_RELATIVE              = Cint(0)
  const _T4A_THRESHOLD_SCALE_ABSOLUTE              = Cint(1)
  const _T4A_SINGULAR_VALUE_MEASURE_VALUE          = Cint(0)
  const _T4A_SINGULAR_VALUE_MEASURE_SQUARED_VALUE  = Cint(1)
  const _T4A_TRUNCATION_RULE_PER_VALUE             = Cint(0)
  const _T4A_TRUNCATION_RULE_DISCARDED_TAIL_SUM    = Cint(1)
  ```
- Define an immutable FFI-layout struct mirroring the C `t4a_svd_truncation_policy`:
  ```
  struct _SvdTruncationPolicyC
      threshold::Cdouble
      scale::Cint
      measure::Cint
      rule::Cint
  end
  ```
  (Keep this type private — it is purely the ABI shape. The public Julia type lives elsewhere, see 1b.)
- Renumber the QTT layout constants:
  ```
  const _T4A_QTT_LAYOUT_INTERLEAVED = Cint(0)   # was Cint(1)
  const _T4A_QTT_LAYOUT_FUSED       = Cint(1)   # was Cint(2)
  ```
- Add a helper `_with_svd_policy_ptr(policy) do ptr ... end` that yields `C_NULL` for `nothing` and a live `Ref{_SvdTruncationPolicyC}` for a populated policy via `GC.@preserve`. This keeps the per-caller boilerplate small.

### 1b. Public Julia type (`TensorNetworks/truncation_policy.jl`)

- Introduce a public, exported Julia type that mirrors the Rust `SvdTruncationPolicy`:
  ```julia
  """
      SvdTruncationPolicy(; threshold=0.0, scale=:relative,
                            measure=:value, rule=:per_value)

  Full SVD truncation policy, mirroring `tensor4all_core::SvdTruncationPolicy`
  / `t4a_svd_truncation_policy` in the C API.

  # Fields
  - `threshold::Float64` — numeric threshold used with `scale`/`measure`/`rule`.
  - `scale::Symbol` — `:relative` (compare against a singular-value reference scale)
    or `:absolute` (compare directly to the measured quantity).
  - `measure::Symbol` — `:value` (singular values) or `:squared_value`
    (squared singular values).
  - `rule::Symbol` — `:per_value` or `:discarded_tail_sum`.

  Passing this type via the `svd_policy` keyword to any truncating function
  (`truncate`, `add`, `contract`, `apply`, `linsolve`, `split_to`, `tensor_svd`)
  bypasses the `rtol` / `cutoff` convenience conversion and gives direct access
  to the full Rust truncation strategy.
  """
  struct SvdTruncationPolicy
      threshold::Float64
      scale::Symbol
      measure::Symbol
      rule::Symbol
  end
  ```
- Provide a keyword constructor with the defaults above and validate that `scale ∈ (:relative, :absolute)`, `measure ∈ (:value, :squared_value)`, `rule ∈ (:per_value, :discarded_tail_sum)`, `threshold >= 0`, with actionable `ArgumentError` messages naming the bad symbol.
- Provide an internal converter `_to_c_policy(p::SvdTruncationPolicy) -> _SvdTruncationPolicyC` that maps the symbols to the `Cint` enum constants from 1a.

Export `SvdTruncationPolicy` from `TensorNetworks` so `Tensor4all.TensorNetworks.SvdTruncationPolicy` is the canonical public name. No re-export at the top-level `Tensor4all` module unless already standard for other `TensorNetworks` public types.

### 1c. Resolution helper

- Add a resolver that maps the user-visible kwarg set to either `nothing` (no SVD truncation) or an FFI struct:
  ```
  _resolve_svd_policy(; rtol, cutoff, svd_policy) ::
      Union{Nothing, _SvdTruncationPolicyC}
  ```
  - If `svd_policy !== nothing` and (`rtol != 0` or `cutoff != 0`), raise `ArgumentError("Pass either rtol/cutoff or svd_policy, not both")` — ambiguity is rejected at the Julia boundary.
  - If `svd_policy !== nothing`, convert and return the FFI policy.
  - Else if `rtol == 0 && cutoff == 0`, return `nothing` (caller passes `C_NULL`, matches Rust "no truncation").
  - Else build the convenience policy: `cutoff` takes precedence (`threshold = sqrt(cutoff)`, else `threshold = rtol`), with `scale = :relative`, `measure = :value`, `rule = :per_value`. This exactly reproduces the current numerical semantics of `rtol` / `cutoff` end-to-end.

**Tests:** new unit tests under `test/tensornetworks/truncation_policy.jl`:
- Default construction: `SvdTruncationPolicy()` has `threshold == 0.0`, `scale == :relative`, `measure == :value`, `rule == :per_value`.
- Invalid symbols raise `ArgumentError`.
- `_resolve_svd_policy(; rtol=0, cutoff=0, svd_policy=nothing)` returns `nothing`.
- `_resolve_svd_policy(; rtol=0.1, cutoff=0.04, svd_policy=nothing)` returns policy with `threshold ≈ 0.2` (sqrt of cutoff, which takes precedence).
- Ambiguity: passing both `rtol > 0` and a non-nothing `svd_policy` raises `ArgumentError`.
- FFI struct layout: `sizeof(_SvdTruncationPolicyC) == 8 + 4 + 4 + 4` (with padding), and enum `Cint` values match the C header (guard against silent renumbering).

## Task 2 — migrate `t4a_tensor_svd`

**Affected files:** `src/Core/Tensor.jl` (`svd` implementation).

**What to do:**
- Change the `ccall` signature from `(tensor*, left_inds*, n_left, rtol, cutoff, maxdim, out_u, out_s, out_v)` to `(tensor*, left_inds*, n_left, policy*, maxdim, out_u, out_s, out_v)`.
- Add a new kwarg `svd_policy::Union{Nothing, SvdTruncationPolicy}=nothing` alongside the existing `rtol`, `cutoff`, `maxdim`.
- Resolve the effective policy via Task 1's `_resolve_svd_policy` helper and pass `C_NULL` when it returns `nothing`.
- Keep `rtol` / `cutoff` / `maxdim` behavior bit-compatible with the current implementation when `svd_policy === nothing`.

**Tests:** existing `test/core/tensor_factorize.jl` must pass unchanged. Add regression cases:
- `svd(...; rtol=1e-6)` and `svd(...; svd_policy=SvdTruncationPolicy(threshold=1e-6))` produce identical singular values for a small fixture (verifies the convenience path reproduces the full-policy path).
- `svd(...; svd_policy=SvdTruncationPolicy(threshold=1e-4, scale=:absolute, measure=:squared_value, rule=:discarded_tail_sum))` exercises a strategy combination not reachable from `rtol` / `cutoff` alone.

## Task 3 — migrate `t4a_treetn_truncate`

**Affected files:** `src/TensorNetworks/backend/treetn.jl` (`truncate`).

**What to do:**
- Change the `ccall` signature from `(tt*, rtol, cutoff, maxdim, form)` to `(tt*, policy*, maxdim)`.
- Add a new kwarg `svd_policy::Union{Nothing, SvdTruncationPolicy}=nothing`. Resolve via `_resolve_svd_policy`.
- Keep `form::Symbol = :unitary` kwarg on the Julia side so callers do not break.
- For `form === :unitary`, dispatch normally.
- For `form === :lu`, raise an `ArgumentError`: *"form=:lu is no longer supported by the TreeTN truncation kernel. The backend switched to SVD-only truncation in tensor4all-rs #429. Use `form=:unitary` (the default) or open an issue if LU truncation is needed again."* — this keeps the Julia signature stable while surfacing the backend change. See Open Question 1.

**Tests:** `test/tensornetworks/orthogonalize_truncate.jl` must pass unchanged. Add:
- `form=:lu` raises the new `ArgumentError`.
- `truncate(tt; svd_policy=SvdTruncationPolicy(threshold=1e-6, rule=:discarded_tail_sum))` drops bond dimensions differently than `rtol=1e-6` on a deliberately tail-heavy fixture (confirms the full policy path reaches a strategy that the rtol shortcut cannot).

## Task 4 — migrate `t4a_treetn_add`

**Affected files:** `src/TensorNetworks/backend/treetn.jl` (`add`).

**What to do:**
- Change the `ccall` signature from `(a*, b*, rtol, cutoff, maxdim, out*)` to `(a*, b*, policy*, maxdim, out*)`.
- Add `svd_policy` kwarg, resolve via `_resolve_svd_policy`.
- Keep existing `rtol` / `cutoff` / `maxdim` kwargs.

**Tests:** existing `test/tensornetworks/arithmetic.jl` must pass unchanged. Add a case exercising `add(a, b; cutoff=1e-6, maxdim=4)` and `add(a, b; svd_policy=SvdTruncationPolicy(...))` for a fixture where the two paths measurably differ.

## Task 5 — migrate `t4a_treetn_contract`

**Affected files:** `src/TensorNetworks/backend/treetn_contract.jl` (`contract`).

**What to do:**
- Change the `ccall` signature from `(a*, b*, method, rtol, cutoff, maxdim, nfullsweeps, convergence_tol, factorize_alg, out*)` to `(a*, b*, method, policy*, maxdim, nfullsweeps, convergence_tol, factorize_alg, qr_rtol, out*)`.
- Add `svd_policy` kwarg.
- The new `qr_rtol` C parameter is only consumed when `factorize_alg = :qr`. Add a `qr_rtol::Real=0.0` kwarg on the Julia side so users with `factorize_alg=:qr` can control it; the default `0.0` matches the Rust "use default" sentinel.
- Keep all existing Julia kwargs.

**Tests:** existing `test/tensornetworks/contract.jl` must pass unchanged. Add:
- `contract(...; svd_policy=...)` exercises the full-policy path.
- `contract(...; factorize_alg=:qr, qr_rtol=1e-8)` exercises the new qr_rtol knob.

## Task 6 — migrate `t4a_treetn_apply_operator_chain`

**Affected files:** `src/TensorNetworks/backend/apply.jl` (`apply`).

**What to do:**
- Change the `ccall` signature to insert `policy*, maxdim` in place of the old `rtol, cutoff, maxdim` triple.
- Add `svd_policy` kwarg.

**Tests:** existing `test/tensornetworks/apply.jl` must pass unchanged. Add one `svd_policy=...` case.

## Task 7 — migrate `t4a_treetn_linsolve`

**Affected files:** `src/TensorNetworks/backend/linsolve.jl` (`linsolve`).

**What to do:**
- Change the `ccall` signature to insert `policy*, maxdim` in place of the old `rtol, cutoff, maxdim, form` quadruple.
- Add `svd_policy` kwarg.
- Keep `form::Symbol = :unitary` kwarg. `form === :lu` → same `ArgumentError` as Task 3.

**Tests:** existing `test/tensornetworks/linsolve.jl` must pass unchanged. Add `form=:lu` error case and one `svd_policy=...` case.

## Task 8 — migrate `t4a_treetn_split_to` (and via it `restructure_to`)

**Affected files:** `src/TensorNetworks/backend/restructure/split_to.jl`. `restructure_to.jl` only composes `split_to` / `fuse_to` / `swap_site_indices` on the Julia side and does not call the C API directly, so it requires no C-level change except forwarding a new `split_svd_policy` / `final_svd_policy` pair of kwargs.

**What to do:**
- Change the `split_to` `ccall` signature: replace `rtol, cutoff, maxdim, form, final_sweep` with `policy*, maxdim, final_sweep`.
- Add `svd_policy` kwarg to `split_to`.
- For `restructure_to`, add `split_svd_policy::Union{Nothing, SvdTruncationPolicy}=nothing` and `final_svd_policy::Union{Nothing, SvdTruncationPolicy}=nothing` kwargs, forwarded to `split_to` and `truncate` respectively.
- Keep `form` kwarg on `split_to` / `restructure_to`; `form === :lu` → same `ArgumentError`.
- Verify that `fuse_to` does not need any change (its C signature is untouched).
- Verify `restructure_to` still works end-to-end.

**Tests:** existing `test/tensornetworks/restructure.jl` must pass unchanged. Add:
- `form=:lu` error case on `split_to`.
- `restructure_to(...; split_svd_policy=SvdTruncationPolicy(...), final_svd_policy=SvdTruncationPolicy(...))` reaches a bond-dimension outcome not reachable from `split_rtol` / `split_cutoff` / `final_rtol` / `final_cutoff`.

## Task 9 — re-implement `binaryop_operator` (Issue #53, bullet 2)

**Affected files:** `src/QuanticsTransform/operators.jl`, `src/QuanticsTransform/capi_helpers.jl`.

**What to do:**

The C symbol `t4a_qtransform_binaryop_materialize` is gone. The public Julia functions `binaryop_operator` and `binaryop_operator_multivar` must keep their current signatures and semantics.

Re-implement the body in pure Julia (no C-level `_materialize_binaryop`):
1. Build a Fused-layout affine-pullback MPO over two variables by constructing the appropriate `(a_num, a_den, b_num, b_den, m, n, bc)` tuple and calling the existing `_materialize_affine_pullback` helper. The mapping from `binaryop` coefficients `(a1, b1, a2, b2)` and target variables `(lhs_var, rhs_var)` to the affine pullback's `A`-matrix / shift vector is:
    - `m = 2, n = 2` (two output variables from two input variables).
    - Row `i` of `A` is `[a_i, b_i]`, i.e. `a_num[i*n + j]` encodes a dense 2x2 integer matrix. Denominators are all 1.
    - `b_num = b_den = [0, 0]` (no constant shift — binaryop uses the linear part only).
    - `bc = [bc1, bc2]`.
   (Cross-check the exact sign / transpose convention against the old Rust `binaryop_operator` source in `tensor4all-rs` before implementing. The spec for #53 references this mapping; confirm against the deleted Rust function.)
2. The affine pullback is returned in Fused layout. Convert to the Interleaved layout that `binaryop_operator` historically returned:
    - For each fused site tensor of rank `M + N + 2` (two-var fused layout has 2 bonds + 2 site-index pairs), extract it via `t4a_treetn_tensor`.
    - `reshape` each site tensor into `2` dim-2 physical indices per variable = 4 dim-2 site legs, separated into input/output pairs.
    - Use `t4a_tensor_svd` (Task 2's migrated path) to split the fused tensor into two per-variable tensors.
    - Reassemble into an Interleaved-layout TreeTN via `t4a_treetn_new` (existing primitive).

   Note: this is the "Extract → reshape → SVD → assemble" pipeline sketched in Issue #53.
3. Delete `_materialize_binaryop` from `src/QuanticsTransform/capi_helpers.jl`. Its only callers are `binaryop_operator` and `binaryop_operator_multivar`, both of which are replaced in this task.

**Tests:** `test/quanticstransform/materialize.jl` `@testset "binaryop_operator"`. Acceptance: numerical agreement with a dense reference within `1e-10` on two small cases (1-variable resolution `r ≤ 4`, pairwise `(a1, b1, a2, b2)` with both periodic and open BCs). Add one multivar case (`nvars = 3`, `lhs_var = 1`, `rhs_var = 3`).

## Task 10 — update the tensor4all-rs pin

**Affected files:** `deps/TENSOR4ALL_RS_PIN`.

**What to do:**
- Replace `0d7b954c3f7a315e42ddd5ca1a8b0727d538fe71` with `fd7180c` (full 40-char hash to be confirmed at PR time).
- Rebuild the Rust library and re-run the test suite.

**Note:** this is the last code change before the PR is opened, per `AGENTS.md` cross-repo rules (Rust PR already merged; Julia PR updates the pin).

## Task 11 — tests and documentation sweep

**Affected files:** `test/runtests.jl`, any test file that asserts `Cint(1)` / `Cint(2)` layout values, `docs/src/api.md`, `docs/src/truncation_contract.md` (if it exists), any docs pages referencing the layout enum or `binaryop`.

**What to do:**
- Run `T4A_SKIP_HDF5_TESTS=1 julia --startup-file=no --project=. test/runtests.jl` and confirm full pass.
- Run `julia --project=docs docs/make.jl` and confirm it still builds.
- Grep test files for any `Cint(1)` / `Cint(2)` that referenced the old layout enum and update.
- Include `test/tensornetworks/truncation_policy.jl` in `test/runtests.jl`.
- Update `docs/src/api.md`: add `src/TensorNetworks/truncation_policy.jl` to the `@autodocs Pages = [...]` list for the `TensorNetworks` section so `SvdTruncationPolicy` appears in the API reference (per the CI-enforced coverage rule in `AGENTS.md`).
- Update the Truncation Contract doc page (if present) to describe the new `svd_policy` kwarg, the precedence rules (`svd_policy` wins but cannot coexist with nonzero `rtol`/`cutoff`), and the `:absolute` / `:squared_value` / `:discarded_tail_sum` semantics newly available from Julia.
- Update docstrings on `truncate`, `add`, `contract`, `apply`, `linsolve`, `split_to`, `restructure_to`, and `Core.svd` to mention the new `svd_policy` kwarg and cross-reference `SvdTruncationPolicy`.
- Update `docs/plans/` index or add a short note.

## Task ordering

```
Task 1  (policy helper + enum renumber)         [foundation]
  ├── Task 2 (t4a_tensor_svd)                   [can parallel with 3-7]
  ├── Task 3 (t4a_treetn_truncate)
  ├── Task 4 (t4a_treetn_add)
  ├── Task 5 (t4a_treetn_contract)
  ├── Task 6 (t4a_treetn_apply_operator_chain)
  ├── Task 7 (t4a_treetn_linsolve)
  └── Task 8 (t4a_treetn_split_to + restructure_to)

Task 9  (binaryop re-implementation)             depends on Task 1 (layout enum) and Task 2 (SVD migration)

Task 10 (pin bump)                               depends on Tasks 1-9 complete

Task 11 (tests + docs sweep)                     last
```

Note: the pin cannot be updated before Tasks 1-8 are in a consistent state, because calling any migrated symbol against the old Rust pin would fail to link (the Rust signatures changed, not just semantics). It is acceptable to do local development with `TENSOR4ALL_RS_PATH` pointing at the tensor4all-rs worktree until Tasks 1-9 land.

## Test strategy

1. **Existing suite is the primary oracle.** All existing tests already exercise `rtol`, `cutoff`, `maxdim`, `form`, and `binaryop_operator`. The expectation is green-on-green after the migration.
2. **Foundation unit tests** (Task 1): default construction of `SvdTruncationPolicy`, symbol validation, FFI enum constants match the C header, `_resolve_svd_policy` returns `nothing` for the zero case, `cutoff` precedence reproduces the current `sqrt(cutoff)` mapping, ambiguity between `rtol`/`cutoff` and `svd_policy` raises `ArgumentError`.
3. **Per-migration regression tests**: for every migrated ccall, add one lightweight test that exercises (i) a non-trivial `rtol > 0` or `cutoff > 0` case so the convenience path is hit and (ii) a `svd_policy=SvdTruncationPolicy(...)` case that reaches a strategy combination (e.g. `:discarded_tail_sum` or `:squared_value`) not representable via `rtol` / `cutoff`. This verifies the full Rust strategy is genuinely reachable from Julia.
4. **Convenience-equals-full equivalence**: at least one test that `truncate(tt; rtol=τ)` and `truncate(tt; svd_policy=SvdTruncationPolicy(threshold=τ))` produce bit-identical results (the convenience path is a special case of the full-policy path).
5. **`form=:lu` error path**: one targeted test per affected function (`truncate`, `linsolve`, `split_to`) asserting the new `ArgumentError` is raised with a message mentioning "SVD-only" so future developers can grep for it.
6. **Binaryop parity** (Task 9): dense reference match within `1e-10` on at least one 2-variable and one multivar case, per Issue #53 acceptance criteria.
7. **Build check**: `cargo build -p tensor4all-capi --release` on the new pin; `deps/build.jl` dry-run succeeds.

## Open questions

1. **`form = :lu` handling for `truncate` / `linsolve` / `split_to`.** The new backend dropped LU from these three entry points. Three options:
   - (a) Raise `ArgumentError` with an actionable message. No existing test uses `form=:lu`, so nothing breaks.
   - (b) Silently coerce to `:unitary` with a `@warn`.
   - (c) Leave `form` accepting `:unitary` only and remove `:lu` from the Julia docstring.
   **Plan recommendation:** (a). It surfaces the backend change without changing numerical behavior, keeps the Julia signature stable, and costs only one `ArgumentError`. The maintainer may override.
2. **Public type name for `SvdTruncationPolicy`.** The plan uses `SvdTruncationPolicy` matching the Rust side. Alternatives worth a moment's thought: `TruncationPolicy` (drops `Svd` since the project-level truncation story is already SVD-only), `SVDTruncationPolicy` (all-caps `SVD` is more Julian). Recommendation: keep `SvdTruncationPolicy` to stay one-to-one with the Rust name — simplifies docs and reader mental model.
3. **Scope of `svd_policy` kwarg across functions.** The plan adds `svd_policy` to every truncating function (`truncate`, `add`, `contract`, `apply`, `linsolve`, `split_to`). For `restructure_to` the pair `split_svd_policy` / `final_svd_policy` is introduced. This is consistent with the existing `split_rtol` / `final_rtol` pattern. Confirm the maintainer wants the full symmetric surface; an alternative is to expose `svd_policy` only on `truncate` and have users pre-truncate, but that loses in-kernel control.
4. **Exact numerical semantics of the `(rtol, cutoff) → policy` mapping (convenience path only).** The plan keeps the existing Julia-side sqrt conversion (`cutoff` precedence, `threshold = sqrt(cutoff)`) and feeds it into `ThresholdScale=:relative`, `SingularValueMeasure=:value`, `TruncationRule=:per_value`. This preserves bit-for-bit behavior on existing tests. Users who want tail-sum or squared-value semantics now have `svd_policy` as the escape hatch.
5. **`_canonical_form_code` retention.** It is still needed for `t4a_treetn_orthogonalize` (which retained the `form` parameter). Keep the helper in `treetn.jl`; its callers in `truncate` / `linsolve` / `split_to` go away.
6. **Scope of the PR.** Tasks 1-11 are a single PR. Splitting Task 9 (binaryop) out would require keeping the old `_materialize_binaryop` alive temporarily, which breaks the pin bump (Task 10). One PR is simplest.
