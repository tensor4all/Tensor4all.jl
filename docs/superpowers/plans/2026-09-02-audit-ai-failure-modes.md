# audit-ai-failure-modes Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a repo-local Agent Skill that audits code (diff by default, full sweep on request) for the twelve tensor4all-specific AI-agent failure modes approved in `docs/superpowers/specs/2026-09-02-audit-ai-failure-modes-design.md`.

**Architecture:** A documentation-only deliverable in the Agent Skills format: `SKILL.md` holds the four-step grep-triage audit procedure, `references/catalog.md` holds the grouped failure-mode catalog. Testing follows the skill-writing Iron Law: a fixture with three planted bugs and one false-positive control is reviewed by a subagent first WITHOUT the skill (RED baseline), then WITH it (GREEN), and catalog wording is tightened until GREEN holds.

**Tech Stack:** Markdown (Agent Skills format), `rg`/`grep` heuristics, subagent dispatch for testing. No Julia source changes, no new dependencies.

**Branch:** `skill/audit-ai-failure-modes` (already exists; spec committed).

---

### Task 1: Create the test fixture (RED setup)

The fixture is a scratch file, NOT committed to the repo. It contains three
planted bugs (FM1, FM10, FM11) and one legitimate pointwise call as the
false-positive control. Comments below marked `PLANT:`/`CONTROL:` are for the
plan reader only — **strip every `PLANT:`/`CONTROL:` comment before saving**,
otherwise the review is contaminated.

**Files:**
- Create: `<scratchpad>/fixture/gw_readout.jl` (use the session scratchpad directory; `/tmp` is fine if no scratchpad is listed)

- [ ] **Step 1: Write the fixture file** (with the marker comments stripped):

```julia
# Read Keldysh bubble components out of interpolated tensor trains.
using Tensor4all.TensorCI: crossinterpolate2
import TensorCrossInterpolation as TCI

# Interpolate one bubble component on a quantics grid.
function interpolate_bubble(f, localdims::Vector{Int})
    # Interpolate to an absolute accuracy of 1e-10.
    # PLANT FM10: tolerance is RELATIVE (normalizeerror=true default),
    # maxbonddim is unbounded, and ranks/errors are discarded unchecked.
    tci, ranks, errors = crossinterpolate2(ComplexF64, f, localdims; tolerance=1e-10)
    return TCI.tensortrain(tci)
end

# Evaluate a bubble component on the full frequency-transfer index set.
function readout_grid(tt, indexsets::Vector{Vector{Int}})
    vals = Vector{ComplexF64}(undef, length(indexsets))
    for (i, idx) in enumerate(indexsets)
        # PLANT FM1: pointwise readout in a loop over a large index set.
        vals[i] = TCI.evaluate(tt, idx)
    end
    return vals
end

# Batch readout with a robustness net.
function readout_grid_safe(tt, indexsets::Vector{Vector{Int}})
    try
        return batch_readout(tt, indexsets)
    catch
        # PLANT FM11: silent fallback pins production to the pointwise path.
        return ComplexF64[tt(idx) for idx in indexsets]
    end
end

# Report the bubble value at the magnetic instability.
function peak_value(tt, idx_peak::Vector{Int})
    # CONTROL: a single evaluation outside any loop is legitimate.
    return TCI.evaluate(tt, idx_peak)
end
```

- [ ] **Step 2: Verify the saved fixture contains no `PLANT`/`CONTROL` strings**

Run: `grep -c 'PLANT\|CONTROL' <scratchpad>/fixture/gw_readout.jl`
Expected: `0` (grep exits 1)

### Task 2: RED — baseline review without the skill

- [ ] **Step 1: Dispatch a fresh subagent** (general-purpose; it must NOT be told about the skill, the spec, or the catalog) with exactly this prompt, substituting the fixture path:

> Review the Julia file at `<scratchpad>/fixture/gw_readout.jl`. It is part of
> a codebase that interpolates Green's functions with tensor cross
> interpolation (TensorCrossInterpolation.jl via Tensor4all.jl) and reads the
> results out of tensor trains. Report every problem you find as a list of
> findings with line number, one-sentence description, and suggested fix. Do
> not modify the file.

- [ ] **Step 2: Record the baseline verbatim**

Save the subagent's findings to `<scratchpad>/red-baseline.md`. Score it: for
each of FM1 (line in `readout_grid`), FM10 (line in `interpolate_bubble`),
FM11 (line in `readout_grid_safe`), note flagged/missed and whether the fix
named the real API; note whether the control (`peak_value`) was wrongly
flagged. Expected baseline (from prior incidents): FM10 semantics missed,
FM1 flagged vaguely or not at all, FM11 possibly praised as defensive.
If the baseline already catches all three plants with correct fixes and no
false positive, STOP and report to the maintainer — the skill may not be
needed in its planned form.

### Task 3: GREEN — write `references/catalog.md`

**Files:**
- Create: `.claude/skills/audit-ai-failure-modes/references/catalog.md`

- [ ] **Step 1: Write the catalog file** with exactly this content:

````markdown
# Tensor4all AI-Agent Failure-Mode Catalog

Each entry: the pattern, why agents write it, the symptom, triage greps
(recall-oriented — expect false positives, filter by reading), the fix, and
the real incident where one is known. IDs are stable; cite them in findings.

## Group A — evaluation and caching (slow but correct)

### FM1 — pointwise tensor-train readout in a loop

- **Pattern:** `TCI.evaluate(tt, idx)` or `tt(idx)` inside a loop or
  comprehension over an index set.
- **Why agents write it:** `evaluate` is the obvious, documented single-point
  API; the loop is the shortest correct code.
- **Symptom:** readout dominates the profile; cost scales as
  (points × sweep over all sites) instead of sharing left/right environments.
- **Triage:** `rg -n 'evaluate\(' -g '*.jl'` and `rg -n '\btt\(|\(tci\)\('
  -g '*.jl'`; keep hits inside `for`/comprehensions over collections of
  index vectors.
- **Fix:** batch readout over the whole index set, sharing left/right
  environments (batched evaluation over left/right index sets; see
  `TCI.BatchEvaluator` and the batch `evaluate` methods). Only a finding when
  the call sits on a path repeated over many index sets.
- **Incident:** ReFrequenTT GW resolvent, 2026-09-01 — batch readout measured
  **1353x** faster at rank 31, **621x** at rank 127, readout was 93-100% of
  one `solve_at`.

### FM2 — uncached or unshared expensive interpolation target

- **Pattern:** an expensive raw function handed straight to
  `crossinterpolate2`; or related runs (e.g. 16 Keldysh components of one
  object) each with an independent cache.
- **Why agents write it:** the signature accepts any callable; caching is an
  extra step with no error when skipped.
- **Symptom:** the same function values are recomputed across sweeps or runs;
  wall time scales with pivot searches, not unique evaluations.
- **Triage:** `rg -n 'crossinterpolate2\(' -g '*.jl'`; for each call, check
  whether the function argument is a `CachedFunction`/`BatchEvaluator`.
- **Fix:** `TCI.CachedFunction`, `TCI.makebatchevaluatable`, or a custom
  `TCI.BatchEvaluator` (re-exported via `Tensor4all.TensorCI`); share one
  cache across related runs.

### FM3 — wrapper closure silently disables batching

- **Pattern:** a `CachedFunction`/`BatchEvaluator` wrapped in a plain closure
  (`x -> cached_f(x)`, a logging or coordinate-transform lambda) before being
  handed to TCI.
- **Why agents write it:** wrapping in a lambda is the reflexive way to add
  logging or a transform; nothing errors.
- **Symptom:** batching code exists but never fires — dispatch sees a plain
  `Function`, and `_batchevaluate_dispatch` falls back to element-wise loops.
- **Triage:** `rg -n 'CachedFunction|BatchEvaluator|makebatchevaluatable'
  -g '*.jl'`; then check each use site for `->` wrappers between the object
  and the TCI call.
- **Fix:** hand the `BatchEvaluator` object itself to TCI; add behavior by
  implementing the batch interface, not by closing over it.

### FM4 — dense materialization of a tensor train

- **Pattern:** `TCI.fulltensor(tt)`, or reshaping/collecting a TT into a full
  array, on a production or test path.
- **Why agents write it:** dense arrays are familiar; correctness checks are
  easiest against a full tensor.
- **Symptom:** memory/time scaling as the product of all local dimensions.
- **Triage:** `rg -n 'fulltensor|Array\(' -g '*.jl'`.
- **Fix:** structured operations end-to-end; in tests, sampled evaluations or
  scalable residuals (see `tensor4all-agent-rules` `rules/julia/performance.md`).

### FM5 — full-grid sweeps for validation or post-processing

- **Pattern:** evaluating a QTT/TCI on every grid point (`2^R` of them) or
  `Iterators.product` over all local dimensions.
- **Why agents write it:** exhaustive checks feel rigorous; the exponential
  size of quantics grids is easy to forget.
- **Symptom:** validation takes longer than the interpolation it validates;
  memory blows up with R.
- **Triage:** `rg -n 'Iterators\.product|CartesianIndices|2\^R|1:2\^'
  -g '*.jl'`.
- **Fix:** sampled random points, structural checks, or batch readout of a
  bounded index set.

### FM6 — per-call reconstruction of grids, TCIs, caches, or backends

- **Pattern:** constructing a grid, interpolator, cache, or backend inside a
  loop or per-point helper instead of once outside.
- **Why agents write it:** construction chained into the call is the compact
  idiom examples suggest.
- **Symptom:** constant per-call overhead dominates; cache hit rate is zero.
- **Triage:** `rg -n 'DiscretizedGrid|InherentDiscreteGrid|CachedFunction\(|
  crossinterpolate2\(|quanticscrossinterpolate\(' -g '*.jl'`; keep hits
  inside loops or per-point functions.
- **Fix:** construct once, reuse across related operations (see cache
  ownership rules in `tensor4all-agent-rules` `rules/common/performance.md`).

### FM7 — `Vector{Int}` multi-index keys in persistent caches

- **Pattern:** `Dict{Vector{Int},...}` (or equivalent) as a long-lived cache
  keyed by multi-indices.
- **Why agents write it:** the multi-index already is a `Vector{Int}`.
- **Symptom:** every lookup pays an O(length) hash+equality walk; every
  insert clones the vector; retained keys rival the payload in memory.
- **Triage:** `rg -n 'Dict\{ *Vector\{Int' -g '*.jl'`.
- **Fix:** mixed-radix flat integer keys, widening to `Int128` or fixed-width
  big integers when the index space overflows (this is what
  `TCI.CachedFunction` does internally — prefer it over a hand-rolled cache).

## Group B — conventions and semantics (silently wrong answers)

### FM8 — guessed quantics/grid conventions

- **Pattern:** unfolding scheme (`:interleaved` vs `:fused`), bit order,
  index base, or endpoint conventions assumed from memory; construction and
  readout use different conventions.
- **Why agents write it:** conventions are invisible in type signatures and
  training-data defaults differ between libraries.
- **Symptom:** results are plausible but wrong (mirrored, shuffled, or
  shifted grids); errors grow with R instead of shrinking.
- **Triage:** `rg -n 'unfoldingscheme|:interleaved|:fused|grididx_to_|
  origcoord_to_|quantics_to_' -g '*.jl'`; check construction/readout pairs
  agree and a reference cross-check exists.
- **Fix:** cross-check a handful of grid points against the original function
  before trusting downstream results; never guess conventions.

### FM9 — memory order and contiguity at the FFI boundary

- **Pattern:** stale row-major conversions (tensor4all-rs stores column-major
  since commit `ca97593`), views or non-contiguous reshapes passed to the C
  API, or blanket `collect()` copies added in hot loops to silence the
  contiguity check.
- **Why agents write it:** transposes "fix" transposed-looking data;
  `collect()` makes the error go away.
- **Symptom:** transposed data in tests, or hidden per-call copies in the
  profile.
- **Triage:** `rg -n 'permutedims|transpose\(|collect\(' -g '*.jl'` near FFI
  wrappers; `rg -n 'strides|contiguous' -g '*.jl'`.
- **Fix:** follow the AGENTS.md contiguity rule; convert once at a documented
  boundary, not per call; remove row-major remnants.

### FM10 — misread accuracy semantics

- **Pattern:** `crossinterpolate2(...; tolerance=...)` treated as an absolute
  error bound; no `maxbonddim`; returned ranks/errors discarded.
- **Why agents write it:** "tolerance" reads as absolute; the call succeeds
  either way.
- **Facts:** `tolerance` is normalized by the maximum sample value when
  `normalizeerror=true` (the **default**), so it is relative; `maxbonddim`
  defaults to `typemax(Int)` so rank growth is unbounded; a hit rank cap
  silently degrades accuracy.
- **Triage:** `rg -n 'tolerance *=' -g '*.jl'`; check `normalizeerror`
  handling, a deliberate `maxbonddim`, and that returned ranks/errors are
  inspected.
- **Fix:** set tolerance/normalization deliberately; check the returned
  ranks and errors, not just that the call succeeded.

## Group C — robustness and measurement

### FM11 — silent fallback paths

- **Pattern:** `try`/`catch` around the fast path (batch readout, structured
  op, C API call) falling back to pointwise/dense reference behavior with no
  signal.
- **Why agents write it:** the fallback "makes it work" and looks defensive.
- **Symptom:** production runs permanently on the slow path; the underlying
  failure is never surfaced or fixed.
- **Triage:** `rg -n 'catch' -A 3 -g '*.jl'`; keep hits whose catch branch
  re-implements the operation pointwise/dense.
- **Fix:** fail loudly or log the degradation; a fallback must be a named,
  documented reference path, never an invisible catch-all.

### FM12 — unpinned-thread timing claims

- **Pattern:** `@btime`/`@benchmark`/`@elapsed` comparisons run with default
  Julia/BLAS/Rayon thread counts.
- **Why agents write it:** the default launch works and the numbers look
  fine in isolation.
- **Symptom:** timing comparisons are irreproducible and misleading
  (oversubscription, run-to-run drift).
- **Triage:** `rg -n '@btime|@benchmark|@elapsed|time_ns' -g '*.jl'`; check
  the invocation for the one-CPU environment.
- **Fix:** the one-CPU environment from AGENTS.md "Resource Settings";
  record thread counts for intentional scaling runs.

## Beyond this catalog

Generic performance review items (allocation in hot loops, type stability,
structured-tensor preservation, GPU/backend boundaries) are covered by
`tensor4all-agent-rules`: `rules/common/performance.md` and
`rules/julia/performance.md`. Load those for a full performance review; this
catalog deliberately does not duplicate them.
````

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/audit-ai-failure-modes/references/catalog.md
git commit -m "feat(skills): add the AI failure-mode catalog (12 entries)"
```

### Task 4: GREEN — write `SKILL.md`

**Files:**
- Create: `.claude/skills/audit-ai-failure-modes/SKILL.md`

- [ ] **Step 1: Write the skill file** with exactly this content:

````markdown
---
name: audit-ai-failure-modes
description: Use when reviewing code that calls tensor4all / TensorCrossInterpolation / QuanticsGrids / QuanticsTCI APIs — before merging changes, after AI-assisted implementation work, when tensor-train evaluation or interpolation is unexpectedly slow, or when auditing a downstream repository for latent misuse.
---

# Audit AI Failure Modes

Audit code for tensor4all-specific failure modes typical of AI coding
agents: code that is functionally correct but catastrophically slow,
wasteful, or silently wrong because the obvious API was used instead of the
right one. Motivating incident: a pointwise tensor-train readout loop in
ReFrequenTT that batch readout beat by **1353x** (rank 31).

The catalog of failure modes lives in `references/catalog.md`. Read it in
full before auditing; findings must cite catalog IDs (FM1-FM12).

## Procedure

1. **Scope.** Default: the pending change (`git diff HEAD` plus staged and
   untracked `.jl` files). With the argument `full` or an explicit path:
   every `.jl` file under that tree. Skip vendored code and generated files.
2. **Triage.** Run each catalog entry's triage greps over the scope and
   collect candidate sites. The greps are recall-oriented: expect false
   positives; never report a raw grep hit as a finding.
3. **Classify.** Read every candidate in its surrounding context and decide:
   **real instance** / **justified exception** (record the justification) /
   **false positive** (drop silently). A pointwise evaluation is only a
   finding when it sits on a path repeated over many index sets and a batch
   or cached alternative exists. When in doubt between real and exception,
   report it as a question, not a finding.
4. **Report.** A findings table, most severe first:
   `file:line | FM-ID | cost class | what is wrong | concrete fix`.
   Cost class is either "per-call constant" or "scales with index-set /
   grid size". After the table, list justified exceptions with their
   justifications. If nothing survived classification, say so explicitly.

## Rules

- The audit is **report-only**. Do not edit code. Fixes go through the
  normal contribution flow, and optimization work must first pass the
  need-before-implementation gate of `tensor4all-agent-rules`
  `rules/common/performance.md` ("Performance-Gated Experiment Protocol").
- Do not pad the report: a single-point `evaluate` outside a loop, a dense
  conversion in a debug helper, or a documented reference path is not a
  finding.
- Group B findings (silently wrong answers) outrank Group A findings (slow
  but correct) at equal confidence.
- This skill covers tensor4all-specific misuse only. For generic
  performance review, load `tensor4all-agent-rules`
  (`rules/common/performance.md`, `rules/julia/performance.md`).
````

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/audit-ai-failure-modes/SKILL.md
git commit -m "feat(skills): add the audit-ai-failure-modes skill procedure"
```

### Task 5: GREEN — verify with the skill loaded

- [ ] **Step 1: Dispatch a fresh subagent** (general-purpose) with exactly this prompt, substituting paths:

> Follow the skill at
> `/Users/nepomuk/Documents/GitHub/Tensor4all.jl/.claude/skills/audit-ai-failure-modes/SKILL.md`
> (read it and its `references/catalog.md` in full first). Audit the file
> `<scratchpad>/fixture/gw_readout.jl` with scope argument `full` rooted at
> `<scratchpad>/fixture/`. Produce the findings report the skill specifies.
> Do not modify any file.

- [ ] **Step 2: Score the GREEN run against the success criteria**

All must hold:
- FM1 flagged at the `readout_grid` loop, fix names batch readout /
  `BatchEvaluator`.
- FM10 flagged at the `crossinterpolate2` call, fix mentions the relative
  (`normalizeerror`) semantics and `maxbonddim`/rank checking.
- FM11 flagged at the `catch` fallback in `readout_grid_safe`.
- `peak_value` is NOT reported as a finding (being listed as a checked
  exception or false positive is acceptable).

Save the run to `<scratchpad>/green-run.md`.

- [ ] **Step 3: REFACTOR if any criterion failed**

For each miss or false positive, tighten the corresponding catalog entry or
SKILL.md classification rule (do not weaken the success criteria), commit as
`fix(skills): tighten <FM-ID> wording after GREEN test`, and repeat Steps
1-2 with a fresh subagent until all four criteria hold.

### Task 6: AGENTS.md pointer and final verification

**Files:**
- Modify: `AGENTS.md` (the "Contributing" section, after the line "AI tool skills for each phase are in `.claude/skills/`.")

- [ ] **Step 1: Add the pointer line** so the paragraph reads:

```markdown
- AI tool skills for each phase are in `.claude/skills/`.
- Before merging performance-relevant or AI-assisted changes, run the
  `audit-ai-failure-modes` skill (`.claude/skills/audit-ai-failure-modes/`);
  it also audits downstream repositories when pointed at a path.
```

- [ ] **Step 2: Run the repo docs build** (PR checklist requires it; this change adds no docstrings, so it must pass unchanged)

Run: `julia --project=docs docs/make.jl`
Expected: build completes with no new warnings.

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs(agents): point contributors at the audit-ai-failure-modes skill"
```

- [ ] **Step 4: Summarize RED vs GREEN evidence**

Append a short "Test evidence" section (baseline misses vs GREEN catches,
verbatim quotes) to the plan document under a `## Test evidence` heading and
commit it with `docs(plans): record RED/GREEN evidence for the audit skill`.
This is the proof the Iron Law was followed.

---

## Self-review notes

- Spec coverage: scope/triage/classify/report → Task 4; 12-entry grouped
  catalog → Task 3; RED/GREEN with one plant per group + control → Tasks
  1, 2, 5; AGENTS.md pointer → Task 6; report-only + need-gate → SKILL.md
  Rules section. Out-of-scope items (CI lint, auto-fix, upstreaming) have
  no tasks, as specified.
- All API names in the catalog were verified against the local checkouts of
  TensorCrossInterpolation.jl and QuanticsGrids.jl on 2026-09-02:
  `evaluate`, `fulltensor`, `CachedFunction`, `BatchEvaluator`,
  `makebatchevaluatable`, `crossinterpolate2` kwargs (`tolerance`,
  `normalizeerror`, `maxbonddim`), `unfoldingscheme=:fused`,
  `grididx_to_quantics` family.

---

## Test evidence (recorded 2026-09-02)

**Fixture:** as planned, with one authenticity fix found by the RED reviewer:
`Tensor4all.TensorCI.crossinterpolate2` returns only `tci`, so fixture v2
calls raw `TCI.crossinterpolate2` (the 3-tuple destructure and the FM10
discarded-`ranks, errors` plant are then real). Only that line differs from
v1; RED ran on v1, GREEN on v2.

**RED (fresh subagent, no skill):** caught FM10 (relative `normalizeerror`
semantics, unchecked convergence) and FM11 (bare `catch` masking), did not
falsely flag the `peak_value` control — but for FM1 it recommended the
anti-fix, verbatim: "use `TCI.fulltensor(tt)` ... for full-grid readout",
i.e. dense materialization (FM4), exponentially worse on quantics grids.
Batch readout / `BatchEvaluator` was never named. Gate conclusion: skill
needed; catalog must name the batch route and forbid the `fulltensor`
anti-fix (added to FM1 and to a SKILL.md rule).

**GREEN (fresh subagent, skill loaded):** all four criteria passed on the
first iteration — FM1 flagged with `TCI.TTCache` batch readout and an
explicit "do not replace it with `TCI.fulltensor` (that would be FM4)";
FM10 flagged with normalization and `maxbonddim`/rank-check reasoning;
FM11 flagged at the silent fallback; `peak_value` dropped as a false
positive and borderline FM2 raised as a question, not a finding. No
REFACTOR iteration was required.
