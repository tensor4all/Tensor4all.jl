# Tensor4all Usage Audit

Audit code for performance and correctness failure modes in the use of
tensor4all APIs: code that is functionally correct but catastrophically slow
or wasteful, or silently wrong, because the obvious API was used instead of
the right one. These mistakes are made by humans and coding agents alike;
agents amplify them at scale. Motivating incident: a pointwise tensor-train
readout loop in ReFrequenTT that batch readout beat by **1353x** (rank 31).

This document is the **single canonical location** for the audit procedure
and the failure-mode catalog. Harness skill files are thin launchers that
point here and carry no rule content of their own.

**When to run:** when reviewing code that calls tensor4all /
TensorCrossInterpolation / QuanticsGrids / QuanticsTCI APIs — before merging
changes, when tensor-train evaluation or interpolation is unexpectedly slow,
or when auditing a repository for latent misuse.

## Procedure

Findings must cite catalog IDs (FM1-FM12). Read the catalog below in full
before auditing.

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
- A suggested fix must not itself be a catalog entry: recommending
  `fulltensor` (FM4) to remove a pointwise loop (FM1) trades a slow path
  for an exponential one. Check every fix you propose against the catalog.
- Do not pad the report: a single-point `evaluate` outside a loop, a dense
  conversion in a debug helper, or a documented reference path is not a
  finding.
- Group B findings (silently wrong answers) outrank Group A findings (slow
  but correct) at equal confidence.
- This audit covers tensor4all-specific misuse only. For generic
  performance review, load `tensor4all-agent-rules`
  (`rules/common/performance.md`, `rules/julia/performance.md`).

# Failure-Mode Catalog

Each entry: the pattern, why it gets written, the symptom, triage greps
(recall-oriented — expect false positives, filter by reading), the fix, and
the real incident where one is known. IDs are stable; cite them in findings.

## Group A — evaluation and caching (slow but correct)

### FM1 — pointwise tensor-train readout in a loop

- **Pattern:** `TCI.evaluate(tt, idx)` or `tt(idx)` inside a loop or
  comprehension over an index set.
- **Why it gets written:** `evaluate` is the obvious, documented single-point
  API; the loop is the shortest correct code.
- **Symptom:** readout dominates the profile; cost scales as
  (points × sweep over all sites) instead of sharing left/right environments.
- **Triage:** `rg -n 'evaluate\(' -g '*.jl'` and `rg -n '\btt\(' -g '*.jl'`;
  keep hits inside `for` loops or comprehensions over collections of index
  vectors.
- **Fix:** batch readout over the whole index set, sharing left/right
  environments: wrap the train in `TCI.TTCache` (a `BatchEvaluator` with
  cached partial contractions) and evaluate over left/right index sets, or
  use the batch `evaluate` methods. **Do not "fix" this with
  `TCI.fulltensor(tt)`** — dense materialization is FM4 and exponentially
  worse on quantics grids. Only a finding when the call sits on a path
  repeated over many index sets.
- **Incident:** ReFrequenTT GW resolvent, 2026-09-01 — batch readout measured
  **1353x** faster at rank 31, **621x** at rank 127; the pointwise readout
  was 93-100% of one `solve_at`.

### FM2 — uncached or unshared expensive interpolation target

- **Pattern:** an expensive raw function handed straight to
  `crossinterpolate2`; or related runs (e.g. 16 Keldysh components of one
  object) each with an independent cache.
- **Why it gets written:** the signature accepts any callable; caching is an
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
- **Why it gets written:** wrapping in a lambda is the reflexive way to add
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
- **Why it gets written:** dense arrays are familiar; correctness checks are
  easiest against a full tensor.
- **Symptom:** memory/time scaling as the product of all local dimensions.
- **Triage:** `rg -n 'fulltensor|Array\(' -g '*.jl'`.
- **Fix:** structured operations end-to-end; in tests, sampled evaluations or
  scalable residuals (see `tensor4all-agent-rules`
  `rules/julia/performance.md`).

### FM5 — full-grid sweeps for validation or post-processing

- **Pattern:** evaluating a QTT/TCI on every grid point (`2^R` of them) or
  `Iterators.product` over all local dimensions.
- **Why it gets written:** exhaustive checks feel rigorous; the exponential
  size of quantics grids is easy to forget.
- **Symptom:** validation takes longer than the interpolation it validates;
  memory blows up with R.
- **Triage:** `rg -n 'Iterators\.product|CartesianIndices|\b2\^' -g '*.jl'`.
- **Fix:** sampled random points, structural checks, or batch readout of a
  bounded index set.

### FM6 — per-call reconstruction of grids, TCIs, caches, or backends

- **Pattern:** constructing a grid, interpolator, cache, or backend inside a
  loop or per-point helper instead of once outside.
- **Why it gets written:** construction chained into the call is the compact
  idiom examples suggest.
- **Symptom:** constant per-call overhead dominates; cache hit rate is zero.
- **Triage:** `rg -n 'DiscretizedGrid|InherentDiscreteGrid|CachedFunction\(|crossinterpolate2\(|quanticscrossinterpolate\(' -g '*.jl'`;
  keep hits inside loops or per-point functions.
- **Fix:** construct once, reuse across related operations (see cache
  ownership rules in `tensor4all-agent-rules`
  `rules/common/performance.md`).

### FM7 — `Vector{Int}` multi-index keys in persistent caches

- **Pattern:** `Dict{Vector{Int},...}` (or equivalent) as a long-lived cache
  keyed by multi-indices.
- **Why it gets written:** the multi-index already is a `Vector{Int}`.
- **Symptom:** every lookup pays an O(length) hash+equality walk; every
  insert clones the vector; retained keys rival the payload in memory.
- **Triage:** `rg -n 'Dict\{ *Vector\{Int' -g '*.jl'`.
- **Fix:** mixed-radix flat integer keys, widening to `UInt128` (the
  `TCI.CachedFunction` default) or fixed-width unsigned big integers when the
  index space overflows — prefer `TCI.CachedFunction`, which does exactly
  this internally, over a hand-rolled cache.

## Group B — conventions and semantics (silently wrong answers)

### FM8 — guessed quantics/grid conventions

- **Pattern:** unfolding scheme (`:interleaved` vs `:fused`), bit order,
  index base, or endpoint conventions assumed from memory; construction and
  readout use different conventions.
- **Why it gets written:** conventions are invisible in type signatures and
  defaults differ between libraries.
- **Symptom:** results are plausible but wrong (mirrored, shuffled, or
  shifted grids); errors grow with R instead of shrinking.
- **Triage:** `rg -n 'unfoldingscheme|:interleaved|:fused|grididx_to_|origcoord_to_|quantics_to_' -g '*.jl'`;
  check construction/readout pairs agree and a reference cross-check exists.
- **Fix:** cross-check a handful of grid points against the original function
  before trusting downstream results; never guess conventions.

### FM9 — memory order and contiguity at the FFI boundary

- **Pattern:** stale row-major conversions (tensor4all-rs stores column-major
  since commit `ca97593`), views or non-contiguous reshapes passed to the C
  API, or blanket `collect()` copies added in hot loops to silence the
  contiguity check.
- **Why it gets written:** transposes "fix" transposed-looking data;
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
- **Why it gets written:** "tolerance" reads as absolute; the call succeeds
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
- **Why it gets written:** the fallback "makes it work" and looks defensive.
- **Symptom:** production runs permanently on the slow path; the underlying
  failure is never surfaced or fixed.
- **Triage:** `rg -n 'catch' -A 3 -g '*.jl'`; keep hits whose catch branch
  re-implements the operation pointwise/dense.
- **Fix:** fail loudly or log the degradation; a fallback must be a named,
  documented reference path, never an invisible catch-all.

### FM12 — unpinned-thread timing claims

- **Pattern:** `@btime`/`@benchmark`/`@elapsed` comparisons run with default
  Julia/BLAS/Rayon thread counts.
- **Why it gets written:** the default launch works and the numbers look
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

# Harness Discovery

The audit is harness-agnostic. Every harness reaches this same canonical
document; the launchers below contain discovery metadata and a pointer only —
never rule content.

- **Any harness that reads `AGENTS.md`** (including Codex and Pi): the
  routing rule in this repository's `AGENTS.md` points here. This is the
  portable entry point and always works.
- **Claude Code:** thin launcher at
  `.claude/skills/audit-tensor4all-usage/SKILL.md`, invocable as
  `/audit-tensor4all-usage` (optionally with `full` or a path).
- **OpenCode:** discovers the same Claude Code launcher via its compatible
  skill-discovery paths; the `AGENTS.md` route also applies.
- **Codex and DeepSeek Harness:** thin launcher at
  `.agents/skills/audit-tensor4all-usage/SKILL.md`.

When editing this document, do not add content to the launchers; when adding
a launcher for a new harness, keep it to frontmatter plus the pointer.

# Using this audit from another repository

A repository that builds on Tensor4all.jl (e.g. ReFrequenTT) references the
canonical document from its own `AGENTS.md` — remote first, sibling checkout
as fallback:

```markdown
Before merging changes that call tensor4all / TensorCrossInterpolation /
Quantics APIs, read and follow the usage audit at
https://github.com/tensor4all/Tensor4all.jl/blob/main/rules/tensor4all-usage-audit.md
If the remote cannot be resolved, use the sibling checkout:
../Tensor4all.jl/rules/tensor4all-usage-audit.md
```

Alternatively, copy the thin launcher (not this document) into the
downstream repository's own harness skill directory to make it directly
invocable there.
