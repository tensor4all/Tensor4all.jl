# Design: `audit-ai-failure-modes` skill

Date: 2026-09-02
Status: approved

## Summary

A repo-local Agent Skill at `.claude/skills/audit-ai-failure-modes/` that
audits code for tensor4all-specific failure modes typical of AI coding
agents: code that is functionally correct but catastrophically slow or
wasteful because the agent used the obvious API instead of the right one.

Motivating incident: ReFrequenTT's closed-form GW resolvent spent essentially
all of its resolvent stage reading bubble components out of tensor trains one
point at a time. Batch readout sharing left/right environments was measured
**1353x faster at rank 31** and **621x at rank 127** (2026-09-01, saved
bubbles, agreement with pointwise to 1e-13). The pattern was latent across
the codebase and only discovered late, via profiling.

## Decisions

These were settled with the maintainer during brainstorming:

1. **Audit target:** pending diff/PR by default; a `full` argument (or an
   explicit path) sweeps a whole codebase, so the skill can be pointed at
   downstream repos such as ReFrequenTT.
2. **Catalog scope:** tensor4all-specific AI-misuse patterns owned by the
   skill; generic performance items are referenced from
   `tensor4all-agent-rules`, not duplicated.
3. **Placement:** skill and catalog live entirely in Tensor4all.jl
   (self-contained). Upstreaming the catalog to `tensor4all-agent-rules` is a
   later step, once the catalog has proven itself.
4. **Mechanism:** grep-assisted triage — catalog-defined grep heuristics
   surface candidate sites, the invoking agent reads each candidate in
   context and classifies it. No new lint script to maintain.

## Skill layout

```
.claude/skills/audit-ai-failure-modes/
  SKILL.md              # frontmatter + audit procedure (~1 page)
  references/catalog.md # failure-mode catalog, one section per mode
```

Frontmatter description states triggering conditions only (per Agent Skills
best practice; no workflow summary):

> Use when reviewing code that calls tensor4all / TensorCrossInterpolation /
> Quantics APIs — before merging changes, after AI-assisted implementation
> work, or when tensor-train evaluation or interpolation is unexpectedly
> slow.

## Audit procedure (SKILL.md content outline)

1. **Scope.** Default: the pending diff (`git diff` + staged + untracked
   files touched by the change). With argument `full` or a path: the whole
   tree under that path. Only `.jl` sources and scripts; skip vendored code.
2. **Triage.** For each catalog entry, run its grep heuristics over the
   scope to collect candidate sites. Heuristics are recall-oriented; false
   positives are expected and filtered in the next step.
3. **Classify.** Read each candidate in context. Classify as: real instance
   / justified exception (record the justification) / false positive.
   A pointwise evaluation is only a finding when it sits on a path repeated
   over many index sets and a batch or cached alternative exists.
4. **Report.** Findings table: `file:line`, failure-mode ID, estimated cost
   class (per-call constant vs. scales with index-set size), the concrete
   fix API, and the justification for classification. The audit is
   **report-only**: fixes go through the normal contribution flow, and any
   optimization work must first pass the need-before-implementation gate of
   the shared performance rules (`rules/common/performance.md`,
   "Performance-Gated Experiment Protocol").

## Catalog v1

Each entry carries: the pattern, why agents write it, the symptom, grep
heuristics, the fix with real API names, and the real incident where known.
Entries are grouped so the audit can be scoped ("evaluation and caching
only") when a full pass is too broad.

### Group A — evaluation and caching (slow but correct)

| ID  | Failure mode | Fix |
| --- | --- | --- |
| FM1 | Pointwise `evaluate(tt, idx)` / `tt(idx)` inside loops over index sets | Batch readout sharing left/right environments. Incident: ReFrequenTT, 1353x (rank 31) / 621x (rank 127) |
| FM2 | Expensive raw function handed to `crossinterpolate2`; or no cache shared across related runs (e.g. component-wise interpolation of one physical object) | `TCI.CachedFunction`, `makebatchevaluatable`, or a custom `TCI.BatchEvaluator` (re-exported via `Tensor4all.TensorCI`); share the cache across related runs |
| FM3 | Batching silently disabled by a wrapper: `x -> cached_f(x)` or a logging/transform closure strips the `BatchEvaluator` subtype, so `_batchevaluate_dispatch` falls back to element-wise loops | Keep the `BatchEvaluator` object itself as the callable handed to TCI; wrap by implementing the batch interface, not with a plain closure |
| FM4 | Dense materialization of a tensor train in production paths or tests | Structured operations; sampled evaluations / scalable residuals in tests (see `rules/julia/performance.md`) |
| FM5 | Full-grid sweeps: validating or post-processing a QTT/TCI by evaluating all `2^R` grid points or a dense meshgrid (`Iterators.product` over all local dims) | Sampled random points, structural checks, or batch readout of a bounded index set |
| FM6 | Per-call reconstruction of grids, TCIs, caches, or backends inside loops | Construct once, reuse across related operations |
| FM7 | `Vector{Int}` multi-index keys in persistent caches | Mixed-radix flat integer keys (widen to `Int128`/fixed-width big integers on overflow) |

### Group B — conventions and semantics (silently wrong answers)

| ID  | Failure mode | Fix |
| --- | --- | --- |
| FM8 | Guessed quantics/grid conventions: interleaved vs fused unfolding scheme, bit/endianness order, index base, inclusive vs exclusive endpoints — construction and readout disagree | Cross-check a handful of grid points against the original function before trusting any downstream result; never guess conventions from memory |
| FM9 | Memory-order and contiguity errors at the FFI boundary: stale row-major conversions (tensor4all-rs is column-major since `ca97593`), views/non-contiguous reshapes passed to the C API, or blanket `collect()` copies added inside hot loops to silence the contiguity check | Follow the AGENTS.md contiguity rule; convert once at a documented boundary, not per call |
| FM10 | Misread accuracy semantics: `crossinterpolate2`'s `tolerance` is normalized by the max sample value when `normalizeerror=true` (the default), so it is relative, not absolute; `maxbonddim` defaults to `typemax(Int)` so rank growth is unbounded; a hit rank cap silently degrades accuracy | Set tolerance/normalization deliberately; check the returned ranks and errors, not just that the call succeeded |

### Group C — robustness and measurement

| ID  | Failure mode | Fix |
| --- | --- | --- |
| FM11 | Silent fallback paths: `try`/`catch` around the fast path (batch readout, structured op, C API call) falling back to pointwise/dense reference behavior, leaving production permanently on the slow path with no signal | Fail loudly or log the degradation; a fallback must be a named, documented reference path, never an invisible catch-all |
| FM12 | Timing or benchmark claims from runs without pinned threads (Julia/BLAS/Rayon defaults oversubscribe and make comparisons misleading) | The one-CPU environment from AGENTS.md "Resource Settings"; record thread counts for intentional scaling runs |

The catalog ends with a pointer to `tensor4all-agent-rules`
(`common/performance.md`, `julia/performance.md`) for generic performance
review items (allocation in hot loops, type stability, structured-tensor
preservation), which the skill deliberately does not duplicate.

## Test plan

Per the skill-writing Iron Law (no skill without a failing test first):

- **Fixture:** a scratch scenario containing one planted bug per catalog
  group — FM1 (pointwise readout in a loop over a large index set), FM10
  (tolerance treated as absolute with an unbounded `maxbonddim` and no rank
  check), FM11 (a `try`/`catch` falling back silently to pointwise readout)
  — plus one legitimate pointwise call (single-point evaluation outside any
  loop) as the false-positive control.
- **RED:** a subagent reviews the fixture without the skill; document what
  it misses or misjudges, verbatim.
- **GREEN:** the same review with the skill loaded must flag all three
  plants with the correct fix APIs and must not flag the legitimate call.
- **REFACTOR:** tighten catalog wording until both hold; capture any new
  rationalizations as explicit counters in the skill.

## Repo integration

- One-line pointer in AGENTS.md (skills section) so agents discover the
  skill.
- No `docs/src/api.md` / autodocs impact: the skill adds no Julia source.
- Later (out of scope here): upstream the catalog to
  `tensor4all-agent-rules`, and consider the broader downstream-usage skill
  that `rules/common/agent-consumers.md` mandates — this audit skill is its
  review-side counterpart, not a replacement.

## Out of scope

- A CI lint script (mechanical detection of batchable pointwise loops needs
  semantic judgment; revisit only for narrowly mechanical modes like FM7).
- Auto-fixing findings.
- Broad AI failure modes not tied to tensor4all APIs (tolerance weakening,
  benchmark cherry-picking) — covered by shared rules and generic review
  tooling.
