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
- A suggested fix must not itself be a catalog entry: recommending
  `fulltensor` (FM4) to remove a pointwise loop (FM1) trades a slow path
  for an exponential one. Check every fix you propose against the catalog.
- Do not pad the report: a single-point `evaluate` outside a loop, a dense
  conversion in a debug helper, or a documented reference path is not a
  finding.
- Group B findings (silently wrong answers) outrank Group A findings (slow
  but correct) at equal confidence.
- This skill covers tensor4all-specific misuse only. For generic
  performance review, load `tensor4all-agent-rules`
  (`rules/common/performance.md`, `rules/julia/performance.md`).
