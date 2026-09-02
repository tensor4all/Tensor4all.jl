---
name: audit-tensor4all-usage
description: Use when reviewing code that calls tensor4all / TensorCrossInterpolation / QuanticsGrids / QuanticsTCI APIs — before merging changes, when tensor-train evaluation or interpolation is unexpectedly slow, or when auditing a repository for latent performance or correctness misuse.
---

# Audit Tensor4all Usage (launcher)

This is a thin launcher; it carries no rule content. The canonical audit
procedure and failure-mode catalog live at the repository root in
[`rules/tensor4all-usage-audit.md`](../../../rules/tensor4all-usage-audit.md).
Read that document in full and follow it, passing through any argument
(`full` or a path) as the audit scope.

From another repository, use the canonical copy:
https://github.com/tensor4all/Tensor4all.jl/blob/main/rules/tensor4all-usage-audit.md
(sibling checkout fallback: `../Tensor4all.jl/rules/tensor4all-usage-audit.md`).
