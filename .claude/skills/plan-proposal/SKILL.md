---
name: plan-proposal
description: Write a detailed implementation plan for an approved Tensor4all.jl spec
---

# Plan Proposal

You are helping a contributor write an implementation plan for a Tensor4all.jl
issue whose spec has been approved (`spec_approved` label).

## Before you start

1. Read `CONTRIBUTING.md` for the contribution flow.
2. Read `AGENTS.md` for codebase conventions and architecture.
3. Read the approved spec on the issue.
4. Explore the codebase in depth: relevant source files, existing tests,
   module boundaries.

## Your task

Draft a detailed implementation plan to post as an issue comment.

## Output format

Write the plan with these sections:

```markdown
## Implementation Plan

### Overview
(1-2 sentences: what this plan delivers)

### Task 1: [name]
**Affected files:** (list modules/files, no line numbers needed)
**What to do:** (describe the change)
**Tests:** (what to test)

### Task 2: [name]
...

### Task ordering
(Which tasks depend on others, suggested sequence)

### Test strategy
(Overall approach: unit tests, integration tests, edge cases to cover)

### Open questions
(Anything that needs maintainer input before implementation)
```

## Guidelines

- Break work into small, independently testable tasks.
- Each task should touch a coherent set of files in one module.
- Describe what to change and why, not exact code. The implementer will write
  the code.
- Note any tasks that require C API changes (these need a separate
  tensor4all-rs PR first — see cross-repo dependency rules in `AGENTS.md`).
- Reference the acceptance criteria from the approved spec. Every criterion
  should map to at least one task.
- Prefer TDD: mention which tests to write before which implementation.
