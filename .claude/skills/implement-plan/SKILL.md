---
name: implement-plan
description: Implement an approved plan for Tensor4all.jl (maintainer use)
---

# Implement Plan

You are helping a maintainer implement an approved plan for Tensor4all.jl.

## Before you start

1. Read `CONTRIBUTING.md` for the contribution flow.
2. Read `AGENTS.md` for codebase conventions, error handling, and architecture.
3. Read the approved plan from the issue (should have `plan_approved` label,
   or maintainer has decided to proceed).
4. Understand every task in the plan and the overall test strategy.

## Your task

Implement the plan task by task. Follow TDD: write failing tests first, then
implement, then verify.

## Process

For each task in the plan:

1. **Write failing tests** — based on the test strategy in the plan.
2. **Run tests** — confirm they fail for the right reason.
3. **Implement** — minimal code to make tests pass.
4. **Run tests** — confirm they pass.
5. **Commit** — one commit per task, descriptive message referencing the issue.

## Guidelines

- Follow `AGENTS.md` strictly: error handling, docstrings, validation rules.
- Do not add features beyond what the plan specifies.
- If the plan has gaps or ambiguities, stop and ask rather than guessing.
- If a task requires C API changes in tensor4all-rs, stop and flag it — that
  needs a separate PR merged first (see cross-repo dependency rules).
- Run `Pkg.test()` after all tasks are complete.
- Run `julia --project=docs docs/make.jl` if any exported symbols changed.
- Do not relax test tolerances or skip tests to make them pass.
