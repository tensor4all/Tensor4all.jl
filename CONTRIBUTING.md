# Contributing to Tensor4all.jl

## Contribution Flow

```
1. Open Issue (proposed)
   │  Use the Feature Request or Bug Report template.
   │
2. Maintainer reviews spec
   │  Discussion happens on the issue.
   │  Label changes to `spec_approved` when the approach is agreed.
   │
3. (Optional) Post detailed plan
   │  If you are using AI tools, post an implementation plan
   │  as a comment on the approved issue.
   │  Label changes to `plan_approved` when the plan is accepted.
   │
4. Implementation PR
   │  Open a PR that references the issue (e.g. "Closes #40").
   │  Maintainer reviews and merges.
```

Maintainers may skip labels and move directly to implementation at their
discretion.

## Issue Templates

Use the templates in `.github/ISSUE_TEMPLATE/`:

- **Feature Request** — new functionality or enhancements
- **Bug Report** — unexpected behavior or errors

## Issue Labels

| Label | Meaning |
|-------|---------|
| `proposed` | Issue opened, awaiting maintainer review |
| `spec_approved` | Approach agreed, ready for plan or implementation |
| `plan_approved` | Detailed plan accepted, ready for implementation |

## Writing a Spec (Issue Description)

A good spec includes:

- **Summary** — what you want to add or change (1-2 sentences)
- **Motivation** — why this is needed, which use case it serves
- **Proposed approach** — your recommended design direction
- **Acceptance criteria** — checklist of what "done" means

Keep it concise. Pseudocode is welcome; exact file paths and line numbers are
not required.

## Writing a Plan (Issue Comment)

After `spec_approved`, you may post a detailed implementation plan as a comment.
This is especially useful when working with AI tools.

A good plan includes:

- **Task breakdown** — logical units of work in order
- **Affected modules** — which parts of the codebase are touched
- **Test strategy** — what to test and how
- **Dependencies** — ordering constraints between tasks

You do not need exact code blocks or line numbers. The implementer (human or AI)
will fill in the details.

## AI Tool Skills

This repository provides skills for AI coding tools (Claude Code, OpenCode,
etc.) to assist with each phase:

| Skill | Phase | User |
|-------|-------|------|
| `spec-proposal` | Write a spec for an issue | Contributors |
| `plan-proposal` | Write an implementation plan | Contributors |
| `implement-plan` | Implement from an approved plan | Maintainers |

Skills are located in `.claude/skills/` and are compatible with both Claude Code
and OpenCode.

## Code Guidelines

- Read `AGENTS.md` before working on the codebase.
- Follow the conventions described there (error handling, docstrings, etc.).
- Run `Pkg.test()` before submitting a PR.
- Run `julia --project=docs docs/make.jl` if docs are affected.
