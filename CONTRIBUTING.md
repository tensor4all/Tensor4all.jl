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
   │  For non-trivial changes, post an implementation plan
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
This is recommended for non-trivial changes regardless of your tooling.

A good plan includes:

- **Task breakdown** — logical units of work in order
- **Affected modules** — which parts of the codebase are touched
- **Test strategy** — what to test and how
- **Dependencies** — ordering constraints between tasks

You do not need exact code blocks or line numbers. The implementer will fill in
the details.

## Cross-Repository Changes (C API)

Some features require new C API functions in
[tensor4all-rs](https://github.com/tensor4all/tensor4all-rs). When this is the
case:

1. Open an issue on **tensor4all-rs** using its issue template. Link it to the
   Tensor4all.jl issue in the "Related Tensor4all.jl issue" field.
2. The tensor4all-rs PR must be merged first.
3. Update the pin in `deps/build.jl` to the merged commit.
4. Then open the Tensor4all.jl implementation PR.

If you are unsure whether C API changes are needed, note it in your spec. The
maintainer will advise during review.

## Code Guidelines

- Read `AGENTS.md` for codebase conventions (error handling, module layout,
  docstrings, etc.).
- Run `Pkg.test()` before submitting a PR.
- Run `julia --project=docs docs/make.jl` if docs are affected.

## PR Checklist

- Reference the related issue (e.g. "Closes #40").
- Review `README.md` if the public shape or expected workflow changed.
- Run `Pkg.test()` and confirm all tests pass.
- Run `julia --project=docs docs/make.jl` if any exported symbols changed.
- Keep docs aligned with the Julia frontend architecture described in
  `AGENTS.md`.

## AI Tool Skills

This repository provides optional skills for AI coding tools (Claude Code,
OpenCode, etc.) to assist with each phase of the contribution flow:

| Skill | Phase | User |
|-------|-------|------|
| `spec-proposal` | Write a spec for an issue | Contributors |
| `plan-proposal` | Write an implementation plan | Contributors |
| `implement-plan` | Implement from an approved plan | Maintainers |

### How to use skills

**Claude Code:** Type `/spec-proposal` or `/plan-proposal` as a slash command.

**OpenCode:** Skills are auto-discovered from `.claude/skills/`. Invoke by name.

**Other AI tools:** Ask your AI assistant to read the skill file (e.g.
`.claude/skills/spec-proposal/SKILL.md`) and follow its instructions.
