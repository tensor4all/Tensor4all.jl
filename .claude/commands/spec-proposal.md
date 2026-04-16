# Spec Proposal

You are helping a contributor write a spec for a Tensor4all.jl GitHub issue.

## Before you start

1. Read `CONTRIBUTING.md` for the contribution flow.
2. Read `AGENTS.md` for codebase conventions and architecture.
3. Understand the issue: what is being proposed or reported.

## Your task

Explore the codebase to understand the relevant modules, types, and existing
behavior. Then draft a spec to post as the issue description or comment.

## Output format

Write the spec with these sections:

```markdown
## Summary
(What to add or change, 1-2 sentences)

## Motivation
(Why this is needed, which use case)

## Proposed approach
(Recommended design direction. Reference relevant modules/types.
 Pseudocode is fine; exact code is not required.)

## Alternatives considered
(Other approaches and why they were not chosen)

## Acceptance criteria
- [ ] ...
- [ ] ...
```

## Guidelines

- Keep it concise. Focus on the "what" and "why", not the "how" in detail.
- Reference module names (e.g. `TensorNetworks`, `SimpleTT`) but avoid
  hardcoding line numbers — they change.
- If the change touches the C API boundary, note that explicitly.
- Validate that the proposed approach is consistent with `AGENTS.md` design
  principles (e.g. Julia-owned semantics, minimized C API).
- If the feature requires new C API functions in tensor4all-rs, state this
  explicitly and suggest opening a linked issue on the tensor4all-rs repository.
  See the "Cross-Repository Changes" section of `CONTRIBUTING.md`.
