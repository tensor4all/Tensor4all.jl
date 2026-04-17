## Related Issue

Closes #

## Summary

<!-- Brief description of what this PR does -->

## Checklist

- [ ] Related issue is linked above
- [ ] `Pkg.test()` passes
- [ ] `julia --project=docs docs/make.jl` runs cleanly (if exported symbols changed)
- [ ] If a new `src/**/*.jl` file defines public symbols, it is listed in some `@autodocs` `Pages = [...]` block in `docs/src/api.md` (verified by `scripts/check_autodocs_coverage.jl`)
- [ ] `README.md` updated (if public API or workflow changed)
