# Repository Rules

## Index Identity Discipline

- Never compare `Index` objects by `id(index)` alone unless the code explicitly
  intends to ignore prime level, tags, and dimension.
- Use `==` for exact `Index` equality. In Tensor4all.jl this includes
  `dim`, `id`, `tags`, and `plev`.
- Tensor network structure and metadata matching must use exact `Index`
  equality by default. Tags are semantic metadata and must match.
- If C API round trips change tag metadata, treat that as a bug at the boundary
  that restored the index metadata, not as a reason to weaken equality.
- Any intentional collapse of prime levels must be explicit, for example via
  `noprime` or `setprime`, and covered by a regression test.
- Regression tests for index matching code should include two indices with the
  same `id` but different `plev`, and two indices with the same `id` and `plev`
  but different `tags`.
