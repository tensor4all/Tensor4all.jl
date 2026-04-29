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

## FFI Handle Ownership Discipline

- Any Julia object that owns a Rust/C handle must make ownership explicit in
  its constructor or wrapper type.
- Do not rely on Julia's default `deepcopy` for objects that contain owned raw
  pointers. Define `Base.deepcopy_internal` to either clone/rebuild the
  backend resource or throw an actionable error.
- Public copy operations for handle-backed public objects must create an
  independently owned backend resource. They must not share an owned raw
  pointer with the source object.
- Direct copying of low-level handle wrappers should be disallowed unless the
  wrapper is explicitly non-owning or reference-counted.
- When transferring a newly returned C handle into a Julia finalizer-backed
  object, clear the local raw pointer variable before the surrounding `finally`
  block releases temporary handles.
- Regression tests for handle-backed public objects should check that
  `copy`/`deepcopy` preserve data and metadata while producing distinct backend
  pointers.
