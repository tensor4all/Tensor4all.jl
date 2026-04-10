# Design Documents

Repository-scoped design documents (for example, tenferro internals) live in each
implementation repository, such as
[tenferro-rs/docs/design/](https://github.com/tensor4all/tenferro-rs/tree/main/docs/design).

Cross-repository architecture/design documents are stored here.

The Julia frontend design is split into a hub-and-spoke set:

- [julia_ffi.md](./julia_ffi.md) for the overview and index
- [julia_ffi_core.md](./julia_ffi_core.md) for low-level primitives
- [julia_ffi_tt.md](./julia_ffi_tt.md) for backend TT support
- [julia_ffi_quantics.md](./julia_ffi_quantics.md) for quantics grids and transforms
- [bubbleteaCI.md](./bubbleteaCI.md) for reusable `TTFunction` logic and migration
- [julia_ffi_extensions.md](./julia_ffi_extensions.md) for compatibility and extension glue
- [julia_ffi_roadmap.md](./julia_ffi_roadmap.md) for the implementation plan
