# AGENTS.md

## API Design Principles

Tensor4all.jl wraps `tensor4all-rs` through a C FFI, but the current Julia
frontend is intentionally centered on a pure Julia public object model.

### Public Julia layer split

The intended public layers are:

- `Core`
- `Tensor4all.TensorNetworks`
- `Tensor4all.SimpleTT`
- `Tensor4all.TensorCI`
- `Tensor4all.QuanticsGrids`
- `Tensor4all.QuanticsTCI`
- `Tensor4all.QuanticsTransform`
- adopted and re-exported `QuanticsGrids.jl` / `QuanticsTCI.jl` surfaces

### C API layer

- Expose only general, multi-language useful kernels.
- For the Julia-facing design, prefer a minimized chain-oriented ABI:
  - index primitives
  - tensor primitives
  - tensor-tensor contraction
  - chain-kernel operations that Julia should not own
- Do not grow the C API for operations that can be composed in Julia from
  tensor lists and tensor contraction.

## Documentation

### Docstrings

- Every exported type and function defined in this repository must have a
  concise docstring.
- Keep docstrings concise and task-oriented. Do not add long narrative
  docstrings unless the API surface is genuinely subtle.
- Docstrings should describe the implemented behavior and current layer
  boundary.
- For partially implemented APIs, call out missing behavior explicitly without
  centering the API on unimplemented stubs.

## File Organization

- Keep submodule files responsibility-focused.
- Around 300-400 lines is a soft limit. Once a file grows past that range,
  consider splitting it.
- Around 500-600 lines is a strong signal to split unless the file is still a
  single tight unit.
- Prefer creating a subdirectory such as `src/Foo/` and separating by
  responsibility instead of accumulating unrelated sections in one file.


## Error Handling

### Principle: fail early with actionable messages

Users should never see a raw Rust panic or an opaque "Internal error". Errors
must explain what failed and how to fix it.

### Julia-side validation

- Validate arguments in Julia before calling the C API.
- Use `ArgumentError` for invalid arguments.
- Use `DimensionMismatch` for shape mismatches.
- Include actual versus expected values in the message.

### Array contiguity check

Julia arrays passed to the C API must be contiguous in memory.

- Views and non-contiguous reshapes must not be passed directly.
- Convert with `collect` if needed.
- Preferred message:
  `"Array must be contiguous in memory for C API. Got $(typeof(data)) with strides $(strides(data)). Use collect() to make a contiguous copy."`

### Rust-side errors

- Never discard the Rust error message returned through `last_error_message()`.
- Add Julia-side context around the Rust message when wrapping C API calls.

## Known Issues

### Julia x64 segfault on AMD EPYC (primerose)

On the `primerose` server (AMD EPYC 7713P, znver3), all x64 Julia binaries
segfault during JIT compilation of Pkg when processing `[compat]` or
`[extensions]` sections in `Project.toml`. This is a Julia compiler bug
specific to that CPU/system combination.

**Workaround**: run tests inside a Docker container:

```bash
# Build the Rust library first (on host)
TENSOR4ALL_RS_PATH=/path/to/tensor4all-rs julia --startup-file=no deps/build.jl

docker run --rm \
  -v $(pwd):/workspace/Tensor4all.jl:ro \
  -v /path/to/tensor4all-rs/target/release/libtensor4all_capi.so:/workspace/lib/libtensor4all_capi.so:ro \
  -e RUST_BACKTRACE=1 \
  julia:1.11 \
  bash -c '
    cp -r /workspace/Tensor4all.jl /tmp/T4all
    cp /workspace/lib/libtensor4all_capi.so /tmp/T4all/deps/
    cd /tmp/T4all && rm -f Manifest.toml
    julia --startup-file=no -e "using Pkg; Pkg.activate(\".\"); Pkg.instantiate(); Pkg.test()"
  '
```

### tensor4all-rs column-major migration

As of commit `ca97593` (PR #302), `tensor4all-rs` uses column-major tensor
storage. If transposed data appears in tests, check that no stale row-major
conversion remains on the Julia side.

## Build & Test

- Rust library: `deps/build.jl` builds `libtensor4all_capi.so` from
  `tensor4all-rs`
- When running `deps/build.jl` directly, use
  `julia --startup-file=no --project=. deps/build.jl`
- Rust source resolution:
  `TENSOR4ALL_RS_PATH` > sibling `../tensor4all-rs/` > GitHub clone
- Main tests: `Pkg.test()` or `julia test/runtests.jl`
- Skip HDF5 tests in direct runs with `T4A_SKIP_HDF5_TESTS=1`

## PR Checklist

- Review `README.md` if the public shape or expected workflow changed.
- Run `julia --project=docs docs/make.jl`.
- Keep docs aligned with the restored old Julia frontend architecture.
