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

### TensorNetworks

`TensorNetworks.TensorTrain` is the primary indexed chain type.

```julia
mutable struct TensorTrain
    data::Vector{Tensor}
    llim::Int
    rlim::Int
end
```

- `data` stores the site tensors.
- `llim` and `rlim` follow the historical Julia-side chain convention.
- MPS-like versus MPO-like interpretation is structural, not type-level.
- HDF5 interoperability belongs here through `save_as_mps` / `load_tt`.
- The Phase 2 helper surface also belongs here:
  `findsite`, `findsites`, `findallsiteinds_by_tag`, `findallsites_by_tag`,
  `replace_siteinds!`, `replace_siteinds`, `replace_siteinds_part!`,
  `rearrange_siteinds`, `makesitediagonal`, `extractdiagonal`, and
  optionally `matchsiteinds`.

### SimpleTT

`SimpleTT.TensorTrain{T,N}` is the raw-array tensor-train layer.

```julia
mutable struct TensorTrain{T,N}
    sitetensors::Vector{Array{T,N}}
end
```

- `N=3` is the MPS-like raw-array form.
- `N=4` is the MPO-like raw-array form.
- Compression and MPO-MPO contraction are Julia-owned here.
- `SimpleTT.TensorTrain(tci)` is the Julia-side conversion boundary from
  `TensorCI2`.

### TensorCI

- `TensorCI.crossinterpolate2` should return `TensorCI2`.
- `TensorCI` should re-export the public `TensorCrossInterpolation.jl` surface
  where practical.
- Conversion from `TensorCI2` to `SimpleTT.TensorTrain` is Julia-owned.
- Do not make `TensorCI` return indexed `TensorNetworks.TensorTrain`.

### QuanticsGrids / QuanticsTCI

- `Tensor4all.QuanticsGrids` is a wrapper re-export over `QuanticsGrids.jl`.
- `Tensor4all.QuanticsTCI` is a wrapper re-export over `QuanticsTCI.jl`.
- These modules improve discoverability but do not transfer ownership.

### QuanticsTransform

- `QuanticsTransform` provides quantics-specific operator constructors only.
- `TensorNetworks` owns the generic `LinearOperator` type and `apply`.
- `set_input_space!`, `set_output_space!`, and `set_iospaces!` should accept
  explicit `Vector{Index}` arguments only.
- Do not add `TensorTrain`-based automatic binding for operator I/O spaces.
- Julia owns semantic validation and index mapping.
- Rust is expected to provide only the operator kernels that Julia should not
  reimplement.

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

## Documentation Requirements

### Docstrings

- Every exported type and function should have a concise docstring.
- Prefer `jldoctest` examples when they add real value.
- Docstrings should describe the implemented behavior and current layer
  boundary.

### Source file size

- Keep each submodule source file focused on one responsibility.
- Use about 250-300 lines as a soft limit for a single source file.
- If a submodule grows beyond that or starts mixing distinct responsibilities,
  split it into a dedicated `src/<Submodule>/` directory and separate files.

### Documenter.jl site

- Documentation is built with Documenter under `docs/`.
- `docs/make.jl` must build cleanly for PR-ready work.
- In manual pages, prefer `@autodocs` over long hand-written `@docs` lists when
  a page is primarily mirroring docstrings from source files.
- Use `Pages = [...]` and `Modules = [...]` filters so `@autodocs` stays scoped
  to the intended local source files.

### Documentation structure

- `docs/src/index.md` — overview and current architecture status
- `docs/src/modules.md` — module map and data flow
- `docs/src/api.md` — API-oriented reference notes
- `docs/design/` — design notes for the restored Julia frontend architecture

### Required architecture story

The docs should clearly describe:

- the restored old module split now used by the implementation
- `TensorNetworks.TensorTrain = Vector{Tensor} + llim/rlim`
- `SimpleTT.TensorTrain{T,N}`
- `TensorCI.crossinterpolate2 -> TensorCI2`
- `SimpleTT.TensorTrain(tci)` conversion
- wrapper re-exports for `QuanticsGrids` and `QuanticsTCI`
- `TensorNetworks.LinearOperator` plus `QuanticsTransform` constructors
- the remaining `TensorNetworks` helper names, with implemented behavior and
  missing behavior called out explicitly
- pure Julia `save_as_mps` / `load_tt`
- minimized Julia-facing C API assumptions

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
