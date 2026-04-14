# AGENTS.md

## API Design Principles

Tensor4all.jl wraps tensor4all-rs (Rust) via C FFI. The API is layered:

### C API layer (tensor4all-capi in Rust)
- **General, multi-language primitives only.** If Python/C++ would also need it, put it here.
- Examples: tensor contraction, TreeTN operations, index manipulation, TCI, quantics transforms
- Do NOT add chain-specific or application-specific operations

### Julia layer (this package)
- **TreeTN-general.** Design APIs for arbitrary tree tensor networks, not just chains.
- Chain-specific operations must include **runtime topology checks** (verify the TreeTN is a chain before proceeding).
- **Build on C API primitives.** If an operation can be implemented using `ttn[v]` get/set + tensor contraction + index operations, implement it in Julia — do not add a new C API function.
- Expose Rust functionality that isn't in C API yet only when it's general enough for multi-language use.

### Type system: TensorTrain as the primary chain type
`TensorTrain = TreeTensorNetwork{Int}` is the primary type for chain tensor networks. `MPS` and `MPO` are aliases for the same type — there is no type-level distinction.
- MPS-like: each vertex has 1 site index
- MPO-like: each vertex has 2 site indices (e.g., unprimed input + primed output)
- `is_chain(ttn)`: runtime check that topology is a linear chain with vertices 1, 2, ..., n
- `is_mps_like(tt)`: runtime check that each vertex has exactly 1 site index
- `is_mpo_like(tt)`: runtime check that each vertex has exactly 2 site indices (unprimed + primed pair)
- Operations that need a specific index structure check at runtime via these predicates.

### Key available C API primitives for building Julia-level operations
- `ttn[v]` / `ttn[v] = tensor` — get/set individual tensors at vertices
- `siteinds(ttn, v)` — site indices at a vertex
- `linkind(ttn, v1, v2)` — bond index between vertices
- `vertices(ttn)`, `neighbors(ttn, v)` — topology queries
- `contract(a, b)` — TreeTN-TreeTN contraction
- `orthogonalize!`, `truncate!`, `inner`, `norm`, `to_dense`, `evaluate`

### Known missing C API (should be added to tensor4all-rs)
- **Tensor-Tensor contraction**: Exists in Rust (`TensorDynLen::contract`) but not exposed in C API. This is fundamental and multi-language useful.

### Julia-level API to implement (using C API primitives)
These should NOT go in the C API — they can be built from existing primitives once Tensor-Tensor contraction is exposed:
- **Site summation**: Contract specific site indices on a TreeTN. Use `ttn[v]` + tensor contraction with sum vectors.
- **Tensor diagonal embedding**: Create tensor diagonal in specific indices. Use Julia array operations + Tensor constructor.
- **Partial site index replacement**: Replace specific site indices in TreeTN. Use `ttn[v]` get/set + index operations.
- **Adding/removing site indices at a vertex**: For "MPO-like" structures — manipulate the tensor at a vertex to add primed copies of site indices. This is NOT a type conversion (MPS === MPO), just tensor manipulation.

## Error Handling

### Principle: fail early with actionable messages
Users should never see a raw Rust panic or an opaque "Internal error". Errors must explain **what went wrong** and **how to fix it**.

### Julia-side validation (prefer)
Validate arguments in Julia **before** calling the C API. This produces familiar Julia stack traces and clear messages.
- Check types, dimensions, and index compatibility before C calls
- Use `ArgumentError` for invalid arguments, `DimensionMismatch` for shape mismatches
- Include the actual vs expected values in the message: `"expected MPS with $n sites, got $(nv(ttn))"`
- Runtime topology checks (`is_chain`, `is_mps_like`, `is_mpo_like`) must produce descriptive errors, not just `false`

### Array contiguity check
Julia arrays passed to the C API via pointer must be **contiguous in memory** (dense, column-major). Views, reshapes of non-contiguous data, `SubArray`, and `PermutedDimsArray` are NOT safe to pass directly.
- Before any `ccall` that takes a `Ptr{T}` to array data, verify contiguity or convert: `data = collect(data)` if needed.
- Use `Base.iscontiguous(A)` (Julia ≥ 1.11) or `stride(A,1) == 1 && strides match dense layout` to check.
- Error message: `"Array must be contiguous in memory for C API. Got $(typeof(data)) with strides $(strides(data)). Use collect() to make a contiguous copy."`

### Rust-side errors (propagate fully)
When a C API call fails, `check_status` already retrieves `last_error_message()` from Rust. Rules:
- **Never discard the Rust error message.** Always include it in the Julia exception.
- If wrapping a C API call in a higher-level Julia function, add context: `"Failed to contract TTNs: $rust_msg"`
- Rust error messages should describe the problem from the user's perspective, not internal Rust state.

### Examples of good vs bad error messages
```
# Bad
error("C API error: Internal error")
error("Null pointer error")

# Good
error("Cannot apply Fourier operator: input MPS has $(nv(mps)) sites but operator expects $r sites")
error("contract failed: input and operator have no common site indices. Did you call set_iospaces!() first?")
```

## Documentation Requirements

### Docstring rules
- Every exported type and function **must** have a docstring with a `# Examples` section.
- Examples should be runnable `jldoctest` blocks where possible. Use `julia> ` prompts for testable examples.
- For examples that require the Rust library or external state, use ` ```julia ... ``` ` fenced blocks (non-tested).
- Module-level docstrings (`@doc` at the top of each module file) should include an end-to-end usage overview.

### Documenter.jl site
- Documentation is built with [Documenter.jl](https://documenter.juliadocs.org/) under `docs/`.
- `docs/make.jl` generates the site; `docs/src/` contains manual pages.
- PR checklist includes `julia --project=docs docs/make.jl` passing without errors.

### Documentation structure
- `docs/src/index.md` — overview, module dependency graph, and quick start
- `docs/src/api.md` — auto-generated API reference via `@autodocs` (per-module sections)
- `docs/src/modules.md` — module overview with dependency graph and data flow between modules
- `docs/src/tutorials/` — pipeline-oriented guides that chain multiple modules:
  - Quantics interpolation: QuanticsGrids → QuanticsTCI → SimpleTT → MPS
  - Fourier analysis: QuanticsTCI → MPS → QuanticsTransform(fourier) → evaluate
  - Affine transform on MPS: TreeTN → QuanticsTransform(affine) → TreeTN
  - Tree TCI: TreeTCI → TreeTensorNetwork → operations

### Module dependency graph
```
Core:  Index, Tensor, Algorithm (foundation types)
         ↓
SimpleTT ←→ TreeTN (bidirectional conversion)
         ↓        ↓
  QuanticsGrids   QuanticsTransform (operators on TreeTN)
         ↓
  QuanticsTCI     TreeTCI (outputs TreeTN)
```
Typical pipeline: **QuanticsGrids** (grid定義) → **QuanticsTCI** (関数補間→SimpleTT) → **TreeTN** (MPS変換・操作) → **QuanticsTransform** (Fourier/shift/affine演算) → **TreeTN** (結果評価)

## Known Issues

### Julia x64 segfault on AMD EPYC (primerose)

On the `primerose` server (AMD EPYC 7713P, znver3), all x64 Julia binaries (1.10, 1.11, 1.12) segfault during JIT compilation of Pkg when processing `[compat]` or `[extensions]` sections in Project.toml. This is a Julia compiler bug specific to this CPU/system combination.

**Workaround**: Run tests inside a Docker container:

```bash
# Build the Rust library first (on host)
TENSOR4ALL_RS_PATH=/path/to/tensor4all-rs julia --startup-file=no deps/build.jl
# Or: cargo build -p tensor4all-capi --release in tensor4all-rs

# Run tests in Docker
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

**Important**: Do NOT delete `/sharehome/shinaoka/.julia/compiled/` — the precompile caches were built on a different machine and cannot be regenerated on primerose due to this bug.

### tensor4all-rs column-major migration

As of commit `ca97593` (PR #302), tensor4all-rs uses **column-major** storage for tensor data (matching Julia/Fortran convention). The Julia wrapper was updated to remove the row-major ↔ column-major conversion that was previously needed. If you see transposed data in test failures, check that no stale row-major conversion code remains.

## Build & Test

- Rust library: `deps/build.jl` builds `libtensor4all_capi.so` from `tensor4all-rs`
- When running `deps/build.jl` directly, use `julia --startup-file=no --project=. deps/build.jl`
- Rust source resolution: `TENSOR4ALL_RS_PATH` env var > sibling `../tensor4all-rs/` > GitHub clone
- Tests: `Pkg.test()` or `julia test/runtests.jl` (requires ITensors.jl in test deps)
- Skip HDF5 tests: `T4A_SKIP_HDF5_TESTS=1`

## PR Checklist

- Before opening or updating a PR, review `README.md` and update user-facing examples and workflow notes if the API or expected commands changed.
- Run `julia --project=docs docs/make.jl` and verify documentation builds without errors or missing docstring warnings.
