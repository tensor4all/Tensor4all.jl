# AGENTS.md

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
- Rust source resolution: `TENSOR4ALL_RS_PATH` env var > sibling `../tensor4all-rs/` > GitHub clone
- Tests: `Pkg.test()` or `julia test/runtests.jl` (requires ITensors.jl in test deps)
- Skip HDF5 tests: `T4A_SKIP_HDF5_TESTS=1`
