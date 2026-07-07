# Build script for tensor4all-capi Rust library
#
# This script builds the Rust library and copies it to the deps directory.
# Run via `Pkg.build("Tensor4all")` or `julia deps/build.jl`.
#
# Rust source resolution (in priority order):
#   1. TENSOR4ALL_RS_PATH environment variable
#   2. Sibling directory ../tensor4all-rs/ (relative to package root)
#   3. Persistent cached clone under deps/.tensor4all-rs/
#
# Optional linear algebra backend selection:
#   TENSOR4ALL_LINALG_BACKEND=julia-blas  # default: inject Julia BLAS/LAPACK pointers
#   TENSOR4ALL_LINALG_BACKEND=faer        # Rust/faer backend
#
# Optional extra Cargo feature selection:
#   TENSOR4ALL_RS_FEATURES="feature-a,feature-b"

using Libdl
using RustToolChain: cargo

# Configuration. The pinned tensor4all-rs commit lives in `deps/TENSOR4ALL_RS_PIN`
# so that `deps/build.jl` and the GitHub workflows (.github/workflows/*.yml)
# share a single source of truth — bump the pin in that one file.
const TENSOR4ALL_RS_PIN_FILE = joinpath(@__DIR__, "TENSOR4ALL_RS_PIN")
const TENSOR4ALL_RS_FALLBACK_COMMIT = strip(read(TENSOR4ALL_RS_PIN_FILE, String))
const TENSOR4ALL_RS_REPO = "https://github.com/tensor4all/tensor4all-rs.git"

# Paths
const DEPS_DIR = @__DIR__
const PACKAGE_DIR = dirname(DEPS_DIR)
const SIBLING_RUST_DIR = joinpath(dirname(PACKAGE_DIR), "tensor4all-rs")
const CACHED_RUST_DIR = joinpath(DEPS_DIR, ".tensor4all-rs")

# Installed library name (platform-specific). Copied into `deps/` after build.
const LIB_NAME = "libtensor4all_capi." * Libdl.dlext

# Cargo cdylib base name (`tensor4all-capi` crate -> `tensor4all_capi` artifact).
const CARGO_CDYLIB_BASE_NAME = "tensor4all_capi"

"""
    find_rust_source() -> Union{String, Nothing}

Find the tensor4all-rs source directory.
Returns the path if found locally, or `nothing` if GitHub clone is needed.
"""
function find_rust_source()
    # Priority 1: Environment variable
    env_path = get(ENV, "TENSOR4ALL_RS_PATH", nothing)
    if env_path !== nothing && isdir(env_path)
        cargo_toml = joinpath(env_path, "Cargo.toml")
        if isfile(cargo_toml)
            return env_path
        else
            @warn "TENSOR4ALL_RS_PATH is set but Cargo.toml not found" path=env_path
        end
    end

    # Priority 2: Sibling directory
    if isdir(SIBLING_RUST_DIR)
        cargo_toml = joinpath(SIBLING_RUST_DIR, "Cargo.toml")
        if isfile(cargo_toml)
            return SIBLING_RUST_DIR
        end
    end

    # No local source found
    return nothing
end

"""
    clone_from_github(dest::String)

Clone tensor4all-rs from GitHub to the specified directory.
"""
function clone_from_github(dest::String)
    println("Cloning tensor4all-rs from GitHub (pinned commit: $TENSOR4ALL_RS_FALLBACK_COMMIT)...")
    run(`git clone $TENSOR4ALL_RS_REPO $dest`)
    run(`git -C $dest checkout --detach $TENSOR4ALL_RS_FALLBACK_COMMIT`)
end

function git_rev_parse(dir::String, ref::String)
    return strip(read(`git -C $dir rev-parse $ref`, String))
end

function cached_rust_source_at_pin(dir::String)
    if !isdir(joinpath(dir, ".git")) || !isfile(joinpath(dir, "Cargo.toml"))
        return false
    end
    head = git_rev_parse(dir, "HEAD")
    pinned = git_rev_parse(dir, TENSOR4ALL_RS_FALLBACK_COMMIT)
    return head == pinned
end

"""
    ensure_github_clone() -> String

Ensure a pinned tensor4all-rs checkout exists under `deps/.tensor4all-rs/`.

The clone is kept on disk so rebuilds reuse Cargo artifacts and so Windows
application-control policies that block executables under `%TEMP%` do not break
`cargo` build scripts.
"""
function ensure_github_clone()
    dest = CACHED_RUST_DIR

    if isdir(dest)
        try
            if cached_rust_source_at_pin(dest)
                println("Using cached Rust source: $dest")
                return dest
            end
            println(
                "Updating cached Rust source to pinned commit " *
                "$TENSOR4ALL_RS_FALLBACK_COMMIT...",
            )
            run(`git -C $dest fetch origin`)
            run(`git -C $dest checkout --detach $TENSOR4ALL_RS_FALLBACK_COMMIT`)
            return dest
        catch err
            @warn "Cached Rust source unusable; recloning" path=dest exception=err
            rm(dest; recursive=true, force=true)
        end
    end

    clone_from_github(dest)
    return dest
end

function _split_cargo_features(raw::AbstractString)
    return filter(!isempty, split(strip(raw), r"[\s,]+"))
end

function selected_linalg_backend()
    raw_env = get(ENV, "TENSOR4ALL_LINALG_BACKEND", "julia-blas")
    raw = lowercase(strip(raw_env))
    normalized = replace(raw, "_" => "-")
    if normalized in ("faer", "cpu-faer")
        return :faer
    elseif isempty(normalized) ||
           normalized in ("julia-blas", "blas", "lapack", "inject", "provider-inject")
        return :julia_blas
    end

    error(
        "invalid TENSOR4ALL_LINALG_BACKEND=$(repr(raw)); expected `faer` " *
        "or `julia-blas`",
    )
end

function _dedup_preserve_order(features)
    out = String[]
    seen = Set{String}()
    for feature in features
        feature_str = String(feature)
        if !(feature_str in seen)
            push!(out, feature_str)
            push!(seen, feature_str)
        end
    end
    return out
end

function linalg_backend_label(backend::Symbol = selected_linalg_backend())
    return backend == :julia_blas ? "julia-blas" : String(backend)
end

function cargo_feature_args()
    backend = selected_linalg_backend()
    extra_features = _split_cargo_features(get(ENV, "TENSOR4ALL_RS_FEATURES", ""))

    if backend == :faer && isempty(extra_features)
        return String[]
    end

    backend_features =
        backend == :faer ? ["tenferro-cpu-faer"] : ["tenferro-provider-inject"]
    features = _dedup_preserve_order(vcat(backend_features, extra_features))

    return String["--no-default-features", "--features", join(features, ",")]
end

"""
    cargo_built_library_candidates(rust_dir::String, profile::String) -> Vector{String}

Return possible `cargo build` output paths for the `tensor4all-capi` cdylib.

Rust prefixes dynamic libraries with `lib` on Unix but not on Windows.
"""
function cargo_built_library_candidates(rust_dir::String, profile::String)
    target_dir = joinpath(rust_dir, "target", profile)
    ext = Libdl.dlext
    candidates = String[joinpath(target_dir, LIB_NAME)]
    if Sys.iswindows()
        push!(candidates, joinpath(target_dir, "$CARGO_CDYLIB_BASE_NAME.$ext"))
    end
    return candidates
end

function find_built_library(rust_dir::String, profile::String)
    for path in cargo_built_library_candidates(rust_dir, profile)
        if isfile(path)
            return path
        end
    end
    searched = join(["  $path" for path in cargo_built_library_candidates(rust_dir, profile)], "\n")
    error("Built library not found. Searched:\n$searched")
end

"""
    build_library(rust_dir::String)

Build tensor4all-capi in the specified Rust workspace directory.

Set `TENSOR4ALL_BUILD_DEBUG=1` to build in debug mode (unoptimized, with full
debug symbols).  The default is a release build.
"""
function build_library(rust_dir::String)
    is_debug = get(ENV, "TENSOR4ALL_BUILD_DEBUG", "") in ("1", "true", "yes")
    profile = is_debug ? "debug" : "release"

    println("Building tensor4all-capi ($profile)...")
    println("  Rust source: $rust_dir")
    println("  Using cargo from RustToolChain.jl")
    println("  Linear algebra backend: ", linalg_backend_label())
    feature_args = cargo_feature_args()
    if !isempty(feature_args)
        println("  Cargo features: ", join(feature_args, " "))
    end

    cd(rust_dir) do
        if is_debug
            run(`$(cargo()) build -p tensor4all-capi $feature_args`)
        else
            run(`$(cargo()) build -p tensor4all-capi --release $feature_args`)
        end
    end

    return find_built_library(rust_dir, profile)
end

"""
    install_library(src_lib::String)

Copy the built library to the deps directory.
"""
function install_library(src_lib::String)
    dst_lib = joinpath(DEPS_DIR, LIB_NAME)
    println("Installing library...")
    println("  $src_lib -> $dst_lib")
    cp(src_lib, dst_lib; force=true)
    println("Build complete!")
    println("Library installed to: $dst_lib")
end

function main()
    rust_dir = find_rust_source()

    if rust_dir === nothing
        rust_dir = ensure_github_clone()
    else
        println("Using local Rust source: $rust_dir")
    end

    src_lib = build_library(rust_dir)
    install_library(src_lib)
end

# Run build unless a test includes this file for configuration-only checks.
if get(ENV, "TENSOR4ALL_BUILD_CONFIG_ONLY", "") != "1"
    main()
end
