const _backend_handle = Ref{Ptr{Cvoid}}(C_NULL)

backend_library_name() = "libtensor4all_capi." * Libdl.dlext

"""
    backend_library_path()

Return the expected path to the `tensor4all-rs` C API shared library.

This helper is backend-free and can be used during the skeleton phase to
inspect where the package would look for the compiled library.

# Examples
```jldoctest
julia> using Tensor4all

julia> backend_library_path() isa String
true
```
"""
function backend_library_path()
    return get(
        ENV,
        "TENSOR4ALL_CAPI_PATH",
        normpath(joinpath(@__DIR__, "..", "..", "deps", backend_library_name())),
    )
end

"""
    require_backend()

Load and return the `tensor4all-rs` shared library handle.

This function is part of the skeleton API surface, but it only succeeds when a
compiled backend is available.

# Examples
```julia
julia> require_backend()
ERROR: BackendUnavailableError(...)
```
"""
function require_backend()
    path = backend_library_path()
    isfile(path) || throw(BackendUnavailableError(
        "tensor4all-rs backend unavailable at `$path`. Run `julia --startup-file=no --project=. deps/build.jl` or set `TENSOR4ALL_CAPI_PATH`.",
    ))
    if _backend_handle[] == C_NULL
        _backend_handle[] = Libdl.dlopen(path)
    end
    return _backend_handle[]
end
