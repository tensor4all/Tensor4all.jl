"""
    SkeletonPhaseError(message)

Legacy compatibility error type for code paths that are present in the public
API but not yet implemented during the frontend restoration.

# Examples
```jldoctest
julia> using Tensor4all

julia> err = SkeletonPhaseError("not ready yet");

julia> sprint(showerror, err)
"not ready yet"
```
"""
struct SkeletonPhaseError <: Exception
    message::String
end

Base.showerror(io::IO, err::SkeletonPhaseError) = print(io, err.message)

"""
    SkeletonNotImplemented(api, layer)

Legacy compatibility error type raised by public APIs that are available for
integration but not yet implemented.

# Examples
```jldoctest
julia> using Tensor4all

julia> err = SkeletonNotImplemented(:contract, :core);

julia> sprint(showerror, err)
"Tensor4all transitional API marker: `contract` in the `core` layer is not yet implemented."
```
"""
struct SkeletonNotImplemented <: Exception
    api::Symbol
    layer::Symbol
end

Base.showerror(io::IO, err::SkeletonNotImplemented) = print(
    io,
    "Tensor4all transitional API marker: `",
    err.api,
    "` in the `",
    err.layer,
    "` layer is not yet implemented.",
)

"""
    BackendUnavailableError(message)

Raised when a backend-backed operation is requested but the `tensor4all-rs`
shared library is not available.

# Examples
```jldoctest
julia> using Tensor4all

julia> err = BackendUnavailableError("backend missing");

julia> sprint(showerror, err)
"backend missing"
```
"""
struct BackendUnavailableError <: Exception
    message::String
end

Base.showerror(io::IO, err::BackendUnavailableError) = print(io, err.message)
