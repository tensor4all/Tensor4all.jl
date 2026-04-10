"""
    SkeletonPhaseError(message)

Raised when code expects functionality that is intentionally deferred during the
review-first skeleton phase.

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

Raised by reviewable public APIs that are intentionally present before their
backend behavior is implemented.

# Examples
```jldoctest
julia> using Tensor4all

julia> err = SkeletonNotImplemented(:contract, :core);

julia> sprint(showerror, err)
"Tensor4all skeleton phase: `contract` is planned in the `core` layer but not implemented yet."
```
"""
struct SkeletonNotImplemented <: Exception
    api::Symbol
    layer::Symbol
end

Base.showerror(io::IO, err::SkeletonNotImplemented) = print(
    io,
    "Tensor4all skeleton phase: `",
    err.api,
    "` is planned in the `",
    err.layer,
    "` layer but not implemented yet.",
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
