"""
    Tensor4all

`Tensor4all.jl` is in an API-skeleton review phase.

The previous implementation has been intentionally removed so the package can be
rebuilt against the design documents in `docs/design/` without stale public
surface area leaking into the rework. Importing the package is expected to work,
but the real backend-facing types and operations are deferred to a later phase.
"""
module Tensor4all

const SKELETON_PHASE = true

"""
    SkeletonPhaseError(message)

Raised when code expects functionality that is intentionally deferred during the
review-first reset.
"""
struct SkeletonPhaseError <: Exception
    message::String
end

Base.showerror(io::IO, err::SkeletonPhaseError) = print(io, err.message)

end
