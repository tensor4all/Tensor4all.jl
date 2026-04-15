"""
    apply(op, tt; kwargs...)

Apply a `LinearOperator` to a `TensorTrain`.

This call remains deferred until the TensorNetworks operator backend is wired.
"""
apply(::LinearOperator, ::TensorTrain; kwargs...) = throw(SkeletonNotImplemented(:apply, :tt))

"""
    save_as_mps(args...; kwargs...)

Persist a `TensorTrain` through the HDF5-backed MPS compatibility layer.
"""
save_as_mps(args...; kwargs...) = throw(
    BackendUnavailableError("`save_as_mps` requires the HDF5 extension. Load `HDF5.jl` and retry."),
)

"""
    load_tt(args...; kwargs...)

Load a `TensorTrain` through the HDF5-backed MPS compatibility layer.
"""
load_tt(args...; kwargs...) = throw(
    BackendUnavailableError("`load_tt` requires the HDF5 extension. Load `HDF5.jl` and retry."),
)
