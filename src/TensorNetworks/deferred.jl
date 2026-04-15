replace_siteinds_part!(::TensorTrain, args...; kwargs...) = throw(
    SkeletonNotImplemented(Symbol("replace_siteinds_part!"), :tt),
)

rearrange_siteinds(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:rearrange_siteinds, :tt))
makesitediagonal(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:makesitediagonal, :tt))
extractdiagonal(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:extractdiagonal, :tt))

apply(::LinearOperator, ::TensorTrain; kwargs...) = throw(SkeletonNotImplemented(:apply, :tt))

save_as_mps(args...; kwargs...) = throw(
    BackendUnavailableError("`save_as_mps` requires the HDF5 extension. Load `HDF5.jl` and retry."),
)
load_tt(args...; kwargs...) = throw(
    BackendUnavailableError("`load_tt` requires the HDF5 extension. Load `HDF5.jl` and retry."),
)
