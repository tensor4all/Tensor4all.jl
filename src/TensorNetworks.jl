module TensorNetworks

using ..Tensor4all: Tensor

export TensorTrain

mutable struct TensorTrain
    data::Vector{Tensor}
    llim::Int
    rlim::Int
end

Base.length(tt::TensorTrain) = length(tt.data)

end
