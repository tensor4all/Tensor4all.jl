module SimpleTT

export TensorTrain

struct TensorTrain{T,N}
    sitetensors::Vector{Array{T,N}}
end

Base.length(tt::TensorTrain) = length(tt.sitetensors)

end

