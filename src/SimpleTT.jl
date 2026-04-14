module SimpleTT

export TensorTrain

using LinearAlgebra
import TensorCrossInterpolation: rrlu, MatrixLUCI, left, right, npivots

mutable struct TensorTrain{T,N}
    sitetensors::Vector{Array{T,N}}
end

Base.length(tt::TensorTrain) = length(tt.sitetensors)

function _factorize(
    A::AbstractMatrix{V},
    method::Symbol;
    tolerance::Float64,
    maxbonddim::Int,
    leftorthogonal::Bool=false,
) where {V}
    if method === :LU
        factorization = rrlu(
            A;
            reltol=tolerance,
            abstol=0.0,
            maxrank=maxbonddim,
            leftorthogonal=leftorthogonal,
        )
        return left(factorization), right(factorization), npivots(factorization)
    elseif method === :CI
        factorization = MatrixLUCI(
            A;
            reltol=tolerance,
            abstol=0.0,
            maxrank=maxbonddim,
            leftorthogonal=leftorthogonal,
        )
        return left(factorization), right(factorization), npivots(factorization)
    elseif method === :SVD
        factorization = svd(A)
        singvals = factorization.S
        totalnorm = sum(abs2, singvals)
        cutrank = min(length(singvals), maxbonddim)
        if totalnorm > 0
            for k in 1:length(singvals)
                tailnorm = sum(abs2, @view(singvals[k+1:end]))
                if tailnorm <= tolerance^2 * totalnorm
                    cutrank = min(k, maxbonddim)
                    break
                end
            end
        end
        if leftorthogonal
            return (
                factorization.U[:, 1:cutrank],
                Diagonal(singvals[1:cutrank]) * factorization.Vt[1:cutrank, :],
                cutrank,
            )
        else
            return (
                factorization.U[:, 1:cutrank] * Diagonal(singvals[1:cutrank]),
                factorization.Vt[1:cutrank, :],
                cutrank,
            )
        end
    else
        throw(ArgumentError("Unsupported factorization method $method"))
    end
end

function compress!(
    tt::TensorTrain{T,N},
    method::Symbol=:LU;
    tolerance::Float64=1e-12,
    maxbonddim::Int=typemax(Int),
) where {T,N}
    for ell in 1:length(tt)-1
        shapel = size(tt.sitetensors[ell])
        left, right, newbonddim = _factorize(
            reshape(tt.sitetensors[ell], prod(shapel[1:end-1]), shapel[end]),
            method;
            tolerance=0.0,
            maxbonddim=typemax(Int),
            leftorthogonal=true,
        )
        tt.sitetensors[ell] = Array(reshape(left, shapel[1:end-1]..., newbonddim))
        shaper = size(tt.sitetensors[ell+1])
        nexttensor = right * reshape(tt.sitetensors[ell+1], shaper[1], prod(shaper[2:end]))
        tt.sitetensors[ell+1] = Array(reshape(nexttensor, newbonddim, shaper[2:end]...))
    end

    for ell in length(tt):-1:2
        shaper = size(tt.sitetensors[ell])
        left, right, newbonddim = _factorize(
            reshape(tt.sitetensors[ell], shaper[1], prod(shaper[2:end])),
            method;
            tolerance=tolerance,
            maxbonddim=maxbonddim,
            leftorthogonal=false,
        )
        tt.sitetensors[ell] = Array(reshape(right, newbonddim, shaper[2:end]...))
        shapel = size(tt.sitetensors[ell-1])
        nexttensor = reshape(tt.sitetensors[ell-1], prod(shapel[1:end-1]), shapel[end]) * left
        tt.sitetensors[ell-1] = Array(reshape(nexttensor, shapel[1:end-1]..., newbonddim))
    end

    nothing
end

end
