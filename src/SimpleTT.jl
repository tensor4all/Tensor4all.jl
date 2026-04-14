module SimpleTT

export TensorTrain

using LinearAlgebra
import TensorCrossInterpolation: rrlu, MatrixLUCI, left, right, npivots

mutable struct TensorTrain{T,N}
    sitetensors::Vector{Array{T,N}}
end

Base.length(tt::TensorTrain) = length(tt.sitetensors)

_getindex(x, indices) = ntuple(i -> x[indices[i]], length(indices))

function _contract(
    a::AbstractArray{T1,N1},
    b::AbstractArray{T2,N2},
    idx_a::NTuple{n1,Int},
    idx_b::NTuple{n2,Int},
) where {T1,T2,N1,N2,n1,n2}
    length(idx_a) == length(idx_b) || error("length(idx_a) != length(idx_b)")
    length(unique(idx_a)) == length(idx_a) || error("idx_a contains duplicate elements")
    length(unique(idx_b)) == length(idx_b) || error("idx_b contains duplicate elements")
    all(1 <= idx <= N1 for idx in idx_a) || error("idx_a contains elements out of range")
    all(1 <= idx <= N2 for idx in idx_b) || error("idx_b contains elements out of range")

    rest_idx_a = setdiff(1:N1, idx_a)
    rest_idx_b = setdiff(1:N2, idx_b)

    amat = reshape(
        permutedims(a, (rest_idx_a..., idx_a...)),
        prod(_getindex(size(a), rest_idx_a)),
        prod(_getindex(size(a), idx_a)),
    )
    bmat = reshape(
        permutedims(b, (idx_b..., rest_idx_b...)),
        prod(_getindex(size(b), idx_b)),
        prod(_getindex(size(b), rest_idx_b)),
    )

    return reshape(
        amat * bmat,
        _getindex(size(a), rest_idx_a)...,
        _getindex(size(b), rest_idx_b)...,
    )
end

function _contractsitetensors(a::Array{T,4}, b::Array{T,4})::Array{T,4} where {T}
    ab::Array{T,6} = _contract(a, b, (3,), (2,))
    abpermuted = permutedims(ab, (1, 4, 2, 5, 3, 6))
    return reshape(
        abpermuted,
        size(a, 1) * size(b, 1),
        size(a, 2),
        size(b, 3),
        size(a, 4) * size(b, 4),
    )
end

function _check_mpo_contract_compatibility(a::TensorTrain{T,4}, b::TensorTrain{T,4}) where {T}
    length(a) == length(b) || throw(ArgumentError("Tensor trains must have the same length."))
    for n in 1:length(a)
        size(a.sitetensors[n], 3) == size(b.sitetensors[n], 2) ||
            error("Tensor trains must share the identical index at n=$(n)!")
    end
    return nothing
end

function contract_naive(
    a::TensorTrain{T,4},
    b::TensorTrain{T,4};
    tolerance=0.0,
    maxbonddim=typemax(Int),
) where {T}
    _check_mpo_contract_compatibility(a, b)
    tt = TensorTrain{T,4}(_contractsitetensors.(a.sitetensors, b.sitetensors))
    if tolerance > 0 || maxbonddim < typemax(Int)
        compress!(tt, :SVD; tolerance=tolerance, maxbonddim=maxbonddim)
    end
    return tt
end

function contract_zipup(
    a::TensorTrain{T,4},
    b::TensorTrain{T,4};
    tolerance=1e-12,
    method::Symbol=:SVD,
    maxbonddim::Int=typemax(Int),
    kwargs...,
) where {T}
    isempty(kwargs) ||
        throw(ArgumentError("Unsupported keyword arguments for zipup contraction: $(collect(keys(kwargs)))"))
    _check_mpo_contract_compatibility(a, b)
    R::Array{T,3} = ones(T, 1, 1, 1)
    sitetensors = Vector{Array{T,4}}(undef, length(a))

    for n in 1:length(a)
        RA = _contract(R, a.sitetensors[n], (2,), (1,))
        C = permutedims(_contract(RA, b.sitetensors[n], (2, 4), (1, 2)), (1, 2, 4, 3, 5))
        if n == length(a)
            sitetensors[n] = reshape(C, size(C)[1:3]..., 1)
            break
        end

        left, right, newbonddim = _factorize(
            reshape(C, prod(size(C)[1:3]), prod(size(C)[4:5])),
            method;
            tolerance=tolerance,
            maxbonddim=maxbonddim,
        )
        sitetensors[n] = reshape(left, size(C)[1:3]..., newbonddim)
        R = reshape(right, newbonddim, size(C)[4:5]...)
    end

    return TensorTrain{T,4}(sitetensors)
end

"""
    contract(a::TensorTrain{T,4}, b::TensorTrain{T,4}; algorithm=:naive, tolerance=1e-12,
             maxbonddim=typemax(Int), kwargs...)

Contract two MPO-like `SimpleTT.TensorTrain` objects.
"""
function contract(
    a::TensorTrain{T,4},
    b::TensorTrain{T,4};
    algorithm::Symbol=:naive,
    tolerance=1e-12,
    maxbonddim::Int=typemax(Int),
    kwargs...,
) where {T}
    if algorithm === :naive
        return contract_naive(a, b; tolerance=tolerance, maxbonddim=maxbonddim)
    elseif algorithm === :zipup
        return contract_zipup(a, b; tolerance=tolerance, maxbonddim=maxbonddim, kwargs...)
    else
        throw(ArgumentError("Unknown algorithm $algorithm."))
    end
end

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
