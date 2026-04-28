"""
    TensorTrainEvaluator(tt::TensorTrain)

Dense Julia-side snapshot for repeated scalar point evaluation of a chain
`TensorTrain`. Each site tensor is copied into a block with logical shape
`(left_link_dim, right_link_dim, fused_site_dim)`.

Mutating the source `TensorTrain` after construction does not update the
snapshot. Only `Float64` and `ComplexF64` tensor data are supported.
"""
struct TensorTrainEvaluator{T}
    blocks::Vector{Vector{T}}
    leftdims::Vector{Int}
    rightdims::Vector{Int}
    site_groups::Vector{Vector{Index}}
    site_dims::Vector{Vector{Int}}
    site_strides::Vector{Vector{Int}}
end

"""
    TensorTrainEvalWorkspace(ev::TensorTrainEvaluator)

Reusable buffers for allocation-light [`evaluate!`](@ref) calls.
"""
struct TensorTrainEvalWorkspace{T}
    buf1::Vector{T}
    buf2::Vector{T}
end

function TensorTrainEvaluator(tt::TensorTrain)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty for TensorTrainEvaluator"))

    T = _evaluator_eltype(tt)
    links = linkinds(tt)
    groups = siteinds(tt)
    _validate_evaluator_site_groups(groups)
    nsites = length(tt)
    blocks = Vector{Vector{T}}(undef, nsites)
    leftdims = Vector{Int}(undef, nsites)
    rightdims = Vector{Int}(undef, nsites)
    site_dims = Vector{Vector{Int}}(undef, nsites)
    site_strides = Vector{Vector{Int}}(undef, nsites)

    for n in eachindex(tt.data)
        tensor = tt.data[n]
        tensor_inds = inds(tensor)
        length(unique(tensor_inds)) == length(tensor_inds) || throw(
            ArgumentError("Tensor $n contains duplicate indices; TensorTrainEvaluator requires unique tensor indices"),
        )

        left_link = n == 1 ? nothing : links[n - 1]
        right_link = n == nsites ? nothing : links[n]
        leftdim = left_link === nothing ? 1 : dim(left_link)
        rightdim = right_link === nothing ? 1 : dim(right_link)
        site_group = groups[n]
        dims_n = dim.(site_group)
        strides_n = _site_group_strides(dims_n)

        target_order = Index[]
        left_link === nothing || push!(target_order, left_link)
        right_link === nothing || push!(target_order, right_link)
        append!(target_order, site_group)
        length(target_order) == rank(tensor) || throw(
            ArgumentError("Tensor $n has unsupported extra indices. Expected links and site group $(target_order), got $(tensor_inds)"),
        )
        data = convert(Array{T}, Array(tensor, target_order...))
        blocks[n] = Vector(reshape(data, leftdim * rightdim * prod(dims_n; init=1)))
        leftdims[n] = leftdim
        rightdims[n] = rightdim
        site_dims[n] = dims_n
        site_strides[n] = strides_n
    end

    return TensorTrainEvaluator{T}(blocks, leftdims, rightdims, groups, site_dims, site_strides)
end

function TensorTrainEvalWorkspace(ev::TensorTrainEvaluator{T}) where {T}
    width = maximum(max.(ev.leftdims, ev.rightdims))
    return TensorTrainEvalWorkspace{T}(Vector{T}(undef, width), Vector{T}(undef, width))
end

function _evaluator_eltype(tt::TensorTrain)
    has_complex = false
    for (n, tensor) in pairs(tt.data)
        T = eltype(tensor)
        if T === ComplexF64
            has_complex = true
        elseif T !== Float64
            throw(ArgumentError("TensorTrainEvaluator supports Float64 and ComplexF64 tensors only; tensor $n has element type $T"))
        end
    end
    return has_complex ? ComplexF64 : Float64
end

function _validate_evaluator_site_groups(groups::Vector{Vector{Index}})
    seen = Set{Index}()
    for (n, group) in pairs(groups)
        isempty(group) && throw(ArgumentError("Tensor $n has no site indices; TensorTrainEvaluator requires at least one site index per tensor"))
        for index in group
            index in seen && throw(ArgumentError("Duplicate site index $index in TensorTrainEvaluator site groups"))
            push!(seen, index)
        end
    end
    return nothing
end

function _site_group_strides(dims::Vector{Int})
    strides = Vector{Int}(undef, length(dims))
    stride = 1
    for n in eachindex(dims)
        strides[n] = stride
        stride *= dims[n]
    end
    return strides
end

function _combined_site_index(ev::TensorTrainEvaluator, n::Int, values::AbstractVector{<:Integer})
    dims_n = ev.site_dims[n]
    length(values) == length(dims_n) || throw(DimensionMismatch(
        "grouped_values[$n] must have $(length(dims_n)) entries for site group $(ev.site_groups[n]), got $(length(values))",
    ))

    p = 1
    strides_n = ev.site_strides[n]
    for k in eachindex(dims_n)
        value = Int(values[k])
        1 <= value <= dims_n[k] || throw(ArgumentError(
            "grouped_values[$n][$k] = $value is out of range for index $(ev.site_groups[n][k]) with dimension $(dims_n[k])",
        ))
        p += (value - 1) * strides_n[k]
    end
    return p
end

"""
    evaluate!(ws, ev, grouped_values)

Evaluate `ev` with reusable workspace buffers. `grouped_values[n]` must
contain one 1-based value per site index in `ev.site_groups[n]`.
"""
function evaluate!(
    ws::TensorTrainEvalWorkspace{T},
    ev::TensorTrainEvaluator{T},
    grouped_values::AbstractVector,
) where {T}
    length(grouped_values) == length(ev.blocks) || throw(DimensionMismatch(
        "grouped_values must contain $(length(ev.blocks)) site groups, got $(length(grouped_values))",
    ))
    length(ws.buf1) >= maximum(ev.leftdims) || throw(DimensionMismatch("workspace buf1 is too small for this evaluator"))
    length(ws.buf2) >= maximum(ev.rightdims) || throw(DimensionMismatch("workspace buf2 is too small for this evaluator"))

    left = ws.buf1
    right = ws.buf2
    left[1] = one(T)

    for n in eachindex(ev.blocks)
        χL = ev.leftdims[n]
        χR = ev.rightdims[n]
        p = _combined_site_index(ev, n, grouped_values[n])
        _gemv_transpose_site_slice!(right, ev.blocks[n], p, χL, χR, left)
        left, right = right, left
    end

    return left[1]
end

"""
    evaluate(ev::TensorTrainEvaluator, grouped_values)

Evaluate `ev` with a temporary workspace. Use [`evaluate!`](@ref) in hot paths.
"""
function evaluate(ev::TensorTrainEvaluator, grouped_values::AbstractVector)
    ws = TensorTrainEvalWorkspace(ev)
    return evaluate!(ws, ev, grouped_values)
end

"""
    evaluate!(ws, ev, site_indices, values)
    evaluate(ev, site_indices, values)

Compatibility API accepting a flat site-index order and one 1-based value per
site index. Matching uses exact `Index ==` semantics.
"""
function evaluate!(
    ws::TensorTrainEvalWorkspace,
    ev::TensorTrainEvaluator,
    site_indices::Vector{Index},
    values::AbstractVector{<:Integer},
)
    grouped_values = _group_evaluator_values(ev, site_indices, values)
    return evaluate!(ws, ev, grouped_values)
end

function evaluate(
    ev::TensorTrainEvaluator,
    site_indices::Vector{Index},
    values::AbstractVector{<:Integer},
)
    ws = TensorTrainEvalWorkspace(ev)
    return evaluate!(ws, ev, site_indices, values)
end

function _group_evaluator_values(
    ev::TensorTrainEvaluator,
    site_indices::Vector{Index},
    values::AbstractVector{<:Integer},
)
    length(site_indices) == length(values) || throw(DimensionMismatch(
        "values must have $(length(site_indices)) entries (one per index), got $(length(values))",
    ))
    seen = Set{Index}()
    value_by_index = Dict{Index,Int}()
    for (index, value) in zip(site_indices, values)
        index in seen && throw(ArgumentError("Duplicate site index $index in site_indices"))
        push!(seen, index)
        value_by_index[index] = Int(value)
    end
    grouped = Vector{Vector{Int}}(undef, length(ev.site_groups))
    for (n, group) in pairs(ev.site_groups)
        grouped[n] = Vector{Int}(undef, length(group))
        for (k, index) in pairs(group)
            haskey(value_by_index, index) || throw(ArgumentError("Missing value for evaluator site index $index"))
            grouped[n][k] = value_by_index[index]
        end
    end
    length(value_by_index) == sum(length, ev.site_groups) || throw(
        ArgumentError("site_indices contains indices that are not present in the evaluator site groups"),
    )
    return grouped
end

const _TTEvalBlasInt = LinearAlgebra.BLAS.BlasInt
const _TTEvalIncOne = _TTEvalBlasInt(1)
function _gemv_transpose_site_slice!(
    y::Vector{Float64},
    block::Vector{Float64},
    p::Int,
    χL::Int,
    χR::Int,
    x::Vector{Float64},
)
    trans = UInt8('T')
    m = _TTEvalBlasInt(χL)
    n = _TTEvalBlasInt(χR)
    lda = _TTEvalBlasInt(χL)
    alpha = 1.0
    beta = 0.0
    offset = (p - 1) * χL * χR + 1
    ccall(
        (LinearAlgebra.BLAS.@blasfunc(dgemv_), LinearAlgebra.BLAS.libblastrampoline),
        Cvoid,
        (Ref{UInt8}, Ref{_TTEvalBlasInt}, Ref{_TTEvalBlasInt}, Ref{Float64}, Ptr{Float64}, Ref{_TTEvalBlasInt}, Ptr{Float64}, Ref{_TTEvalBlasInt}, Ref{Float64}, Ptr{Float64}, Ref{_TTEvalBlasInt}),
        trans,
        m,
        n,
        alpha,
        pointer(block, offset),
        lda,
        pointer(x),
        _TTEvalIncOne,
        beta,
        pointer(y),
        _TTEvalIncOne,
    )
    return y
end

function _gemv_transpose_site_slice!(
    y::Vector{ComplexF64},
    block::Vector{ComplexF64},
    p::Int,
    χL::Int,
    χR::Int,
    x::Vector{ComplexF64},
)
    trans = UInt8('T')
    m = _TTEvalBlasInt(χL)
    n = _TTEvalBlasInt(χR)
    lda = _TTEvalBlasInt(χL)
    alpha = one(ComplexF64)
    beta = zero(ComplexF64)
    offset = (p - 1) * χL * χR + 1
    ccall(
        (LinearAlgebra.BLAS.@blasfunc(zgemv_), LinearAlgebra.BLAS.libblastrampoline),
        Cvoid,
        (Ref{UInt8}, Ref{_TTEvalBlasInt}, Ref{_TTEvalBlasInt}, Ref{ComplexF64}, Ptr{ComplexF64}, Ref{_TTEvalBlasInt}, Ptr{ComplexF64}, Ref{_TTEvalBlasInt}, Ref{ComplexF64}, Ptr{ComplexF64}, Ref{_TTEvalBlasInt}),
        trans,
        m,
        n,
        alpha,
        pointer(block, offset),
        lda,
        pointer(x),
        _TTEvalIncOne,
        beta,
        pointer(y),
        _TTEvalIncOne,
    )
    return y
end
