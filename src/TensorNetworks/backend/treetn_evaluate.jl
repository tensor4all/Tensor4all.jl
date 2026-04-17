"""
    evaluate(tt::TensorTrain, indices::Vector{Index},
             values::AbstractMatrix{<:Integer}) -> Vector{ComplexF64}

Evaluate `tt` at one or more points using the backend
`t4a_treetn_evaluate` kernel. Each column of `values` is one point.

# Arguments

- `indices`: the site indices addressed by the rows of `values`. Each entry
  must be a site index of `tt`.
- `values`: integer matrix of shape `(length(indices), n_points)`. Entries
  are **1-based** (Julia convention); the wrapper converts to the 0-based
  convention expected by the C API.

Returns a `Vector{ComplexF64}` of length `n_points`. For real-valued TTs
the imaginary parts are zero.

Throws `ArgumentError` for empty `tt`, mismatched shapes, or out-of-range
values.
"""
function evaluate(
    tt::TensorTrain,
    indices::Vector{Index},
    values::AbstractMatrix{<:Integer},
)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty for evaluate"))
    isempty(indices) && throw(ArgumentError("indices must not be empty for evaluate"))
    n_indices, n_points = size(values)
    n_indices == length(indices) || throw(DimensionMismatch(
        "values must have $(length(indices)) rows (one per index), got $n_indices",
    ))
    n_points >= 1 || throw(ArgumentError("values must contain at least one point"))

    # Validate values are within each index's dimension and convert to 0-based.
    values_c = Vector{Csize_t}(undef, n_indices * n_points)
    for col in 1:n_points, row in 1:n_indices
        v = Int(values[row, col])
        d = dim(indices[row])
        (1 <= v <= d) || throw(ArgumentError(
            "values[$row, $col] = $v is out of range for index of dimension $d",
        ))
        values_c[(col - 1) * n_indices + row] = Csize_t(v - 1)
    end

    scalar_kind = _promoted_scalar_kind(tt)
    tt_handle = _new_treetn_handle(tt, scalar_kind)
    index_handles = Ptr{Cvoid}[]
    try
        for index in indices
            push!(index_handles, _new_index_handle(index))
        end

        out_re = Vector{Float64}(undef, n_points)
        out_im = Vector{Float64}(undef, n_points)
        status = ccall(
            _t4a(:t4a_treetn_evaluate),
            Cint,
            (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Csize_t, Ptr{Csize_t}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}),
            tt_handle,
            index_handles,
            Csize_t(n_indices),
            values_c,
            Csize_t(n_points),
            out_re,
            out_im,
        )
        _check_backend_status(status, "evaluating TensorTrain")
        return ComplexF64.(out_re, out_im)
    finally
        for handle in index_handles
            _release_index_handle(handle)
        end
        _release_treetn_handle(tt_handle)
    end
end

"""
    evaluate(tt::TensorTrain, indices::Vector{Index},
             values::AbstractVector{<:Integer}) -> ComplexF64

Single-point convenience method. Equivalent to evaluating with a 1-column
matrix and returning the scalar.
"""
function evaluate(
    tt::TensorTrain,
    indices::Vector{Index},
    values::AbstractVector{<:Integer},
)
    isempty(indices) && throw(ArgumentError("indices must not be empty for evaluate"))
    length(values) == length(indices) || throw(DimensionMismatch(
        "values must have $(length(indices)) entries (one per index), got $(length(values))",
    ))
    matrix = reshape(collect(values), length(indices), 1)
    return evaluate(tt, indices, matrix)[1]
end
