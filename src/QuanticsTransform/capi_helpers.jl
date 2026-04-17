function _materialized_linear_operator(treetn_handle::Ptr{Cvoid})
    try
        tt = TensorNetworks._treetn_from_handle(treetn_handle)
        input_indices, output_indices = _extract_mpo_io_indices(tt)
        return TensorNetworks.LinearOperator(;
            mpo=tt,
            input_indices=input_indices,
            output_indices=output_indices,
        )
    finally
        TensorNetworks._release_treetn_handle(treetn_handle)
    end
end

function _materialize_single_target(materialize::Function, context::AbstractString)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = materialize(out)
    TensorNetworks._check_backend_status(status, context)
    return _materialized_linear_operator(out[])
end

function _bc_code(bc::Symbol)
    bc === :periodic && return TensorNetworks._T4A_BC_PERIODIC
    bc === :open && return TensorNetworks._T4A_BC_OPEN
    throw(ArgumentError("unknown boundary condition $bc. Expected :periodic or :open"))
end

function _new_qtt_layout_handle(nvars::Integer, resolutions::Vector{<:Integer})
    nvars > 0 || throw(ArgumentError("nvars must be positive, got $nvars"))
    length(resolutions) == nvars || throw(
        DimensionMismatch("expected $nvars variable resolutions, got $(length(resolutions))"),
    )
    all(>(0), resolutions) || throw(
        ArgumentError("variable resolutions must all be positive, got $resolutions"),
    )
    length(unique(resolutions)) == 1 || throw(
        ArgumentError("fused QTT layouts require all variable resolutions to match, got $resolutions"),
    )

    out = Ref{Ptr{Cvoid}}(C_NULL)
    res_c = Csize_t[Csize_t(r) for r in resolutions]
    status = ccall(
        TensorNetworks._t4a(:t4a_qtt_layout_new),
        Cint,
        (Cint, Csize_t, Ptr{Csize_t}, Ref{Ptr{Cvoid}}),
        TensorNetworks._T4A_QTT_LAYOUT_FUSED,
        Csize_t(nvars),
        res_c,
        out,
    )
    TensorNetworks._check_backend_status(status, "creating QTT layout")
    return out[]
end

function _release_qtt_layout_handle(ptr::Ptr{Cvoid})
    ptr == C_NULL && return nothing
    ccall(TensorNetworks._t4a(:t4a_qtt_layout_release), Cvoid, (Ptr{Cvoid},), ptr)
    return nothing
end

function _new_univariate_layout(r::Integer)
    _require_positive_integer("r", r)
    return _new_qtt_layout_handle(1, [r])
end

function _new_multivar_layout(r::Integer, nvars::Integer)
    _require_positive_integer("r", r)
    _require_positive_integer("nvars", nvars)
    return _new_qtt_layout_handle(nvars, fill(r, nvars))
end

function _extract_mpo_io_indices(tt::TensorNetworks.TensorTrain)
    input_indices = Index[]
    output_indices = Index[]
    for site_inds in TensorNetworks.siteinds(tt)
        length(site_inds) == 2 || throw(
            ArgumentError("Expected 2 site indices per MPO tensor, got $(length(site_inds))"),
        )
        push!(output_indices, site_inds[1])
        push!(input_indices, site_inds[2])
    end
    return input_indices, output_indices
end

function _materialize_shift(
    layout_handle::Ptr{Cvoid},
    target_var::Integer,
    offset::Integer,
    bc::Symbol,
    context::AbstractString,
)
    return _materialize_single_target(context) do out
        ccall(
            TensorNetworks._t4a(:t4a_qtransform_shift_materialize),
            Cint,
            (Ptr{Cvoid}, Csize_t, Int64, Cint, Ref{Ptr{Cvoid}}),
            layout_handle,
            Csize_t(target_var),
            Int64(offset),
            _bc_code(bc),
            out,
        )
    end
end

function _materialize_flip(
    layout_handle::Ptr{Cvoid},
    target_var::Integer,
    bc::Symbol,
    context::AbstractString,
)
    return _materialize_single_target(context) do out
        ccall(
            TensorNetworks._t4a(:t4a_qtransform_flip_materialize),
            Cint,
            (Ptr{Cvoid}, Csize_t, Cint, Ref{Ptr{Cvoid}}),
            layout_handle,
            Csize_t(target_var),
            _bc_code(bc),
            out,
        )
    end
end

function _materialize_phase_rotation(
    layout_handle::Ptr{Cvoid},
    target_var::Integer,
    theta,
    context::AbstractString,
)
    return _materialize_single_target(context) do out
        ccall(
            TensorNetworks._t4a(:t4a_qtransform_phase_rotation_materialize),
            Cint,
            (Ptr{Cvoid}, Csize_t, Cdouble, Ref{Ptr{Cvoid}}),
            layout_handle,
            Csize_t(target_var),
            Float64(theta),
            out,
        )
    end
end

function _materialize_cumsum(
    layout_handle::Ptr{Cvoid},
    target_var::Integer,
    context::AbstractString,
)
    return _materialize_single_target(context) do out
        ccall(
            TensorNetworks._t4a(:t4a_qtransform_cumsum_materialize),
            Cint,
            (Ptr{Cvoid}, Csize_t, Ref{Ptr{Cvoid}}),
            layout_handle,
            Csize_t(target_var),
            out,
        )
    end
end

function _materialize_fourier(
    layout_handle::Ptr{Cvoid},
    target_var::Integer,
    forward::Bool,
    maxbonddim::Integer,
    tolerance::Real,
    context::AbstractString,
)
    _require_nonnegative_integer("maxbonddim", maxbonddim)
    _require_nonnegative_real("tolerance", tolerance)

    return _materialize_single_target(context) do out
        ccall(
            TensorNetworks._t4a(:t4a_qtransform_fourier_materialize),
            Cint,
            (Ptr{Cvoid}, Csize_t, Int32, Csize_t, Cdouble, Ref{Ptr{Cvoid}}),
            layout_handle,
            Csize_t(target_var),
            Int32(forward),
            Csize_t(maxbonddim),
            Float64(tolerance),
            out,
        )
    end
end

function _materialize_affine(
    layout_handle::Ptr{Cvoid},
    a_num,
    a_den,
    b_num,
    b_den,
    bc::Symbol,
    context::AbstractString,
)
    _require_nonzero_integer("a_den", a_den)
    _require_nonzero_integer("b_den", b_den)

    a_num_c = Int64[Int64(a_num)]
    a_den_c = Int64[Int64(a_den)]
    b_num_c = Int64[Int64(b_num)]
    b_den_c = Int64[Int64(b_den)]
    bc_c = Cint[_bc_code(bc)]
    return _materialize_single_target(context) do out
        ccall(
            TensorNetworks._t4a(:t4a_qtransform_affine_materialize),
            Cint,
            (
                Ptr{Cvoid},
                Ptr{Int64},
                Ptr{Int64},
                Ptr{Int64},
                Ptr{Int64},
                Csize_t,
                Csize_t,
                Ptr{Cint},
                Ref{Ptr{Cvoid}},
            ),
            layout_handle,
            a_num_c,
            a_den_c,
            b_num_c,
            b_den_c,
            Csize_t(1),
            Csize_t(1),
            bc_c,
            out,
        )
    end
end

function _materialize_binaryop(
    layout_handle::Ptr{Cvoid},
    lhs_var::Integer,
    rhs_var::Integer,
    a1::Integer,
    b1::Integer,
    a2::Integer,
    b2::Integer,
    bc1::Symbol,
    bc2::Symbol,
    context::AbstractString,
)
    for (name, val) in (("a1", a1), ("b1", b1), ("a2", a2), ("b2", b2))
        -128 <= val <= 127 || throw(ArgumentError("$name must fit in Int8, got $val"))
    end
    return _materialize_single_target(context) do out
        ccall(
            TensorNetworks._t4a(:t4a_qtransform_binaryop_materialize),
            Cint,
            (
                Ptr{Cvoid},
                Csize_t,
                Csize_t,
                Int8,
                Int8,
                Int8,
                Int8,
                Cint,
                Cint,
                Ref{Ptr{Cvoid}},
            ),
            layout_handle,
            Csize_t(lhs_var),
            Csize_t(rhs_var),
            Int8(a1),
            Int8(b1),
            Int8(a2),
            Int8(b2),
            _bc_code(bc1),
            _bc_code(bc2),
            out,
        )
    end
end
