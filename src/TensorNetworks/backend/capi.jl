const _T4A_SUCCESS = Cint(0)
const _T4A_SCALAR_KIND_F64 = Cint(0)
const _T4A_SCALAR_KIND_C64 = Cint(1)

const _T4A_CONTRACT_METHOD_ZIPUP = Cint(0)
const _T4A_CONTRACT_METHOD_FIT = Cint(1)
const _T4A_CONTRACT_METHOD_NAIVE = Cint(2)

const _T4A_QTT_LAYOUT_INTERLEAVED = Cint(1)
const _T4A_QTT_LAYOUT_FUSED = Cint(2)

const _T4A_BC_PERIODIC = Cint(0)
const _T4A_BC_OPEN = Cint(1)

_t4a(symbol::Symbol) = Libdl.dlsym(require_backend(), symbol)

function _bc_code(bc::Symbol)
    bc === :periodic && return _T4A_BC_PERIODIC
    bc === :open && return _T4A_BC_OPEN
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
        _t4a(:t4a_qtt_layout_new),
        Cint,
        (Cint, Csize_t, Ptr{Csize_t}, Ref{Ptr{Cvoid}}),
        _T4A_QTT_LAYOUT_FUSED,
        Csize_t(nvars),
        res_c,
        out,
    )
    _check_backend_status(status, "creating QTT layout")
    return out[]
end

function _release_qtt_layout_handle(ptr::Ptr{Cvoid})
    ptr == C_NULL && return nothing
    ccall(_t4a(:t4a_qtt_layout_release), Cvoid, (Ptr{Cvoid},), ptr)
    return nothing
end

function _c_string_from_buffer(buf::Vector{UInt8})
    nul = findfirst(==(0x00), buf)
    if nul === nothing
        return String(copy(buf))
    elseif nul == 1
        return ""
    else
        return String(copy(buf[1:nul-1]))
    end
end

function _last_backend_error_message()
    out_len = Ref{Csize_t}(0)
    status = ccall(
        _t4a(:t4a_last_error_message),
        Cint,
        (Ptr{UInt8}, Csize_t, Ref{Csize_t}),
        C_NULL,
        0,
        out_len,
    )
    status == _T4A_SUCCESS || return "backend error (status $status)"
    out_len[] == 0 && return "backend error"

    buffer = Vector{UInt8}(undef, Int(out_len[]))
    status = ccall(
        _t4a(:t4a_last_error_message),
        Cint,
        (Ptr{UInt8}, Csize_t, Ref{Csize_t}),
        buffer,
        length(buffer),
        out_len,
    )
    status == _T4A_SUCCESS || return "backend error (status $status)"
    return _c_string_from_buffer(buffer)
end

function _throw_backend_status(status::Integer, context::AbstractString)
    message = _last_backend_error_message()
    lowered = lowercase(message)
    if occursin("dimension", lowered) || occursin("mismatch", lowered)
        throw(DimensionMismatch("$context failed: $message"))
    end
    throw(ArgumentError("$context failed: $message"))
end

function _check_backend_status(status::Integer, context::AbstractString)
    status == _T4A_SUCCESS && return nothing
    _throw_backend_status(status, context)
end

function _release_index_handle(ptr::Ptr{Cvoid})
    ptr == C_NULL && return nothing
    ccall(_t4a(:t4a_index_release), Cvoid, (Ptr{Cvoid},), ptr)
    return nothing
end

function _release_tensor_handle(ptr::Ptr{Cvoid})
    ptr == C_NULL && return nothing
    ccall(_t4a(:t4a_tensor_release), Cvoid, (Ptr{Cvoid},), ptr)
    return nothing
end

function _release_treetn_handle(ptr::Ptr{Cvoid})
    ptr == C_NULL && return nothing
    ccall(_t4a(:t4a_treetn_release), Cvoid, (Ptr{Cvoid},), ptr)
    return nothing
end

function _split_tags_csv(csv::AbstractString)
    isempty(csv) && return String[]
    return split(csv, ',')
end

function _new_index_handle(index::Index)
    tags_csv = join(tags(index), ",")
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(
        _t4a(:t4a_index_new_with_id),
        Cint,
        (Csize_t, UInt64, Cstring, Int64, Ref{Ptr{Cvoid}}),
        dim(index),
        id(index),
        tags_csv,
        plev(index),
        out,
    )
    _check_backend_status(status, "creating backend index $index")
    return out[]
end

function _index_tags_from_handle(ptr::Ptr{Cvoid})
    out_len = Ref{Csize_t}(0)
    status = ccall(
        _t4a(:t4a_index_tags),
        Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ref{Csize_t}),
        ptr,
        C_NULL,
        0,
        out_len,
    )
    _check_backend_status(status, "querying backend index tags")

    buffer = Vector{UInt8}(undef, Int(out_len[]))
    status = ccall(
        _t4a(:t4a_index_tags),
        Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ref{Csize_t}),
        ptr,
        buffer,
        length(buffer),
        out_len,
    )
    _check_backend_status(status, "copying backend index tags")
    return _split_tags_csv(_c_string_from_buffer(buffer))
end

function _index_from_handle(ptr::Ptr{Cvoid})
    out_dim = Ref{Csize_t}(0)
    out_id = Ref{UInt64}(0)
    out_plev = Ref{Int64}(0)

    status = ccall(_t4a(:t4a_index_dim), Cint, (Ptr{Cvoid}, Ref{Csize_t}), ptr, out_dim)
    _check_backend_status(status, "querying backend index dimension")

    status = ccall(_t4a(:t4a_index_id), Cint, (Ptr{Cvoid}, Ref{UInt64}), ptr, out_id)
    _check_backend_status(status, "querying backend index id")

    status = ccall(_t4a(:t4a_index_plev), Cint, (Ptr{Cvoid}, Ref{Int64}), ptr, out_plev)
    _check_backend_status(status, "querying backend index prime level")

    return Index(
        Int(out_dim[]);
        tags=_index_tags_from_handle(ptr),
        plev=Int(out_plev[]),
        id=out_id[],
    )
end
