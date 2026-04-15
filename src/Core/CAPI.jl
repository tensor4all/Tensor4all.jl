const _StatusCode = Cint
const _T4A_SUCCESS = _StatusCode(0)
const _T4A_BUFFER_TOO_SMALL = _StatusCode(-5)

const _T4A_STORAGE_DENSE_F64 = Cint(0)
const _T4A_STORAGE_DENSE_C64 = Cint(1)
const _T4A_STORAGE_DIAG_F64 = Cint(2)
const _T4A_STORAGE_DIAG_C64 = Cint(3)

_capi_symbol(symbol::Symbol) = Libdl.dlsym(require_backend(), symbol)

function _last_error_message()
    out_len = Ref{Csize_t}(0)
    status = ccall(
        _capi_symbol(:t4a_last_error_message),
        _StatusCode,
        (Ptr{UInt8}, Csize_t, Ref{Csize_t}),
        C_NULL,
        0,
        out_len,
    )
    status == _T4A_SUCCESS || return "no backend error available"
    buf = Vector{UInt8}(undef, out_len[])
    status = ccall(
        _capi_symbol(:t4a_last_error_message),
        _StatusCode,
        (Ptr{UInt8}, Csize_t, Ref{Csize_t}),
        buf,
        length(buf),
        out_len,
    )
    status == _T4A_SUCCESS || return "no backend error available"
    nul = findfirst(==(0x00), buf)
    return String(isnothing(nul) ? buf : buf[1:(nul - 1)])
end

function _throw_last_error(context::AbstractString)
    msg = _last_error_message()
    isempty(msg) && (msg = "no backend error available")
    throw(ErrorException("tensor4all-capi failure in $(context): $(msg)"))
end

function _check_status(status::_StatusCode, context::AbstractString)
    status == _T4A_SUCCESS && return
    _throw_last_error(context)
end

function _check_ptr(ptr::Ptr{Cvoid}, context::AbstractString)
    ptr != C_NULL && return ptr
    _throw_last_error(context)
end

function _query_cstring(symbol::Symbol, ptr::Ptr{Cvoid}, context::AbstractString)
    out_len = Ref{Csize_t}(0)
    status = ccall(
        _capi_symbol(symbol),
        _StatusCode,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ref{Csize_t}),
        ptr,
        C_NULL,
        0,
        out_len,
    )
    _check_status(status, context)
    buf = Vector{UInt8}(undef, out_len[])
    status = ccall(
        _capi_symbol(symbol),
        _StatusCode,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ref{Csize_t}),
        ptr,
        buf,
        length(buf),
        out_len,
    )
    _check_status(status, context)
    nul = findfirst(==(0x00), buf)
    raw = isnothing(nul) ? buf : buf[1:(nul - 1)]
    return String(raw)
end

function _release_handle!(release_symbol::Symbol, obj, field::Symbol)
    ptr = getfield(obj, field)
    ptr == C_NULL && return
    ccall(_capi_symbol(release_symbol), Cvoid, (Ptr{Cvoid},), ptr)
    setfield!(obj, field, C_NULL)
    return
end
