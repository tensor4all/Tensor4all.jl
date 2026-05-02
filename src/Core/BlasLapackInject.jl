const _inject_registration_lock = ReentrantLock()
const _inject_registration_done = Ref(false)

const _CBLAS_GEMM_INJECT_SYMBOLS = (:sgemm, :dgemm, :cgemm, :zgemm)
const _LAPACK_INJECT_SYMBOLS = (
    :dgesvd, :zgesvd,
    :dgeqrf, :zgeqrf,
    :dorgqr, :zungqr,
    :dtrtrs, :ztrtrs,
    :dpotrf, :zpotrf,
    :dgetrf, :zgetrf,
    :dsyev, :zheev,
    :dgeev, :zgeev,
    :dgetc2, :zgetc2,
    :dgesc2, :zgesc2,
)

_inject_blas_interface() = LinearAlgebra.BLAS.USE_BLAS64 ? :ilp64 : :lp64

function _dlsym_or_nothing(handle::Ptr{Cvoid}, name::Symbol)
    ptr = Libdl.dlsym_e(handle, name)
    return ptr == C_NULL ? C_NULL : ptr
end

function _inject_registration_available(handle::Ptr{Cvoid})::Bool
    interface = _inject_blas_interface()
    check_sym = Symbol("cblas_inject_register_dgemm_", interface)
    return _dlsym_or_nothing(handle, check_sym) != C_NULL
end

function _inject_provider_pointer(name::Symbol, interface::Symbol)::Ptr{Cvoid}
    return LinearAlgebra.BLAS.lbt_get_forward(string(name, "_"), interface, :plain)
end

function _inject_missing_provider_pointer(ptr::Ptr{Cvoid})::Bool
    return ptr == C_NULL || UInt(ptr) == typemax(UInt)
end

function _register_inject_symbol!(
    handle::Ptr{Cvoid},
    register_name::Symbol,
    provider_name::Symbol,
    interface::Symbol,
)
    register_func = _dlsym_or_nothing(handle, register_name)
    if register_func == C_NULL
        throw(BackendUnavailableError(
            "inject registration: backend exports provider-inject support but " *
            "is missing expected registration symbol `$(register_name)`.",
        ))
    end
    provider_ptr = _inject_provider_pointer(provider_name, interface)
    if _inject_missing_provider_pointer(provider_ptr)
        throw(BackendUnavailableError(
            "inject registration: provider pointer for `$(provider_name)` " *
            "(interface `$(interface)`) is null. " *
            "Ensure Julia is linked with a BLAS/LAPACK provider that exports `$(provider_name)_`.",
        ))
    end
    status = ccall(register_func, Cint, (Ptr{Cvoid},), provider_ptr)
    if status != 0 && status != 2
        throw(BackendUnavailableError(
            "inject registration: `$(register_name)` returned status $(status) " *
            "for provider `$(provider_name)` (interface `$(interface)`). " *
            "Expected 0 (success) or 2 (already registered).",
        ))
    end
    return nothing
end

function _register_blas_lapack_provider_if_available!(handle::Ptr{Cvoid})
    _inject_registration_done[] && return
    lock(_inject_registration_lock) do
        _inject_registration_done[] && return
        interface = _inject_blas_interface()
        if !_inject_registration_available(handle)
            return
        end
        for name in _CBLAS_GEMM_INJECT_SYMBOLS
            register_name = Symbol("cblas_inject_register_", name, "_", interface)
            _register_inject_symbol!(handle, register_name, name, interface)
        end
        for name in _LAPACK_INJECT_SYMBOLS
            register_name = Symbol("register_", name, "_", interface)
            _register_inject_symbol!(handle, register_name, name, interface)
        end
        _inject_registration_done[] = true
    end
    return nothing
end
