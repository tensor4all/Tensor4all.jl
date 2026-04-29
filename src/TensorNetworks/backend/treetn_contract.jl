function _factorize_alg_code(alg::Symbol)
    alg === :svd && return _T4A_FACTORIZE_ALG_SVD
    alg === :qr && return _T4A_FACTORIZE_ALG_QR
    alg === :lu && return _T4A_FACTORIZE_ALG_LU
    alg === :ci && return _T4A_FACTORIZE_ALG_CI
    throw(ArgumentError("unknown factorize_alg $alg. Expected :svd, :qr, :lu, or :ci"))
end

"""
    contract(a::TensorTrain, b::TensorTrain;
             method::Symbol = :zipup,
             threshold::Real = 0.0,
             maxdim::Integer = 0,
             svd_policy=nothing,
             nfullsweeps::Integer = 0,
             convergence_tol::Real = 0.0,
             factorize_alg::Symbol = :svd,
             qr_rtol::Real = 0.0) -> TensorTrain

Contract two `TensorTrain`s over their shared site indices using the backend
TreeTN contraction kernel (`t4a_treetn_contract`).

# Keyword arguments

- `method`: contraction algorithm. One of `:zipup` (default), `:fit`, `:naive`.
- `threshold`, `maxdim`, `svd_policy`: truncation contract. See the
  Truncation Policy chapter of the docs.
- `nfullsweeps`: for `method=:fit`, number of variational full sweeps.
  `0` (default) lets the backend pick (currently 1).
- `convergence_tol`: for `method=:fit`, early-termination tolerance.
  `0.0` (default) disables early termination.
- `factorize_alg`: factorization used for the contract step. One of `:svd`
  (default), `:qr`, `:lu`, `:ci`.
- `qr_rtol`: relative tolerance for the QR step when `factorize_alg=:qr`.
  `0.0` (default) lets the backend pick.

Returns a new `TensorTrain`. Throws `ArgumentError` for invalid arguments.
"""
function contract(
    a::TensorTrain,
    b::TensorTrain;
    method::Symbol=:zipup,
    threshold::Union{Nothing,Real}=nothing,
    maxdim::Union{Nothing,Integer}=nothing,
    svd_policy::Union{Nothing, SvdTruncationPolicy}=nothing,
    nfullsweeps::Integer=0,
    convergence_tol::Real=0.0,
    factorize_alg::Symbol=:svd,
    qr_rtol::Real=0.0,
)
    isempty(a.data) && throw(ArgumentError("Left TensorTrain must not be empty for contract"))
    isempty(b.data) && throw(ArgumentError("Right TensorTrain must not be empty for contract"))
    threshold_value = _normalize_threshold(threshold)
    maxdim_value = _normalize_maxdim(maxdim)
    nfullsweeps >= 0 || throw(ArgumentError("nfullsweeps must be nonnegative, got $nfullsweeps"))
    convergence_tol >= 0 || throw(ArgumentError("convergence_tol must be nonnegative, got $convergence_tol"))
    qr_rtol >= 0 || throw(ArgumentError("qr_rtol must be nonnegative, got $qr_rtol"))

    method_code = _contract_method_code(method)
    factorize_code = _factorize_alg_code(factorize_alg)
    ffi_policy = _resolve_svd_policy(; threshold=threshold_value, svd_policy)

    scalar_kind = _promoted_scalar_kind(a, b)
    a_handle = _new_treetn_handle(a, scalar_kind)
    b_handle = _new_treetn_handle(b, scalar_kind)
    result_handle = C_NULL
    try
        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = _with_svd_policy_ptr(ffi_policy) do policy_ptr
            ccall(
                _t4a(:t4a_treetn_contract),
                Cint,
                (
                    Ptr{Cvoid},
                    Ptr{Cvoid},
                    Cint,
                    Ptr{Cvoid},
                    Csize_t,
                    Csize_t,
                    Cdouble,
                    Cint,
                    Cdouble,
                    Ref{Ptr{Cvoid}},
                ),
                a_handle,
                b_handle,
                method_code,
                policy_ptr,
                Csize_t(maxdim_value),
                Csize_t(nfullsweeps),
                float(convergence_tol),
                factorize_code,
                float(qr_rtol),
                out,
            )
        end
        _check_backend_status(status, "contracting two TensorTrains")
        result_handle = out[]
        return _treetn_from_handle(result_handle)
    finally
        _release_treetn_handle(result_handle)
        _release_treetn_handle(b_handle)
        _release_treetn_handle(a_handle)
    end
end
