"""
    linsolve(op::LinearOperator, rhs::TensorTrain;
             init::Union{TensorTrain, Nothing} = nothing,
             a0::Real = 0.0,
             a1::Real = 1.0,
             center_vertex::Integer = 1,
             rtol::Real = 0.0,
             cutoff::Real = 0.0,
             maxdim::Integer = 0,
             form::Symbol = :unitary,
             nfullsweeps::Integer = 5,
             krylov_tol::Real = 1e-12,
             krylov_maxiter::Integer = 30,
             krylov_dim::Integer = 30,
             convergence_tol::Real = 0.0) -> TensorTrain

Solve the chain-form TT linear system `(a0 * I + a1 * A) x = b` where the
operator `A` is `op`, the right-hand side `b` is `rhs`, and the returned
`x` is a `TensorTrain` over the same site indices as `init`. With the
default coefficients (`a0 = 0`, `a1 = 1`) this reduces to `A x = b`.

The optional `init` argument provides the starting guess for the
variational sweep; if omitted, `rhs` is used. The current backend
sweep-based linsolve assumes that `init` and `rhs` share the same true
site-index set; mixed input / output spaces are not yet supported.

# Keyword arguments

- `a0`, `a1`: coefficients in `(a0 * I + a1 * A) x = b`. Default
  `a0 = 0`, `a1 = 1`.
- `center_vertex`: 1-based starting vertex for the sweep.
- `rtol`, `cutoff`, `maxdim`, `form`: truncation controls applied to the
  intermediate solution after each local update. See [`truncate`](@ref).
- `nfullsweeps`: number of full sweeps over the chain.
- `krylov_tol`, `krylov_maxiter`, `krylov_dim`: GMRES-like local-solver
  controls per sweep step.
- `convergence_tol`: residual-based early termination tolerance.
  `0.0` (default) disables early termination.

The keyword defaults mirror the backend defaults; tighten `nfullsweeps`
or `convergence_tol` for production use.
"""
function linsolve(
    op::LinearOperator,
    rhs::TensorTrain;
    init::Union{TensorTrain, Nothing}=nothing,
    a0::Real=0.0,
    a1::Real=1.0,
    center_vertex::Integer=1,
    rtol::Real=0.0,
    cutoff::Real=0.0,
    maxdim::Integer=0,
    form::Symbol=:unitary,
    nfullsweeps::Integer=5,
    krylov_tol::Real=1.0e-12,
    krylov_maxiter::Integer=30,
    krylov_dim::Integer=30,
    convergence_tol::Real=0.0,
)
    isempty(rhs.data) && throw(ArgumentError("rhs TensorTrain must not be empty"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative, got $rtol"))
    cutoff >= 0 || throw(ArgumentError("cutoff must be nonnegative, got $cutoff"))
    maxdim >= 0 || throw(ArgumentError("maxdim must be nonnegative, got $maxdim"))
    nfullsweeps >= 1 || throw(ArgumentError("nfullsweeps must be >= 1, got $nfullsweeps"))
    krylov_tol > 0 || throw(ArgumentError("krylov_tol must be positive, got $krylov_tol"))
    krylov_maxiter >= 1 || throw(ArgumentError("krylov_maxiter must be >= 1, got $krylov_maxiter"))
    krylov_dim >= 1 || throw(ArgumentError("krylov_dim must be >= 1, got $krylov_dim"))
    convergence_tol >= 0 || throw(ArgumentError("convergence_tol must be nonnegative, got $convergence_tol"))
    isfinite(a0) || throw(ArgumentError("a0 must be finite, got $a0"))
    isfinite(a1) || throw(ArgumentError("a1 must be finite, got $a1"))

    mpo = op.mpo
    mpo === nothing && throw(ArgumentError("LinearOperator.mpo must be set before linsolve"))
    init_tt = init === nothing ? rhs : init
    isempty(init_tt.data) && throw(ArgumentError("init TensorTrain must not be empty"))
    1 <= center_vertex <= length(init_tt) || throw(ArgumentError(
        "center_vertex must be in 1:$(length(init_tt)), got $center_vertex",
    ))

    rhs_sites = _state_site_order(rhs)
    init_sites = _state_site_order(init_tt)
    _validate_operator_chain!(mpo, op.input_indices, op.output_indices)

    true_inputs = _bound_operator_indices(op.true_input, "op.true_input")
    true_outputs = _bound_operator_indices(op.true_output, "op.true_output")
    _validate_operator_spaces!(op.input_indices, op.output_indices, true_inputs, true_outputs)
    mapped_positions = _mapped_state_positions(true_inputs, rhs_sites)

    scalar_kind = _promoted_scalar_kind(rhs, init_tt, mpo)
    operator_handle = _new_treetn_handle(mpo, scalar_kind)
    rhs_handle = _new_treetn_handle(rhs, scalar_kind)
    init_handle = _new_treetn_handle(init_tt, scalar_kind)
    true_input_handles = Ptr{Cvoid}[]
    internal_input_handles = Ptr{Cvoid}[]
    true_output_handles = Ptr{Cvoid}[]
    internal_output_handles = Ptr{Cvoid}[]
    result_handle = C_NULL

    try
        for index in true_inputs
            push!(true_input_handles, _new_index_handle(index))
        end
        for index in op.input_indices
            push!(internal_input_handles, _new_index_handle(index))
        end
        for index in true_outputs
            push!(true_output_handles, _new_index_handle(index))
        end
        for index in op.output_indices
            push!(internal_output_handles, _new_index_handle(index))
        end

        result_ref = Ref{Ptr{Cvoid}}(C_NULL)
        mapped_positions_c = Csize_t[(position - 1) for position in mapped_positions]
        status = ccall(
            _t4a(:t4a_treetn_linsolve),
            Cint,
            (
                Ptr{Cvoid},
                Ptr{Cvoid},
                Ptr{Cvoid},
                Csize_t,
                Ptr{Csize_t},
                Csize_t,
                Ptr{Ptr{Cvoid}},
                Ptr{Ptr{Cvoid}},
                Ptr{Ptr{Cvoid}},
                Ptr{Ptr{Cvoid}},
                Cdouble,
                Cdouble,
                Csize_t,
                Cint,
                Csize_t,
                Cdouble,
                Csize_t,
                Csize_t,
                Cdouble,
                Cdouble,
                Cdouble,
                Ref{Ptr{Cvoid}},
            ),
            operator_handle,
            rhs_handle,
            init_handle,
            Csize_t(center_vertex - 1),
            mapped_positions_c,
            length(mapped_positions_c),
            true_input_handles,
            internal_input_handles,
            true_output_handles,
            internal_output_handles,
            float(rtol),
            float(cutoff),
            Csize_t(maxdim),
            _canonical_form_code(form),
            Csize_t(nfullsweeps),
            float(krylov_tol),
            Csize_t(krylov_maxiter),
            Csize_t(krylov_dim),
            float(a0),
            float(a1),
            float(convergence_tol),
            result_ref,
        )
        _check_backend_status(status, "linear-solving (a0 + a1 * A) x = b on TensorTrain")
        result_handle = result_ref[]
        return _treetn_from_handle(result_handle)
    finally
        _release_treetn_handle(result_handle)
        for handle in internal_output_handles
            _release_index_handle(handle)
        end
        for handle in true_output_handles
            _release_index_handle(handle)
        end
        for handle in internal_input_handles
            _release_index_handle(handle)
        end
        for handle in true_input_handles
            _release_index_handle(handle)
        end
        _release_treetn_handle(init_handle)
        _release_treetn_handle(rhs_handle)
        _release_treetn_handle(operator_handle)
    end
end
