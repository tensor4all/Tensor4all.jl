function _bound_operator_indices(indices::Vector{Union{Index, Nothing}}, label::AbstractString)
    bound = Index[]
    for (n, index) in pairs(indices)
        index === nothing && throw(
            ArgumentError("$label[$n] is not bound. Call set_input_space!, set_output_space!, or set_iospaces! first."),
        )
        push!(bound, index)
    end
    return bound
end

function _state_site_order(tt::TensorTrain)
    isempty(tt.data) && throw(ArgumentError("state TensorTrain must not be empty"))

    siteinds_by_tensor = _siteinds_by_tensor(tt)
    ordered = Index[]
    for (position, siteinds) in pairs(siteinds_by_tensor)
        length(siteinds) == 1 || throw(
            ArgumentError("state tensor $position must have exactly one site index, got $(length(siteinds))"),
        )
        push!(ordered, only(siteinds))
    end

    length(Set(ordered)) == length(ordered) || throw(
        ArgumentError("state TensorTrain must use unique site indices"),
    )
    return ordered
end

function _validate_operator_chain!(
    mpo::TensorTrain,
    input_indices::Vector{Index},
    output_indices::Vector{Index},
)
    isempty(mpo.data) && throw(ArgumentError("operator mpo must not be empty"))
    length(input_indices) == length(mpo) || throw(
        ArgumentError("expected $(length(mpo)) operator input indices, got $(length(input_indices))"),
    )
    length(output_indices) == length(mpo) || throw(
        ArgumentError("expected $(length(mpo)) operator output indices, got $(length(output_indices))"),
    )

    siteinds_by_tensor = _siteinds_by_tensor(mpo)
    for position in eachindex(siteinds_by_tensor)
        siteinds = siteinds_by_tensor[position]
        length(siteinds) == 2 || throw(
            ArgumentError("operator tensor $position must have exactly two site indices, got $(length(siteinds))"),
        )

        input_index = input_indices[position]
        output_index = output_indices[position]
        input_index == output_index && throw(
            ArgumentError("operator tensor $position reuses the same index for input and output"),
        )
        input_index in siteinds || throw(
            ArgumentError("operator input index $input_index is not attached to tensor $position"),
        )
        output_index in siteinds || throw(
            ArgumentError("operator output index $output_index is not attached to tensor $position"),
        )
    end

    length(Set(input_indices)) == length(input_indices) || throw(
        ArgumentError("operator input indices must be unique"),
    )
    length(Set(output_indices)) == length(output_indices) || throw(
        ArgumentError("operator output indices must be unique"),
    )
    return nothing
end

function _validate_operator_spaces!(
    internal_inputs::Vector{Index},
    internal_outputs::Vector{Index},
    true_inputs::Vector{Index},
    true_outputs::Vector{Index},
)
    length(true_inputs) == length(internal_inputs) || throw(
        ArgumentError("expected $(length(internal_inputs)) bound input indices, got $(length(true_inputs))"),
    )
    length(true_outputs) == length(internal_outputs) || throw(
        ArgumentError("expected $(length(internal_outputs)) bound output indices, got $(length(true_outputs))"),
    )

    length(Set(true_inputs)) == length(true_inputs) || throw(
        ArgumentError("bound input indices must be unique"),
    )
    length(Set(true_outputs)) == length(true_outputs) || throw(
        ArgumentError("bound output indices must be unique"),
    )

    for n in eachindex(internal_inputs)
        dim(internal_inputs[n]) == dim(true_inputs[n]) || throw(
            DimensionMismatch(
                "operator input index $n has dimension $(dim(internal_inputs[n])) but bound input has dimension $(dim(true_inputs[n]))",
            ),
        )
        dim(internal_outputs[n]) == dim(true_outputs[n]) || throw(
            DimensionMismatch(
                "operator output index $n has dimension $(dim(internal_outputs[n])) but bound output has dimension $(dim(true_outputs[n]))",
            ),
        )
    end
    return nothing
end

function _mapped_state_positions(true_inputs::Vector{Index}, state_sites::Vector{Index})
    positions = Dict(index => position for (position, index) in pairs(state_sites))
    mapped = Int[]
    for input_index in true_inputs
        position = get(positions, input_index, nothing)
        position === nothing && throw(
            ArgumentError("bound input index $input_index is not a site index of the state TensorTrain"),
        )
        push!(mapped, position)
    end

    for n in 2:length(mapped)
        mapped[n] > mapped[n - 1] || throw(
            ArgumentError(
                "operator sites must follow state chain order. Got mapped tensor positions $mapped",
            ),
        )
    end
    return mapped
end

function _contract_method_code(method::Symbol)
    if method === :zipup
        return _T4A_CONTRACT_METHOD_ZIPUP
    elseif method === :fit
        return _T4A_CONTRACT_METHOD_FIT
    elseif method === :naive
        return _T4A_CONTRACT_METHOD_NAIVE
    end
    throw(ArgumentError("unknown apply method $method. Expected :zipup, :fit, or :naive"))
end

"""
    apply(op, state; method=:zipup, threshold=0.0, maxdim=0,
          svd_policy=nothing, nfullsweeps=0, convergence_tol=0.0)

Apply a chain-compatible `LinearOperator` to a chain `TensorTrain`.

# Keyword arguments

- `method`: contraction algorithm. One of `:zipup` (default), `:fit`, or `:naive`.
- `threshold`, `maxdim`, `svd_policy`: truncation contract. See the
  Truncation Policy chapter of the docs.
- `nfullsweeps`: for `method=:fit`, number of variational full sweeps.
  `0` (default) lets the backend pick (currently 1).
- `convergence_tol`: for `method=:fit`, early-termination tolerance.
  `0.0` (default) disables early termination.

The current backend path expects one site index per state tensor and exactly
one input/output site-index pair per operator tensor. Bind explicit spaces
with `set_input_space!`, `set_output_space!`, or `set_iospaces!` before
calling `apply`.
"""
function apply(
    op::LinearOperator,
    state::TensorTrain;
    method::Symbol=:zipup,
    threshold::Union{Nothing,Real}=nothing,
    maxdim::Union{Nothing,Integer}=nothing,
    svd_policy::Union{Nothing, SvdTruncationPolicy}=nothing,
    nfullsweeps::Integer=0,
    convergence_tol::Real=0.0,
)
    threshold_value = _normalize_threshold(threshold)
    maxdim_value = _normalize_maxdim(maxdim)
    nfullsweeps >= 0 || throw(ArgumentError("nfullsweeps must be nonnegative, got $nfullsweeps"))
    convergence_tol >= 0 || throw(ArgumentError("convergence_tol must be nonnegative, got $convergence_tol"))

    mpo = op.mpo
    mpo === nothing && throw(ArgumentError("LinearOperator.mpo must be set before apply"))
    state_sites = _state_site_order(state)
    _validate_operator_chain!(mpo, op.input_indices, op.output_indices)

    true_inputs = _bound_operator_indices(op.true_input, "op.true_input")
    true_outputs = _bound_operator_indices(op.true_output, "op.true_output")
    _validate_operator_spaces!(op.input_indices, op.output_indices, true_inputs, true_outputs)
    mapped_positions = _mapped_state_positions(true_inputs, state_sites)

    ffi_policy = _resolve_svd_policy(; threshold=threshold_value, svd_policy)

    scalar_kind = _promoted_scalar_kind(state, mpo)
    state_handle = _new_treetn_handle(state, scalar_kind)
    mpo_handle = _new_treetn_handle(mpo, scalar_kind)
    input_handles = Ptr{Cvoid}[]
    output_handles = Ptr{Cvoid}[]
    true_output_handles = Ptr{Cvoid}[]
    result_handle = C_NULL

    try
        for index in op.input_indices
            push!(input_handles, _new_index_handle(index))
        end
        for index in op.output_indices
            push!(output_handles, _new_index_handle(index))
        end
        for index in true_outputs
            push!(true_output_handles, _new_index_handle(index))
        end

        result_ref = Ref{Ptr{Cvoid}}(C_NULL)
        mapped_positions_c = Csize_t[(position - 1) for position in mapped_positions]
        status = _with_svd_policy_ptr(ffi_policy) do policy_ptr
            ccall(
                _t4a(:t4a_treetn_apply_operator_chain),
                Cint,
                (
                    Ptr{Cvoid},
                    Ptr{Cvoid},
                    Ptr{Csize_t},
                    Csize_t,
                    Ptr{Ptr{Cvoid}},
                    Ptr{Ptr{Cvoid}},
                    Ptr{Ptr{Cvoid}},
                    Cint,
                    Ptr{Cvoid},
                    Csize_t,
                    Csize_t,
                    Cdouble,
                    Ref{Ptr{Cvoid}},
                ),
                mpo_handle,
                state_handle,
                mapped_positions_c,
                length(mapped_positions_c),
                input_handles,
                output_handles,
                true_output_handles,
                _contract_method_code(method),
                policy_ptr,
                Csize_t(maxdim_value),
                Csize_t(nfullsweeps),
                float(convergence_tol),
                result_ref,
            )
        end
        _check_backend_status(status, "applying LinearOperator")
        result_handle = result_ref[]
        return _treetn_from_handle(result_handle)
    finally
        _release_treetn_handle(result_handle)
        for handle in true_output_handles
            _release_index_handle(handle)
        end
        for handle in output_handles
            _release_index_handle(handle)
        end
        for handle in input_handles
            _release_index_handle(handle)
        end
        _release_treetn_handle(mpo_handle)
        _release_treetn_handle(state_handle)
    end
end
