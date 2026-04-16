function _new_treetn_handle(tt::TensorTrain, scalar_kind::Symbol)
    tensor_handles = Ptr{Cvoid}[]
    try
        for tensor in tt.data
            push!(tensor_handles, _new_tensor_handle(tensor, scalar_kind))
        end

        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            _t4a(:t4a_treetn_new),
            Cint,
            (Ptr{Ptr{Cvoid}}, Csize_t, Ref{Ptr{Cvoid}}),
            tensor_handles,
            length(tensor_handles),
            out,
        )
        _check_backend_status(status, "creating backend TensorTrain")
        return out[]
    finally
        for handle in tensor_handles
            _release_tensor_handle(handle)
        end
    end
end

function _treetn_num_vertices(ptr::Ptr{Cvoid})
    out_n = Ref{Csize_t}(0)
    status = ccall(
        _t4a(:t4a_treetn_num_vertices),
        Cint,
        (Ptr{Cvoid}, Ref{Csize_t}),
        ptr,
        out_n,
    )
    _check_backend_status(status, "querying backend TensorTrain length")
    return Int(out_n[])
end

function _treetn_tensor_handle(ptr::Ptr{Cvoid}, vertex::Integer)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(
        _t4a(:t4a_treetn_tensor),
        Cint,
        (Ptr{Cvoid}, Csize_t, Ref{Ptr{Cvoid}}),
        ptr,
        vertex,
        out,
    )
    _check_backend_status(status, "querying backend tensor at vertex $(vertex + 1)")
    return out[]
end

function _treetn_canonical_region(ptr::Ptr{Cvoid})
    out_len = Ref{Csize_t}(0)
    status = ccall(
        _t4a(:t4a_treetn_canonical_region),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ref{Csize_t}),
        ptr,
        C_NULL,
        Csize_t(0),
        out_len,
    )
    _check_backend_status(status, "querying backend canonical region length")

    nvertices = Int(out_len[])
    nvertices == 0 && return Int[]

    vertices = Vector{Csize_t}(undef, nvertices)
    status = ccall(
        _t4a(:t4a_treetn_canonical_region),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ref{Csize_t}),
        ptr,
        vertices,
        Csize_t(nvertices),
        out_len,
    )
    _check_backend_status(status, "copying backend canonical region")
    return Int.(vertices)
end

function _derive_llim_rlim(canonical_region::Vector{Int}, ntensors::Int)
    if length(canonical_region) == 1
        center = only(canonical_region)
        return center, center + 2
    end
    return 0, ntensors + 1
end

const _T4A_CANONICAL_FORM_UNITARY = Cint(0)
const _T4A_CANONICAL_FORM_LU = Cint(1)

function _canonical_form_code(form::Symbol)
    form === :unitary && return _T4A_CANONICAL_FORM_UNITARY
    form === :lu && return _T4A_CANONICAL_FORM_LU
    throw(ArgumentError("unknown canonical form $form. Expected :unitary or :lu"))
end

function _treetn_scale(tt::TensorTrain, re::Float64, im::Float64)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty for scalar multiply"))

    scalar_kind = im == 0.0 ? _promoted_scalar_kind(tt) : :c64
    tt_handle = _new_treetn_handle(tt, scalar_kind)
    result_handle = C_NULL
    try
        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            _t4a(:t4a_treetn_scale),
            Cint,
            (Ptr{Cvoid}, Cdouble, Cdouble, Ref{Ptr{Cvoid}}),
            tt_handle,
            re,
            im,
            out,
        )
        _check_backend_status(status, "scaling TensorTrain")
        result_handle = out[]
        return _treetn_from_handle(result_handle)
    finally
        _release_treetn_handle(result_handle)
        _release_treetn_handle(tt_handle)
    end
end

function _validate_tt_binary(a::TensorTrain, b::TensorTrain, op::AbstractString)
    isempty(a.data) && throw(ArgumentError("TensorTrain must not be empty for $op"))
    isempty(b.data) && throw(ArgumentError("TensorTrain must not be empty for $op"))
    length(a) == length(b) || throw(
        DimensionMismatch("$op requires equal length TensorTrains, got $(length(a)) and $(length(b))"),
    )

    a_siteinds = _siteinds_by_tensor(a)
    b_siteinds = _siteinds_by_tensor(b)
    for position in eachindex(a_siteinds)
        a_siteinds[position] == b_siteinds[position] && continue
        throw(
            ArgumentError(
                "$op requires matching site indices at tensor $position, got $(a_siteinds[position]) and $(b_siteinds[position])",
            ),
        )
    end
    return nothing
end

function _treetn_from_handle(ptr::Ptr{Cvoid})
    ntensors = _treetn_num_vertices(ptr)
    tensors = Tensor[]
    for vertex in 0:(ntensors - 1)
        tensor_handle = _treetn_tensor_handle(ptr, vertex)
        try
            push!(tensors, _tensor_from_handle(tensor_handle))
        finally
            _release_tensor_handle(tensor_handle)
        end
    end

    canonical_region = _treetn_canonical_region(ptr)
    llim, rlim = _derive_llim_rlim(canonical_region, ntensors)
    return TensorTrain(tensors, llim, rlim)
end

"""
    orthogonalize(tt, site; form=:unitary)

Orthogonalize `tt` to the requested one-based `site` using the backend chain
canonicalization routine.
"""
function orthogonalize(tt::TensorTrain, site::Integer; form::Symbol=:unitary)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty"))
    1 <= site <= length(tt) || throw(ArgumentError("site must be in 1:$(length(tt)), got $site"))

    scalar_kind = _promoted_scalar_kind(tt)
    tt_handle = _new_treetn_handle(tt, scalar_kind)
    try
        status = ccall(
            _t4a(:t4a_treetn_orthogonalize),
            Cint,
            (Ptr{Cvoid}, Csize_t, Cint),
            tt_handle,
            Csize_t(site - 1),
            _canonical_form_code(form),
        )
        _check_backend_status(status, "orthogonalizing TensorTrain to site $site")
        return _treetn_from_handle(tt_handle)
    finally
        _release_treetn_handle(tt_handle)
    end
end

"""
    truncate(tt; rtol=0.0, cutoff=0.0, maxdim=0)

Truncate TensorTrain bond dimensions with backend truncation controls.
"""
function truncate(tt::TensorTrain; rtol::Real=0.0, cutoff::Real=0.0, maxdim::Integer=0)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative, got $rtol"))
    cutoff >= 0 || throw(ArgumentError("cutoff must be nonnegative, got $cutoff"))
    maxdim >= 0 || throw(ArgumentError("maxdim must be nonnegative, got $maxdim"))
    (rtol == 0.0 && cutoff == 0.0 && maxdim == 0) && throw(
        ArgumentError("At least one of rtol, cutoff, or maxdim must be specified"),
    )

    scalar_kind = _promoted_scalar_kind(tt)
    tt_handle = _new_treetn_handle(tt, scalar_kind)
    try
        status = ccall(
            _t4a(:t4a_treetn_truncate),
            Cint,
            (Ptr{Cvoid}, Cdouble, Cdouble, Csize_t),
            tt_handle,
            float(rtol),
            float(cutoff),
            Csize_t(maxdim),
        )
        _check_backend_status(status, "truncating TensorTrain")
        return _treetn_from_handle(tt_handle)
    finally
        _release_treetn_handle(tt_handle)
    end
end

"""
    add(a, b; rtol=0.0, cutoff=0.0, maxdim=0)

Add two TensorTrain chains, optionally applying backend truncation controls.
"""
function add(a::TensorTrain, b::TensorTrain; rtol::Real=0.0, cutoff::Real=0.0, maxdim::Integer=0)
    _validate_tt_binary(a, b, "add")

    scalar_kind = _promoted_scalar_kind(a, b)
    a_handle = _new_treetn_handle(a, scalar_kind)
    b_handle = _new_treetn_handle(b, scalar_kind)
    result_handle = C_NULL
    try
        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            _t4a(:t4a_treetn_add),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Cdouble, Cdouble, Csize_t, Ref{Ptr{Cvoid}}),
            a_handle,
            b_handle,
            float(rtol),
            float(cutoff),
            Csize_t(maxdim),
            out,
        )
        _check_backend_status(status, "adding TensorTrains")
        result_handle = out[]
        return _treetn_from_handle(result_handle)
    finally
        _release_treetn_handle(result_handle)
        _release_treetn_handle(b_handle)
        _release_treetn_handle(a_handle)
    end
end

"""
    dot(a, b)

Return the backend inner product of two TensorTrain chains.
"""
function dot(a::TensorTrain, b::TensorTrain)
    _validate_tt_binary(a, b, "dot")

    scalar_kind = _promoted_scalar_kind(a, b)
    a_handle = _new_treetn_handle(a, scalar_kind)
    b_handle = _new_treetn_handle(b, scalar_kind)
    try
        out_re = Ref{Cdouble}(0.0)
        out_im = Ref{Cdouble}(0.0)
        status = ccall(
            _t4a(:t4a_treetn_inner),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{Cdouble}, Ref{Cdouble}),
            a_handle,
            b_handle,
            out_re,
            out_im,
        )
        _check_backend_status(status, "computing TensorTrain inner product")
        return ComplexF64(out_re[], out_im[])
    finally
        _release_treetn_handle(b_handle)
        _release_treetn_handle(a_handle)
    end
end

"""
    inner(a, b)

Alias for [`dot`](@ref) on TensorTrain chains.
"""
inner(a::TensorTrain, b::TensorTrain) = dot(a, b)

function LinearAlgebra.norm(tt::TensorTrain)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty for norm"))

    tt_handle = _new_treetn_handle(tt, _promoted_scalar_kind(tt))
    try
        out_norm = Ref{Cdouble}(0.0)
        status = ccall(
            _t4a(:t4a_treetn_norm),
            Cint,
            (Ptr{Cvoid}, Ref{Cdouble}),
            tt_handle,
            out_norm,
        )
        _check_backend_status(status, "computing TensorTrain norm")
        return out_norm[]
    finally
        _release_treetn_handle(tt_handle)
    end
end

function Base.isapprox(
    a::TensorTrain,
    b::TensorTrain;
    atol::Real=0,
    rtol::Real=Base.rtoldefault(Float64, Float64, atol),
)
    d = LinearAlgebra.norm(a - b)
    isfinite(d) || error("In `isapprox(a::TensorTrain, b::TensorTrain)`, `norm(a - b)` is not finite")
    return d <= max(atol, rtol * max(LinearAlgebra.norm(a), LinearAlgebra.norm(b)))
end

"""
    dist(a, b)

Return the Euclidean distance between two TensorTrain chains.
"""
function dist(a::TensorTrain, b::TensorTrain)
    _validate_tt_binary(a, b, "dist")
    aa = dot(a, a)
    bb = dot(b, b)
    ab = dot(a, b)
    return sqrt(abs(aa + bb - 2 * real(ab)))
end

Base.:*(α::Number, tt::TensorTrain) = _treetn_scale(tt, Float64(real(α)), Float64(imag(α)))
Base.:*(tt::TensorTrain, α::Number) = α * tt
Base.:/(tt::TensorTrain, α::Number) = tt * inv(α)
Base.:-(tt::TensorTrain) = _treetn_scale(tt, -1.0, 0.0)
Base.:+(a::TensorTrain, b::TensorTrain) = add(a, b)

function Base.:-(a::TensorTrain, b::TensorTrain)
    _validate_tt_binary(a, b, "subtract")
    return add(a, _treetn_scale(b, -1.0, 0.0))
end
