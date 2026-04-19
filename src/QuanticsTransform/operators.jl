_placeholder_operator(kind::Symbol; kwargs...) = TensorNetworks.LinearOperator(metadata=(; kind, kwargs...))

function _require_positive_integer(name::AbstractString, value::Integer)
    value > 0 || throw(ArgumentError("$name must be positive, got $value"))
    return value
end

function _require_multivar_target(nvars::Integer, target::Integer)
    _require_positive_integer("nvars", nvars)
    1 <= target <= nvars || throw(
        ArgumentError("target must be between 1 and $nvars, got $target"),
    )
    return target
end

function _require_nonnegative_integer(name::AbstractString, value::Integer)
    value >= 0 || throw(ArgumentError("$name must be nonnegative, got $value"))
    return value
end

function _require_nonnegative_real(name::AbstractString, value::Real)
    value >= 0 || throw(ArgumentError("$name must be nonnegative, got $value"))
    return value
end

function _require_nonzero_integer(name::AbstractString, value)
    value == 0 && throw(ArgumentError("$name must be nonzero, got $value"))
    return value
end

"""
    shift_operator(r, offset; bc=:periodic)

Construct a quantics shift operator materialized as a `TensorNetworks.LinearOperator`.
"""
function shift_operator(r::Integer, offset::Integer; bc=:periodic)
    layout_handle = _new_univariate_layout(r)
    try
        return _materialize_shift(layout_handle, 0, offset, bc, "materializing shift operator")
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    shift_operator_multivar(r, offset, nvars, target; bc=:periodic)

Construct a multivariable quantics shift operator targeting the `target`th variable.
"""
function shift_operator_multivar(
    r::Integer,
    offset::Integer,
    nvars::Integer,
    target::Integer;
    bc=:periodic,
)
    _require_multivar_target(nvars, target)
    layout_handle = _new_multivar_layout(r, nvars)
    try
        return _materialize_shift(
            layout_handle,
            target - 1,
            offset,
            bc,
            "materializing multivar shift operator",
        )
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    flip_operator(r; bc=:periodic)

Construct a quantics flip operator materialized as a `TensorNetworks.LinearOperator`.
"""
function flip_operator(r::Integer; bc=:periodic)
    layout_handle = _new_univariate_layout(r)
    try
        return _materialize_flip(layout_handle, 0, bc, "materializing flip operator")
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    flip_operator_multivar(r, nvars, target; bc=:periodic)

Construct a multivariable quantics flip operator targeting the `target`th variable.
"""
function flip_operator_multivar(r::Integer, nvars::Integer, target::Integer; bc=:periodic)
    _require_multivar_target(nvars, target)
    layout_handle = _new_multivar_layout(r, nvars)
    try
        return _materialize_flip(
            layout_handle,
            target - 1,
            bc,
            "materializing multivar flip operator",
        )
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    phase_rotation_operator(r, theta)

Construct a quantics phase-rotation operator materialized as a `TensorNetworks.LinearOperator`.
"""
function phase_rotation_operator(r::Integer, theta)
    layout_handle = _new_univariate_layout(r)
    try
        return _materialize_phase_rotation(
            layout_handle,
            0,
            theta,
            "materializing phase rotation operator",
        )
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    phase_rotation_operator_multivar(r, theta, nvars, target)

Construct a multivariable phase-rotation operator targeting the `target`th variable.
"""
function phase_rotation_operator_multivar(r::Integer, theta, nvars::Integer, target::Integer)
    _require_multivar_target(nvars, target)
    layout_handle = _new_multivar_layout(r, nvars)
    try
        return _materialize_phase_rotation(
            layout_handle,
            target - 1,
            theta,
            "materializing multivar phase rotation operator",
        )
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    cumsum_operator(r)

Construct a quantics cumulative-sum operator materialized as a `TensorNetworks.LinearOperator`.
"""
function cumsum_operator(r::Integer)
    layout_handle = _new_univariate_layout(r)
    try
        return _materialize_cumsum(layout_handle, 0, "materializing cumsum operator")
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    fourier_operator(r; forward=true, maxbonddim=typemax(Int), tolerance=0.0)

Construct a quantics Fourier operator materialized as a `TensorNetworks.LinearOperator`.
"""
function fourier_operator(
    r::Integer;
    forward::Bool=true,
    maxbonddim::Int=typemax(Int),
    tolerance::Float64=0.0,
)
    layout_handle = _new_univariate_layout(r)
    try
        return _materialize_fourier(
            layout_handle,
            0,
            forward,
            maxbonddim,
            tolerance,
            "materializing Fourier operator",
        )
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    affine_operator(r, a_num, a_den, b_num, b_den; bc=:periodic)

Construct a single-variable quantics **forward** affine operator materialized
as a `TensorNetworks.LinearOperator`. Maps a state `g(x)` of one variable to
`f(y) = g((y - b) / a)` by applying the coordinate map `y = a * x + b`
(`a = a_num / a_den`, `b = b_num / b_den`).

To obtain the pullback operator `f(y) = g(a * y + b)` call `transpose` on the
returned operator; the pullback is exactly the transpose of the forward
operator.
"""
function affine_operator(
    r::Integer,
    a_num::Integer,
    a_den::Integer,
    b_num::Integer,
    b_den::Integer;
    bc::Symbol=:periodic,
)
    layout_handle = _new_univariate_layout(r)
    try
        return _materialize_affine(
            layout_handle,
            Int64[a_num],
            Int64[a_den],
            Int64[b_num],
            Int64[b_den],
            1,
            1,
            Symbol[bc],
            "materializing affine operator",
        )
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    affine_operator_multivar(r, a_num, a_den, b_num, b_den, m, n;
                             bc=fill(:periodic, m))

Multi-variable forward affine operator. Applies the coordinate map
`y = A * x + b`, where:

- `A` is an `M × N` rational matrix supplied in column-major order through
  the parallel arrays `a_num`, `a_den` (each of length `m * n`).
- `b` is the `M`-dimensional translation vector through `b_num`, `b_den`
  (each of length `m`).
- `bc` is the per-output-dimension boundary condition (`Vector{Symbol}` of
  length `m`); each entry is one of `:periodic` or `:open`.

The layout uses `max(m, n)` variables of `r` bits each (Fused). To obtain the
pullback `f(y) = g(A * y + b)` call `transpose` on the returned operator.
"""
function affine_operator_multivar(
    r::Integer,
    a_num::AbstractVector{<:Integer},
    a_den::AbstractVector{<:Integer},
    b_num::AbstractVector{<:Integer},
    b_den::AbstractVector{<:Integer},
    m::Integer,
    n::Integer;
    bc::AbstractVector{Symbol}=fill(:periodic, m),
)
    _require_positive_integer("m", m)
    _require_positive_integer("n", n)
    layout_handle = _new_multivar_layout(r, max(Int(m), Int(n)))
    try
        return _materialize_affine(
            layout_handle,
            a_num,
            a_den,
            b_num,
            b_den,
            m,
            n,
            bc,
            "materializing multivar affine operator",
        )
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    affine_pullback_operator(r, a_num, a_den, b_num, b_den; bc=:periodic)

Construct a single-variable quantics **affine pullback** operator
materialized as a `TensorNetworks.LinearOperator`.

The pullback maps a source state `g(x)` of one variable to a target
state `f(y) = g(a * y + b)`, with `a = a_num / a_den` and `b = b_num / b_den`
expressed as exact rationals. `r` is the number of bits per variable; the
layout is the same single-variable fused layout used by
[`affine_operator`](@ref).
"""
function affine_pullback_operator(
    r::Integer,
    a_num::Integer,
    a_den::Integer,
    b_num::Integer,
    b_den::Integer;
    bc::Symbol=:periodic,
)
    layout_handle = _new_univariate_layout(r)
    try
        return _materialize_affine_pullback(
            layout_handle,
            Int64[a_num],
            Int64[a_den],
            Int64[b_num],
            Int64[b_den],
            1,
            1,
            Symbol[bc],
            "materializing affine pullback operator",
        )
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    affine_pullback_operator_multivar(r, a_num, a_den, b_num, b_den, m, n;
                                      bc=fill(:periodic, m))

Multi-variable affine pullback operator. Maps a source state
`g(x_1, ..., x_M)` to `f(y_1, ..., y_N) = g(A * y + b)` where:

- `A` is an `M × N` rational matrix supplied in column-major order through
  the parallel arrays `a_num`, `a_den` (each of length `m * n`).
- `b` is the `M`-dimensional translation vector through `b_num`, `b_den`
  (each of length `m`).
- `bc` is the per-source-dimension boundary condition (`Vector{Symbol}` of
  length `m`); each entry is one of `:periodic` or `:open`.

The layout uses `max(m, n)` variables of `r` bits each (Fused), matching
the convention of the C-API tests in tensor4all-rs#427.
"""
function affine_pullback_operator_multivar(
    r::Integer,
    a_num::AbstractVector{<:Integer},
    a_den::AbstractVector{<:Integer},
    b_num::AbstractVector{<:Integer},
    b_den::AbstractVector{<:Integer},
    m::Integer,
    n::Integer;
    bc::AbstractVector{Symbol}=fill(:periodic, m),
)
    _require_positive_integer("m", m)
    _require_positive_integer("n", n)
    layout_handle = _new_multivar_layout(r, max(Int(m), Int(n)))
    try
        return _materialize_affine_pullback(
            layout_handle,
            a_num,
            a_den,
            b_num,
            b_den,
            m,
            n,
            bc,
            "materializing multivar affine pullback operator",
        )
    finally
        _release_qtt_layout_handle(layout_handle)
    end
end

"""
    binaryop_operator(r, a1, b1, a2, b2; bc1=:periodic, bc2=:periodic)

Construct a quantics binary operator over two variables, materialized as a
`TensorNetworks.LinearOperator`. Coefficients `a1`, `b1`, `a2`, `b2` must fit
in `Int8`.

Semantics: maps `g(x, y)` to
`f(x, y) = g(a1 * x + b1 * y, a2 * x + b2 * y)` (affine pullback with linear
matrix `A = [[a1, b1]; [a2, b2]]` and zero shift).

Implemented as a thin wrapper over [`affine_pullback_operator_multivar`](@ref)
with a zero shift vector; the returned operator uses a Fused QTT layout.
"""
function binaryop_operator(
    r::Integer,
    a1::Integer,
    b1::Integer,
    a2::Integer,
    b2::Integer;
    bc1::Symbol=:periodic,
    bc2::Symbol=:periodic,
)
    _require_int8_range("a1", a1)
    _require_int8_range("b1", b1)
    _require_int8_range("a2", a2)
    _require_int8_range("b2", b2)
    return binaryop_operator_multivar(
        r, a1, b1, a2, b2, 2, 1, 2; bc1=bc1, bc2=bc2,
    )
end

"""
    binaryop_operator_multivar(r, a1, b1, a2, b2, nvars, lhs_var, rhs_var;
                               bc1=:periodic, bc2=:periodic)

Multi-variable variant of [`binaryop_operator`](@ref). `lhs_var` and `rhs_var`
are 1-based indices into a layout with `nvars` variables; they must be
distinct. The transformation acts as the 2×2 affine map on variables
`(lhs_var, rhs_var)` and as the identity on every other variable.

Implemented as a thin wrapper over [`affine_pullback_operator_multivar`](@ref)
with an `nvars × nvars` linear matrix that embeds the 2×2 coefficient matrix
at the `(lhs_var, rhs_var)` rows/cols and identity elsewhere.
"""
function binaryop_operator_multivar(
    r::Integer,
    a1::Integer,
    b1::Integer,
    a2::Integer,
    b2::Integer,
    nvars::Integer,
    lhs_var::Integer,
    rhs_var::Integer;
    bc1::Symbol=:periodic,
    bc2::Symbol=:periodic,
)
    _require_multivar_target(nvars, lhs_var)
    _require_multivar_target(nvars, rhs_var)
    lhs_var == rhs_var && throw(ArgumentError("lhs_var and rhs_var must be distinct, got $lhs_var"))
    _require_int8_range("a1", a1)
    _require_int8_range("b1", b1)
    _require_int8_range("a2", a2)
    _require_int8_range("b2", b2)

    # Build an nvars x nvars integer matrix A. Initialize to identity.
    # Then overwrite rows (lhs_var, rhs_var) x cols (lhs_var, rhs_var)
    # with [[a1, b1]; [a2, b2]].
    n = Int(nvars)
    A_num = zeros(Int, n, n)
    for i in 1:n
        A_num[i, i] = 1
    end
    A_num[lhs_var, lhs_var] = a1
    A_num[lhs_var, rhs_var] = b1
    A_num[rhs_var, lhs_var] = a2
    A_num[rhs_var, rhs_var] = b2

    # Flatten column-major (matches `affine_pullback_operator_multivar`).
    a_num = vec(A_num)
    a_den = ones(Int, n * n)
    b_num = zeros(Int, n)
    b_den = ones(Int, n)

    # Per-source-dimension boundary conditions: bc1/bc2 at the two acted-on
    # rows, periodic elsewhere (identity; bc is immaterial for the identity
    # map but must be a valid symbol).
    bc = fill(:periodic, n)
    bc[lhs_var] = bc1
    bc[rhs_var] = bc2

    return affine_pullback_operator_multivar(
        r, a_num, a_den, b_num, b_den, n, n; bc=bc,
    )
end

function _require_int8_range(name::AbstractString, value::Integer)
    -128 <= value <= 127 || throw(ArgumentError("$name must fit in Int8, got $value"))
end
