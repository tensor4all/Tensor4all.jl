using Test
using Tensor4all
using LinearAlgebra: norm

const TN = Tensor4all.TensorNetworks

function make_chain_indices()
    sites = [Index(2; tags=["s", "s=$n"]) for n in 1:3]
    links = [Index(2; tags=["Link", "l=$n"]) for n in 1:2]
    return sites, links
end

function make_test_mps(; sites::Union{Nothing, Vector{Index}}=nothing, links::Union{Nothing, Vector{Index}}=nothing, variant::Symbol=:a)
    sites === nothing && ((sites, links) = make_chain_indices())
    links === nothing && throw(ArgumentError("links must be provided when sites are provided"))

    if variant === :a
        t1_data = reshape([1.0, -0.5, 0.25, 2.0], 2, 2)
        t2_data = reshape([1.0, -1.0, 0.5, 0.25, 0.75, -0.5, 1.5, 2.0], 2, 2, 2)
        t3_data = reshape([1.0, 0.5, -0.75, 2.0], 2, 2)
    elseif variant === :b
        t1_data = reshape([0.5, 1.0, -1.5, 0.25], 2, 2)
        t2_data = reshape([0.25, 1.5, -0.5, 0.75, 2.0, -1.0, 0.5, 1.25], 2, 2, 2)
        t3_data = reshape([1.5, -0.5, 0.25, 1.0], 2, 2)
    else
        throw(ArgumentError("unknown MPS variant $variant"))
    end

    return TN.TensorTrain(
        Tensor[
            Tensor(t1_data, [sites[1], links[1]]),
            Tensor(t2_data, [links[1], sites[2], links[2]]),
            Tensor(t3_data, [links[2], sites[3]]),
        ],
    )
end

_tt_arith_prod_dims(xs) = isempty(xs) ? 1 : prod(xs)

function _tt_arith_dense_contract(
    a::AbstractArray,
    ainds::Vector{Index},
    b::AbstractArray,
    binds::Vector{Index},
)
    common = [index for index in ainds if index in binds]
    a_common_axes = [findfirst(==(index), ainds) for index in common]
    b_common_axes = [findfirst(==(index), binds) for index in common]
    a_rest_axes = [axis for axis in eachindex(ainds) if axis ∉ a_common_axes]
    b_rest_axes = [axis for axis in eachindex(binds) if axis ∉ b_common_axes]

    amat = reshape(
        permutedims(a, (a_rest_axes..., a_common_axes...)),
        _tt_arith_prod_dims(size(a)[a_rest_axes]),
        _tt_arith_prod_dims(size(a)[a_common_axes]),
    )
    bmat = reshape(
        permutedims(b, (b_common_axes..., b_rest_axes...)),
        _tt_arith_prod_dims(size(b)[b_common_axes]),
        _tt_arith_prod_dims(size(b)[b_rest_axes]),
    )

    data = reshape(
        amat * bmat,
        size(a)[a_rest_axes]...,
        size(b)[b_rest_axes]...,
    )
    return data, [ainds[a_rest_axes]..., binds[b_rest_axes]...]
end

function tt_arith_dense_tensor(tt::TN.TensorTrain, target_inds::Vector{Index})
    data = copy(tt[1].data)
    current_inds = inds(tt[1])
    for n in 2:length(tt)
        data, current_inds = _tt_arith_dense_contract(data, current_inds, tt[n].data, inds(tt[n]))
    end

    boundary_axes = [axis for (axis, index) in pairs(current_inds) if hastag(index, "Link")]
    if !isempty(boundary_axes)
        data = dropdims(data; dims=Tuple(boundary_axes))
        current_inds = [index for index in current_inds if !hastag(index, "Link")]
    end

    permutation = map(target_inds) do index
        axis = findfirst(==(index), current_inds)
        axis === nothing && error("Target index $index not found")
        axis
    end
    return permutedims(data, Tuple(permutation))
end

function make_known_two_site_mps()
    s1 = Index(2; tags=["k", "k=1"])
    s2 = Index(2; tags=["k", "k=2"])
    link = Index(2; tags=["Link", "k-link=1"])
    tt = TN.TensorTrain([
        Tensor([1.0 0.0; 0.0 1.0], [s1, link]),
        Tensor([2.0 0.0; 0.0 -1.0], [link, s2]),
    ])
    dense = [2.0 0.0; 0.0 -1.0]
    return tt, [s1, s2], dense
end

function make_mpo_like_indices()
    inputs = [Index(3; tags=["Site$n"], plev=0) for n in 1:3]
    outputs = prime.(inputs)
    links = [Index(9; tags=["Link", "l=1"]), Index(3; tags=["Link", "l=2"])]
    return inputs, outputs, links
end

function make_mpo_like_tt(; scale::Float64=1.0, inputs=nothing, outputs=nothing, links=nothing)
    if inputs === nothing
        inputs, outputs, links = make_mpo_like_indices()
    end

    return TN.TensorTrain([
        Tensor(
            scale .* reshape(collect(1.0:81.0), 3, 3, 9),
            [inputs[1], outputs[1], links[1]],
        ),
        Tensor(
            scale .* reshape(collect(1.0:243.0), 9, 3, 3, 3),
            [links[1], inputs[2], outputs[2], links[2]],
        ),
        Tensor(
            scale .* reshape(collect(1.0:27.0), 3, 3, 3),
            [links[2], inputs[3], outputs[3]],
        ),
    ])
end

@testset "TensorTrain arithmetic" begin
    @testset "scalar multiply" begin
        tt = make_test_mps()
        scaled = 2.0 * tt

        @test length(scaled) == 3
        @test 0 <= scaled.llim <= length(scaled)
        @test 1 <= scaled.rlim <= length(scaled) + 1
    end

    @testset "scalar multiply from right" begin
        tt = make_test_mps()
        scaled = tt * 3.0
        @test length(scaled) == 3
    end

    @testset "scalar divide" begin
        tt = make_test_mps()
        divided = tt / 2.0
        @test length(divided) == 3
    end

    @testset "unary negation" begin
        tt = make_test_mps()
        negated = -tt
        @test length(negated) == 3
    end

    @testset "empty TensorTrain errors" begin
        empty_tt = TN.TensorTrain(Tensor[])
        @test_throws ArgumentError 2.0 * empty_tt
    end
end

@testset "TensorTrain addition" begin
    @testset "exact addition" begin
        sites, links = make_chain_indices()
        tt1 = make_test_mps(; sites, links, variant=:a)
        tt2 = make_test_mps(; sites, links, variant=:b)
        result = tt1 + tt2

        @test length(result) == 3
        @test 0 <= result.llim <= length(result)
        @test 1 <= result.rlim <= length(result) + 1
    end

    @testset "subtraction" begin
        sites, links = make_chain_indices()
        tt1 = make_test_mps(; sites, links, variant=:a)
        tt2 = make_test_mps(; sites, links, variant=:b)
        result = tt1 - tt2
        @test length(result) == 3
    end

    @testset "length mismatch" begin
        sites = [Index(2; tags=["s", "s=$n"]) for n in 1:2]
        link = Index(2; tags=["Link", "l=1"])
        tt1 = TN.TensorTrain([
            Tensor(reshape([1.0, 0.0, 0.0, 1.0], 2, 2), [sites[1], link]),
            Tensor(reshape([1.0, 0.0, 0.0, 1.0], 2, 2), [link, sites[2]]),
        ])
        tt2 = TN.TensorTrain([Tensor([1.0, -1.0], [sites[1]])])
        @test_throws DimensionMismatch tt1 + tt2
    end

    @testset "site mismatch" begin
        tt1 = make_test_mps()
        tt2 = make_test_mps()
        @test_throws ArgumentError tt1 + tt2
    end

    @testset "empty train errors" begin
        empty_tt = TN.TensorTrain(Tensor[])
        sites, links = make_chain_indices()
        tt = make_test_mps(; sites, links)
        @test_throws ArgumentError empty_tt + tt
    end

    @testset "same-id prime-pair MPO-like addition is rejected" begin
        inputs, outputs, links = make_mpo_like_indices()
        tt1 = make_mpo_like_tt(; scale=1.0, inputs, outputs, links)
        tt2 = make_mpo_like_tt(; scale=-0.5, inputs, outputs, links)

        @test_throws ArgumentError tt1 + tt2
        @test_throws ArgumentError TensorNetworks.add(
            tt1,
            tt2;
            threshold=1e-20,
            svd_policy=Tensor4all.ITensorCompat.ITENSORS_CUTOFF_POLICY,
        )
    end
end

@testset "TensorTrain truncated add" begin
    sites, links = make_chain_indices()
    tt1 = make_test_mps(; sites, links, variant=:a)
    tt2 = make_test_mps(; sites, links, variant=:b)
    result = TensorNetworks.add(tt1, tt2; threshold=1e-10)
    @test length(result) == 3

    # Explicit policy path.
    result_pol = TensorNetworks.add(tt1, tt2;
        threshold=1e-10,
        svd_policy=TensorNetworks.SvdTruncationPolicy(rule=:discarded_tail_sum),
    )
    @test length(result_pol) == 3

    # Negative threshold rejected.
    @test_throws ArgumentError TensorNetworks.add(tt1, tt2; threshold=-1.0)
end

@testset "TensorTrain inner/dot" begin
    sites, links = make_chain_indices()
    tt1 = make_test_mps(; sites, links, variant=:a)
    tt2 = make_test_mps(; sites, links, variant=:b)

    d = TensorNetworks.dot(tt1, tt2)
    @test d isa Number

    self_dot = TensorNetworks.dot(tt1, tt1)
    @test real(self_dot) >= 0
    @test abs(imag(self_dot)) < 1e-10

    @test TensorNetworks.inner(tt1, tt2) ≈ d
end

@testset "TensorTrain norm" begin
    sites, links = make_chain_indices()
    tt = make_test_mps(; sites, links, variant=:a)
    n = TensorNetworks.norm(tt)
    @test n >= 0
    @test n ≈ sqrt(real(TensorNetworks.dot(tt, tt)))

    @testset "same-id prime-pair MPO-like norm/dot is rejected" begin
        inputs, outputs, links = make_mpo_like_indices()
        tt1 = make_mpo_like_tt(; scale=1.0, inputs, outputs, links)
        tt2 = make_mpo_like_tt(; scale=-0.5, inputs, outputs, links)

        @test_throws ArgumentError TensorNetworks.norm(tt1)
        @test_throws ArgumentError TensorNetworks.dot(tt1, tt2)
    end
end

@testset "TensorTrain isapprox" begin
    sites, links = make_chain_indices()
    tt1 = make_test_mps(; sites, links, variant=:a)
    tt2 = make_test_mps(; sites, links, variant=:b)

    @test isapprox(tt1, tt1)
    @test !isapprox(tt1, tt2; atol=1e-10)
end

@testset "TensorTrain dist" begin
    sites, links = make_chain_indices()
    tt1 = make_test_mps(; sites, links, variant=:a)
    tt2 = make_test_mps(; sites, links, variant=:b)

    d = TensorNetworks.dist(tt1, tt2)
    @test d >= 0
    @test TensorNetworks.dist(tt1, tt1) < 1e-10
end

@testset "norm/inner empty train errors" begin
    empty_tt = TN.TensorTrain(Tensor[])
    sites, links = make_chain_indices()
    tt = make_test_mps(; sites, links, variant=:a)

    @test_throws ArgumentError TensorNetworks.norm(empty_tt)
    @test_throws ArgumentError TensorNetworks.dot(empty_tt, tt)
end

@testset "Numerical correctness (small dense reference)" begin
    tt, sites, dense_ref = make_known_two_site_mps()

    @testset "norm matches dense" begin
        @test TensorNetworks.norm(tt) ≈ norm(dense_ref)
    end

    @testset "2*tt matches dense" begin
        scaled = 2.0 * tt
        @test tt_arith_dense_tensor(scaled, sites) ≈ 2.0 .* dense_ref
        @test TensorNetworks.norm(scaled) ≈ 2.0 * TensorNetworks.norm(tt)
    end

    @testset "tt + tt ≈ 2*tt" begin
        sum_tt = tt + tt
        scaled_tt = 2.0 * tt
        @test tt_arith_dense_tensor(sum_tt, sites) ≈ 2.0 .* dense_ref
        @test isapprox(sum_tt, scaled_tt; atol=1e-12)
    end

    @testset "tt - tt ≈ 0" begin
        diff = tt - tt
        @test tt_arith_dense_tensor(diff, sites) ≈ zeros(2, 2)
        @test TensorNetworks.norm(diff) < 1e-12
    end

    @testset "dist(tt, tt) ≈ 0" begin
        @test TensorNetworks.dist(tt, tt) < 1e-12
    end

    @testset "complex scalar" begin
        scaled = (1.0 + 2.0im) * tt
        @test tt_arith_dense_tensor(scaled, sites) ≈ (1.0 + 2.0im) .* dense_ref
        @test TensorNetworks.norm(scaled) ≈ abs(1.0 + 2.0im) * TensorNetworks.norm(tt)
    end
end
