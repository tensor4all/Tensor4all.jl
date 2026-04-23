using Test
using Tensor4all

const TN = Tensor4all.TensorNetworks

@testset "TensorTrain llim/rlim" begin
    @testset "setindex! widens ortho limits" begin
        i1 = Index(2; tags=["s1"])
        i2 = Index(2; tags=["s2"])
        i3 = Index(2; tags=["s3"])
        link1 = Index(2; tags=["l1"])
        link2 = Index(2; tags=["l2"])

        t1 = Tensor(randn(2, 2), [i1, link1])
        t2 = Tensor(randn(2, 2, 2), [link1, i2, link2])
        t3 = Tensor(randn(2, 2), [link2, i3])

        tt = TN.TensorTrain([t1, t2, t3], 1, 3)

        tt[1] = t1
        @test tt.llim == 0
        @test tt.rlim == 3
    end

    @testset "setindex! widens rlim" begin
        i1 = Index(2; tags=["s1"])
        i2 = Index(2; tags=["s2"])
        link = Index(2; tags=["l"])

        t1 = Tensor(randn(2, 2), [i1, link])
        t2 = Tensor(randn(2, 2), [link, i2])

        tt = TN.TensorTrain([t1, t2], 0, 2)

        tt[2] = t2
        @test tt.llim == 0
        @test tt.rlim == 3
    end

    @testset "default constructor has no ortho" begin
        i = Index(2; tags=["s"])
        t = Tensor(randn(2), [i])
        tt = TN.TensorTrain([t])
        @test tt.llim == 0
        @test tt.rlim == 2
    end

    @testset "matchsiteinds resets llim/rlim for MPS-like embedding" begin
        sites = [Index(2; tags=["x", "x=$n"]) for n in 1:3]
        link = Index(2; tags=["Link", "l=1"])

        tt = TN.TensorTrain(
            Tensor[
                Tensor(randn(2, 2), [sites[1], link]),
                Tensor(randn(2, 2), [link, sites[3]]),
            ],
            1,
            3,
        )

        embedded = TN.matchsiteinds(tt, sites)
        @test length(embedded) == 3
        @test embedded.llim == 0
        @test embedded.rlim == 4
    end

    @testset "matchsiteinds resets llim/rlim for MPO-like embedding" begin
        input_sites = [Index(2; tags=["xin", "xin=$n"]) for n in 1:4]
        output_sites = [Index(2; tags=["xout", "xout=$n"]) for n in 1:4]
        link = Index(2; tags=["Link", "l=1"])

        tt = TN.TensorTrain(
            Tensor[
                Tensor(randn(2, 2, 2), [input_sites[1], output_sites[1], link]),
                Tensor(randn(2, 2, 2), [link, input_sites[4], output_sites[4]]),
            ],
            1,
            3,
        )

        embedded = TN.matchsiteinds(tt, input_sites, output_sites)
        @test length(embedded) == 4
        @test embedded.llim == 0
        @test embedded.rlim == 5
    end

    @testset "apply() syncs llim/rlim from backend" begin
        input_sites = [Index(2; tags=["x", "x=$n"]) for n in 1:2]
        state_link = Index(2; tags=["Link", "state-l=1"])
        state = TN.TensorTrain(
            Tensor[
                Tensor(reshape([1.0, 2.0, 3.0, 4.0], 2, 2), [input_sites[1], state_link]),
                Tensor(reshape([1.0, -1.0, 0.5, 2.0], 2, 2), [state_link, input_sites[2]]),
            ],
            1,
            2,
        )

        output_internal = [Index(2; tags=["tmpout", "tmpout=$n"]) for n in 1:2]
        output_true = [Index(2; tags=["y", "y=$n"]) for n in 1:2]
        op_link = Index(1; tags=["Link", "op-l=1"])
        mpo = TN.TensorTrain(
            Tensor[
                Tensor(
                    reshape([1.0, 0.0, 0.0, 1.0], 2, 2, 1),
                    [output_internal[1], input_sites[1], op_link],
                ),
                Tensor(
                    reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2),
                    [op_link, output_internal[2], input_sites[2]],
                ),
            ],
            0,
            3,
        )
        op = TN.LinearOperator(; mpo, input_indices=copy(input_sites), output_indices=copy(output_internal))
        TN.set_iospaces!(op, input_sites, output_true)

        result = TN.apply(op, state)

        @test (state.llim, state.rlim) == (1, 2)
        @test (result.llim, result.rlim) == (0, 2)
        @test result.llim != state.llim
        @test TN.findallsiteinds_by_tag(result; tag="y") == output_true
    end

    @testset "public TensorTrain mutation helpers" begin
        i1 = Index(2; tags=["s1"])
        i2 = Index(2; tags=["s2"])
        l = Index(2; tags=["Link", "l=1"])
        t1 = Tensor(randn(2, 2), [i1, l])
        t2 = Tensor(randn(2, 2), [l, i2])
        tt = TN.TensorTrain([t1, t2], 1, 2)

        @test TN.invalidate_canonical!(tt, 1) === tt
        @test (tt.llim, tt.rlim) == (0, 2)

        tt = TN.TensorTrain([t1, t2], 1, 2)
        @test TN.replaceblock!(tt, 2, t2) === tt
        @test (tt.llim, tt.rlim) == (1, 3)

        newsite = Index(2; tags=["s3"])
        newtensor = Tensor(randn(2), [newsite])
        @test insert!(tt, 2, newtensor) === tt
        @test (tt.llim, tt.rlim) == (0, length(tt) + 1)
        @test deleteat!(tt, 2) === tt
        @test (tt.llim, tt.rlim) == (0, length(tt) + 1)
        @test push!(tt, newtensor) === tt
        @test pushfirst!(tt, newtensor) === tt
        @test (tt.llim, tt.rlim) == (0, length(tt) + 1)
    end
end
