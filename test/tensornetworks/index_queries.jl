using Test
using Tensor4all

const TN = Tensor4all.TensorNetworks

function assert_throws_with_message(f::Function, exception_type, needle::AbstractString)
    err = try
        f()
        nothing
    catch caught
        caught
    end
    @test err isa exception_type
    if err !== nothing
        @test occursin(needle, sprint(showerror, err))
    end
    return nothing
end

function mps_like_fixture()
    s1 = Index(2; tags=["x", "x=1"])
    s2 = Index(2; tags=["x", "x=2"])
    s3 = Index(2; tags=["x", "x=3"])
    l1 = Index(1; tags=["Link", "l=1"])
    l2 = Index(1; tags=["Link", "l=2"])

    tt = TN.TensorTrain(
        Tensor[
            Tensor(ones(2, 1), [s1, l1]),
            Tensor(ones(1, 2, 1), [l1, s2, l2]),
            Tensor(ones(1, 2), [l2, s3]),
        ],
        0,
        4,
    )

    return (; tt, sites=[s1, s2, s3], links=[l1, l2])
end

function mpo_like_fixture()
    xin1 = Index(2; tags=["xin", "xin=1"])
    xin2 = Index(2; tags=["xin", "xin=2"])
    xin3 = Index(2; tags=["xin", "xin=3"])
    xout1 = Index(2; tags=["xout", "xout=1"])
    xout2 = Index(2; tags=["xout", "xout=2"])
    xout3 = Index(2; tags=["xout", "xout=3"])
    l1 = Index(1; tags=["Link", "l=1"])
    l2 = Index(1; tags=["Link", "l=2"])

    tt = TN.TensorTrain(
        Tensor[
            Tensor(ones(2, 2, 1), [xin1, xout1, l1]),
            Tensor(ones(1, 2, 2, 1), [l1, xin2, xout2, l2]),
            Tensor(ones(1, 2, 2), [l2, xin3, xout3]),
        ],
        0,
        4,
    )

    return (; tt, input_sites=[xin1, xin2, xin3], output_sites=[xout1, xout2, xout3], links=[l1, l2])
end

function numbered_tag_scan_fixture()
    s1 = Index(2; tags=["scan", "scan=2"])
    s2 = Index(2; tags=["scan", "scan=1"])
    s3 = Index(2; tags=["scan", "scan=3"])
    s4 = Index(2; tags=["scan"])
    l1 = Index(1; tags=["Link", "l=1"])
    l2 = Index(1; tags=["Link", "l=2"])
    l3 = Index(1; tags=["Link", "l=3"])

    tt = TN.TensorTrain(
        Tensor[
            Tensor(ones(2, 1), [s1, l1]),
            Tensor(ones(1, 2, 1), [l1, s2, l2]),
            Tensor(ones(1, 2, 1), [l2, s3, l3]),
            Tensor(ones(1, 2), [l3, s4]),
        ],
        0,
        5,
    )

    return (; tt, numbered_sites=[s2, s1, s3], bare_only=s4)
end

function duplicate_numbered_tag_fixture()
    s1 = Index(2; tags=["dup", "dup=1"])
    s2 = Index(2; tags=["dup", "dup=1"])
    s3 = Index(2; tags=["dup", "dup=2"])
    l1 = Index(1; tags=["Link", "l=1"])
    l2 = Index(1; tags=["Link", "l=2"])

    tt = TN.TensorTrain(
        Tensor[
            Tensor(ones(2, 1), [s1, l1]),
            Tensor(ones(1, 2, 1), [l1, s2, l2]),
            Tensor(ones(1, 2), [l2, s3]),
        ],
        0,
        4,
    )

    return (; tt)
end

function numbered_tag_gap_fixture()
    s1 = Index(2; tags=["gap", "gap=1"])
    s2 = Index(2; tags=["gap", "gap=3"])
    s3 = Index(2; tags=["gap"])
    l1 = Index(1; tags=["Link", "l=1"])
    l2 = Index(1; tags=["Link", "l=2"])

    tt = TN.TensorTrain(
        Tensor[
            Tensor(ones(2, 1), [s1, l1]),
            Tensor(ones(1, 2, 1), [l1, s2, l2]),
            Tensor(ones(1, 2), [l2, s3]),
        ],
        0,
        4,
    )

    return (; tt, prefix_positions=[1], prefix_sites=[s1], trailing_numbered=s2, bare_only=s3)
end

@testset "TensorNetworks index queries" begin
    @testset "findsite and findsites on MPS-like TensorTrain" begin
        fixture = mps_like_fixture()
        missing = Index(2; tags=["missing"])

        @test TN.findsite(fixture.tt, fixture.sites[2]) == 2
        @test TN.findsite(fixture.tt, [fixture.sites[2], fixture.sites[3]]) == 2
        @test TN.findsite(fixture.tt, [fixture.sites[3], fixture.sites[2]]) == 2
        @test TN.findsite(fixture.tt, [missing, fixture.sites[3]]) == 3
        @test TN.findsite(fixture.tt, fixture.links[1]) == 1
        @test TN.findsite(fixture.tt, missing) === nothing

        @test TN.findsites(fixture.tt, fixture.sites[2]) == [2]
        @test TN.findsites(fixture.tt, [fixture.sites[1], fixture.sites[3]]) == [1, 3]
        @test TN.findsites(fixture.tt, [fixture.sites[3], fixture.sites[1]]) == [1, 3]
        @test TN.findsites(fixture.tt, [missing, fixture.sites[3]]) == [3]
        @test TN.findsites(fixture.tt, fixture.links[1]) == [1, 2]
        @test isempty(TN.findsites(fixture.tt, missing))
    end

    @testset "findsite and findsites on MPO-like TensorTrain" begin
        fixture = mpo_like_fixture()
        missing = Index(2; tags=["missing"])

        @test TN.findsite(fixture.tt, fixture.input_sites[2]) == 2
        @test TN.findsite(fixture.tt, [fixture.output_sites[2], fixture.input_sites[3]]) == 2
        @test TN.findsite(fixture.tt, [fixture.input_sites[3], fixture.output_sites[2]]) == 2
        @test TN.findsite(fixture.tt, [missing, fixture.output_sites[3]]) == 3
        @test TN.findsite(fixture.tt, fixture.links[1]) == 1
        @test TN.findsite(fixture.tt, missing) === nothing

        @test TN.findsites(fixture.tt, fixture.output_sites[2]) == [2]
        @test TN.findsites(fixture.tt, [fixture.output_sites[1], fixture.input_sites[3]]) == [1, 3]
        @test TN.findsites(fixture.tt, [fixture.input_sites[3], fixture.output_sites[1]]) == [1, 3]
        @test TN.findsites(fixture.tt, [missing, fixture.output_sites[3]]) == [3]
        @test TN.findsites(fixture.tt, fixture.links[1]) == [1, 2]
        @test isempty(TN.findsites(fixture.tt, missing))
    end

    @testset "findallsites_by_tag and findallsiteinds_by_tag" begin
        mps = mps_like_fixture()
        mpo = mpo_like_fixture()
        scan = numbered_tag_scan_fixture()
        dup = duplicate_numbered_tag_fixture()
        gap = numbered_tag_gap_fixture()

        @test TN.findallsites_by_tag(mps.tt; tag="x") == [1, 2, 3]
        @test TN.findallsiteinds_by_tag(mps.tt; tag="x") == mps.sites
        @test isempty(TN.findallsites_by_tag(mps.tt; tag="missing"))
        @test isempty(TN.findallsiteinds_by_tag(mps.tt; tag="missing"))

        @test TN.findallsites_by_tag(mpo.tt; tag="xin") == [1, 2, 3]
        @test TN.findallsiteinds_by_tag(mpo.tt; tag="xout") == mpo.output_sites
        @test isempty(TN.findallsites_by_tag(mpo.tt; tag="missing"))
        @test isempty(TN.findallsiteinds_by_tag(mpo.tt; tag="missing"))

        @test TN.findallsites_by_tag(scan.tt; tag="scan") == [2, 1, 3]
        @test TN.findallsiteinds_by_tag(scan.tt; tag="scan") == scan.numbered_sites
        @test scan.bare_only ∉ TN.findallsiteinds_by_tag(scan.tt; tag="scan")

        @test TN.findallsites_by_tag(gap.tt; tag="gap") == gap.prefix_positions
        @test TN.findallsiteinds_by_tag(gap.tt; tag="gap") == gap.prefix_sites
        @test gap.trailing_numbered ∉ TN.findallsiteinds_by_tag(gap.tt; tag="gap")
        @test gap.bare_only ∉ TN.findallsiteinds_by_tag(gap.tt; tag="gap")

        assert_throws_with_message(ArgumentError, "Invalid tag") do
            TN.findallsites_by_tag(mps.tt; tag="x=1")
        end
        assert_throws_with_message(ArgumentError, "Invalid tag") do
            TN.findallsiteinds_by_tag(mps.tt; tag="x=1")
        end
        assert_throws_with_message(ArgumentError, "dup=1") do
            TN.findallsites_by_tag(dup.tt; tag="dup")
        end
        assert_throws_with_message(ArgumentError, "dup=1") do
            TN.findallsiteinds_by_tag(dup.tt; tag="dup")
        end
    end

    @testset "replace_siteinds is non-mutating" begin
        @test begin
            fixture = mps_like_fixture()
            oldsites = [fixture.sites[3], fixture.sites[1]]
            newsites = [
                Index(2; tags=["y", "y=3"]),
                Index(2; tags=["y", "y=1"]),
            ]
            replaced = TN.replace_siteinds(fixture.tt, oldsites, newsites)
            replaced !== fixture.tt
        end

        @test begin
            fixture = mps_like_fixture()
            oldsites = [fixture.sites[3], fixture.sites[1]]
            newsites = [
                Index(2; tags=["y", "y=3"]),
                Index(2; tags=["y", "y=1"]),
            ]
            replaced = TN.replace_siteinds(fixture.tt, oldsites, newsites)
            replaced[2] !== fixture.tt[2]
        end

        @test begin
            fixture = mps_like_fixture()
            oldsites = [fixture.sites[3], fixture.sites[1]]
            newsites = [
                Index(2; tags=["y", "y=3"]),
                Index(2; tags=["y", "y=1"]),
            ]
            replaced = TN.replace_siteinds(fixture.tt, oldsites, newsites)
            Array(replaced[2], inds(replaced[2])...) == Array(fixture.tt[2], inds(fixture.tt[2])...)
        end

        @test begin
            fixture = mps_like_fixture()
            oldsites = [fixture.sites[3], fixture.sites[1]]
            newsites = [
                Index(2; tags=["y", "y=3"]),
                Index(2; tags=["y", "y=1"]),
            ]
            replaced = TN.replace_siteinds(fixture.tt, oldsites, newsites)
            inds(replaced[1]) == [newsites[2], fixture.links[1]]
        end

        @test begin
            fixture = mps_like_fixture()
            oldsites = [fixture.sites[3], fixture.sites[1]]
            newsites = [
                Index(2; tags=["y", "y=3"]),
                Index(2; tags=["y", "y=1"]),
            ]
            replaced = TN.replace_siteinds(fixture.tt, oldsites, newsites)
            inds(replaced[2]) == [fixture.links[1], fixture.sites[2], fixture.links[2]]
        end

        @test begin
            fixture = mps_like_fixture()
            oldsites = [fixture.sites[3], fixture.sites[1]]
            newsites = [
                Index(2; tags=["y", "y=3"]),
                Index(2; tags=["y", "y=1"]),
            ]
            replaced = TN.replace_siteinds(fixture.tt, oldsites, newsites)
            inds(replaced[3]) == [fixture.links[2], newsites[1]]
        end

        @test begin
            fixture = mps_like_fixture()
            oldsites = [fixture.sites[3], fixture.sites[1]]
            newsites = [
                Index(2; tags=["y", "y=3"]),
                Index(2; tags=["y", "y=1"]),
            ]
            TN.replace_siteinds(fixture.tt, oldsites, newsites)
            inds(fixture.tt[1]) == [fixture.sites[1], fixture.links[1]]
        end

        @test begin
            fixture = mps_like_fixture()
            oldsites = [fixture.sites[3], fixture.sites[1]]
            newsites = [
                Index(2; tags=["y", "y=3"]),
                Index(2; tags=["y", "y=1"]),
            ]
            TN.replace_siteinds(fixture.tt, oldsites, newsites)
            inds(fixture.tt[2]) == [fixture.links[1], fixture.sites[2], fixture.links[2]]
        end

        @test begin
            fixture = mps_like_fixture()
            oldsites = [fixture.sites[3], fixture.sites[1]]
            newsites = [
                Index(2; tags=["y", "y=3"]),
                Index(2; tags=["y", "y=1"]),
            ]
            TN.replace_siteinds(fixture.tt, oldsites, newsites)
            inds(fixture.tt[3]) == [fixture.links[2], fixture.sites[3]]
        end
    end

    @testset "replace_siteinds! mutates in place" begin
        @test begin
            fixture = mpo_like_fixture()
            oldsites = [fixture.output_sites[3], fixture.output_sites[1]]
            newsites = [
                Index(2; tags=["yout", "yout=3"]),
                Index(2; tags=["yout", "yout=1"]),
            ]
            replaced = TN.replace_siteinds!(fixture.tt, oldsites, newsites)
            replaced === fixture.tt
        end

        @test begin
            fixture = mpo_like_fixture()
            oldsites = [fixture.output_sites[3], fixture.output_sites[1]]
            newsites = [
                Index(2; tags=["yout", "yout=3"]),
                Index(2; tags=["yout", "yout=1"]),
            ]
            TN.replace_siteinds!(fixture.tt, oldsites, newsites)
            inds(fixture.tt[1]) == [fixture.input_sites[1], newsites[2], fixture.links[1]]
        end

        @test begin
            fixture = mpo_like_fixture()
            oldsites = [fixture.output_sites[3], fixture.output_sites[1]]
            newsites = [
                Index(2; tags=["yout", "yout=3"]),
                Index(2; tags=["yout", "yout=1"]),
            ]
            TN.replace_siteinds!(fixture.tt, oldsites, newsites)
            inds(fixture.tt[2]) == [fixture.links[1], fixture.input_sites[2], fixture.output_sites[2], fixture.links[2]]
        end

        @test begin
            fixture = mpo_like_fixture()
            oldsites = [fixture.output_sites[3], fixture.output_sites[1]]
            newsites = [
                Index(2; tags=["yout", "yout=3"]),
                Index(2; tags=["yout", "yout=1"]),
            ]
            TN.replace_siteinds!(fixture.tt, oldsites, newsites)
            inds(fixture.tt[3]) == [fixture.links[2], fixture.input_sites[3], newsites[1]]
        end

        @test begin
            fixture = mps_like_fixture()
            fixture.tt.llim = 1
            fixture.tt.rlim = 3
            original = fixture.tt[1]
            newsite = Index(2; tags=["y", "y=1"])
            TN.replace_siteinds!(fixture.tt, [fixture.sites[1]], [newsite])
            fixture.tt[1] === original &&
                inds(fixture.tt[1]) == [newsite, fixture.links[1]] &&
                (fixture.tt.llim, fixture.tt.rlim) == (1, 3)
        end

        @testset "replace_siteinds preserves diagonal storage" begin
            i = Index(2; tags=["x", "x=1"])
            j = Index(2; tags=["y", "y=1"])
            jp = Index(2; tags=["z", "z=1"])
            tt = TN.TensorTrain([Tensor4all.delta(i, j)], 0, 2)
            original = tt[1]
            TN.replace_siteinds!(tt, [j], [jp])
            @test tt[1] === original
            @test Tensor4all.storage_kind(tt[1]) == :diagonal
            @test Tensor4all.axis_classes(tt[1]) == [0, 0]
        end
    end

    @testset "replacement helper validation" begin
        fixture = mps_like_fixture()
        new_sites = [
            Index(2; tags=["y", "y=1"]),
            Index(2; tags=["y", "y=2"]),
            Index(2; tags=["y", "y=3"]),
        ]
        missing = Index(2; tags=["y", "y=9"])
        repeated_aux = Index(1; tags=["aux"])
        repeated_aux_tt = TN.TensorTrain(
            Tensor[
                Tensor(ones(2, 1), [fixture.sites[1], repeated_aux]),
                Tensor(ones(1, 2), [repeated_aux, fixture.sites[2]]),
            ],
            0,
            3,
        )

        assert_throws_with_message(DimensionMismatch, "Length mismatch") do
            TN.replace_siteinds(fixture.tt, fixture.sites, new_sites[1:2])
        end
        assert_throws_with_message(ArgumentError, "Not found") do
            TN.replace_siteinds!(fixture.tt, [missing], [new_sites[1]])
        end
        assert_throws_with_message(ArgumentError, "duplicate") do
            TN.replace_siteinds(fixture.tt, [fixture.sites[1], fixture.sites[1]], new_sites[1:2])
        end
        assert_throws_with_message(ArgumentError, "site") do
            TN.replace_siteinds!(fixture.tt, [fixture.links[1]], [Index(1; tags=["ylink"])])
        end
        assert_throws_with_message(ArgumentError, "site") do
            TN.replace_siteinds(repeated_aux_tt, [repeated_aux], [Index(1; tags=["yaux"])])
        end
    end
end
