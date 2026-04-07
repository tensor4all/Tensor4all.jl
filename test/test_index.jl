@testset "Index" begin
    @testset "creation" begin
        # Basic creation
        i = T4AIndex(5)
        @test Tensor4all.dim(i) == 5
        @test Tensor4all.tags(i) == ""
        @test Tensor4all.id(i) != 0

        # With tags
        j = T4AIndex(3; tags="Site,n=1")
        @test Tensor4all.dim(j) == 3
        @test Tensor4all.hastag(j, "Site")
        @test Tensor4all.hastag(j, "n=1")
        @test !Tensor4all.hastag(j, "Missing")

        # Tags string contains both
        t = Tensor4all.tags(j)
        @test occursin("Site", t)
        @test occursin("n=1", t)
    end

    @testset "custom ID" begin
        id_val = UInt64(0x12345678_9ABCDEF0)
        i = T4AIndex(4, id_val; tags="Custom")
        @test Tensor4all.dim(i) == 4
        @test Tensor4all.id(i) == id_val
        @test Tensor4all.hastag(i, "Custom")
    end

    @testset "copy" begin
        i = T4AIndex(5; tags="Original")
        j = copy(i)

        @test Tensor4all.dim(i) == Tensor4all.dim(j)
        @test Tensor4all.id(i) == Tensor4all.id(j)
        @test Tensor4all.tags(i) == Tensor4all.tags(j)
        @test i == j  # Equal by ID
    end

    @testset "equality and hashing" begin
        i = T4AIndex(5)
        j = copy(i)
        k = T4AIndex(5)  # Different ID

        @test i == j
        @test i != k
        @test hash(i) == hash(j)
        @test hash(i) != hash(k)  # Very likely different
    end

    @testset "display" begin
        i = T4AIndex(3; tags="Site")
        s = sprint(show, i)
        @test occursin("dim=3", s)
        @test occursin("Site", s)
    end

    @testset "error handling" begin
        @test_throws ArgumentError T4AIndex(0)
        @test_throws ArgumentError T4AIndex(-1)
    end

    @testset "prime level" begin
        i = T4AIndex(5; tags="Site")

        # Default plev is 0
        @test Tensor4all.plev(i) == 0

        # prime
        ip = Tensor4all.prime(i)
        @test Tensor4all.plev(ip) == 1
        @test Tensor4all.id(ip) == Tensor4all.id(i)

        # double prime
        ipp = Tensor4all.prime(ip)
        @test Tensor4all.plev(ipp) == 2

        # noprime
        i0 = Tensor4all.noprime(ipp)
        @test Tensor4all.plev(i0) == 0

        # setprime
        i3 = Tensor4all.setprime(i, 3)
        @test Tensor4all.plev(i3) == 3

        # equality includes plev
        @test i != ip
        @test i == Tensor4all.noprime(ip)

        # hash includes plev
        @test hash(i) != hash(ip)
        @test hash(i) == hash(Tensor4all.noprime(ip))

        # index matching includes plev
        @test !Tensor4all.hascommoninds([i], [ip])
        @test isempty(Tensor4all.commoninds([i], [ip]))

        # sim preserves plev
        ip_sim = Tensor4all.sim(ip)
        @test Tensor4all.plev(ip_sim) == 1
        @test Tensor4all.id(ip_sim) != Tensor4all.id(ip)

        # display shows prime
        s = sprint(show, ip)
        @test occursin("'", s)
    end
end
