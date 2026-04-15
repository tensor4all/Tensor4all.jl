using Test
using Tensor4all
using HDF5

@testset "HDF5 MPS-schema roundtrip" begin
    mktempdir() do dir
        path = joinpath(dir, "tt.h5")
        i = Tensor4all.Index(2; tags=["Site", "n=1"])
        t = Tensor4all.Tensor(ones(2), [i])
        tt = Tensor4all.TensorNetworks.TensorTrain([t], 0, 2)

        ext = Base.get_extension(Tensor4all, :Tensor4allHDF5Ext)
        @test ext !== nothing

        Tensor4all.TensorNetworks.save_as_mps(path, "psi", tt)
        tt2 = Tensor4all.TensorNetworks.load_tt(path, "psi")

        @test tt2 isa Tensor4all.TensorNetworks.TensorTrain
        @test tt2.llim == 0
        @test tt2.rlim == 2
        @test length(tt2) == 1
        @test tt2.data[1].data == t.data
        @test Tensor4all.inds(tt2.data[1]) == Tensor4all.inds(t)

        h5open(path, "r") do f
            g = f["psi"]
            @test read(attributes(g)["type"]) == "MPS"
            @test read(attributes(g)["version"]) == 1
            @test read(g["length"]) == 1
            @test read(g["llim"]) == 0
            @test read(g["rlim"]) == 2
        end
    end

    mktempdir() do dir
        path = joinpath(dir, "tt2.h5")
        i1 = Tensor4all.Index(2; tags=["Site", "n=1"])
        i2 = Tensor4all.Index(2; tags=["Site", "n=2"])
        t1 = Tensor4all.Tensor([1.0, 0.0], [i1])
        t2 = Tensor4all.Tensor([0.0, 1.0], [i2])
        tt = Tensor4all.TensorNetworks.TensorTrain([t1, t2], 0, 3)

        ext = Base.get_extension(Tensor4all, :Tensor4allHDF5Ext)
        @test ext !== nothing
        Tensor4all.TensorNetworks.save_as_mps(path, "psi", tt)
        tt2 = Tensor4all.TensorNetworks.load_tt(path, "psi")

        @test length(tt2) == 2
        @test tt2.llim == 0
        @test tt2.rlim == 3
        @test tt2.data[1].data == t1.data
        @test tt2.data[2].data == t2.data

        h5open(path, "r") do f
            g = f["psi"]
            @test read(g["length"]) == 2
            @test haskey(g, "MPS[1]")
            @test haskey(g, "MPS[2]")
        end
    end
end
