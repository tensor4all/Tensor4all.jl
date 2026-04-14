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

        ext.save_as_mps(path, "psi", tt)
        tt2 = ext.load_tt(path, "psi")

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
end
