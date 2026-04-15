using Test
using Tensor4all
using HDF5
using ITensors
using ITensorMPS

@testset "HDF5 ITensorMPS interoperability" begin
    mktempdir() do dir
        path = joinpath(dir, "tensor4all_to_itensormps.h5")
        i = Tensor4all.Index(2; tags=["Site", "n=1"])
        t = Tensor4all.Tensor([1.0, 0.0], [i])
        tt = Tensor4all.TensorNetworks.TensorTrain([t], 0, 2)

        Tensor4all.TensorNetworks.save_as_mps(path, "psi", tt)

        h5open(path, "r") do f
            psi = read(f, "psi", ITensorMPS.MPS)
            @test length(psi) == 1
            @test vec(ITensors.array(psi[1], ITensors.inds(psi[1])...)) == [1.0, 0.0]
        end
    end

    mktempdir() do dir
        path = joinpath(dir, "itensormps_to_tensor4all.h5")
        sites = siteinds("S=1/2", 2)
        psi = MPS(sites, "Up")

        h5open(path, "w") do f
            write(f, "psi", psi)
        end

        tt = Tensor4all.TensorNetworks.load_tt(path, "psi")
        @test length(tt) == 2
        for n in 1:2
            data, _ = Tensor4all._dense_array(tt.data[n])
            expected = ITensors.array(psi[n], ITensors.inds(psi[n])...)
            @test size(data) == size(expected)
            @test vec(data) == vec(expected)
        end
    end
end
