using Test
using Tensor4all

const TN = Tensor4all.TensorNetworks

function canonical_mps_tensor(tensor::Tensor, position::Int, length_tt::Int)
    data = copy_data(tensor)
    if ndims(data) == 1
        return Array(reshape(copy(data), 1, size(data, 1), 1))
    elseif ndims(data) == 2
        if position == 1
            return Array(reshape(copy(data), 1, size(data, 1), size(data, 2)))
        elseif position == length_tt
            return Array(reshape(copy(data), size(data, 1), size(data, 2), 1))
        end
    elseif ndims(data) == 3
        return copy(data)
    end
    error("Unsupported MPS-like tensor shape $(size(data)) at position $position")
end

function canonical_mpo_tensor(tensor::Tensor, position::Int, length_tt::Int)
    data = copy_data(tensor)
    if ndims(data) == 2
        return Array(reshape(copy(data), 1, size(data, 1), size(data, 2), 1))
    elseif ndims(data) == 3
        if position == 1
            return Array(reshape(copy(data), 1, size(data, 1), size(data, 2), size(data, 3)))
        elseif position == length_tt
            return Array(reshape(copy(data), size(data, 1), size(data, 2), size(data, 3), 1))
        end
    elseif ndims(data) == 4
        return copy(data)
    end
    error("Unsupported MPO-like tensor shape $(size(data)) at position $position")
end

function dense_mps(tt::TN.TensorTrain, sites::Vector{Index})
    length(tt) == length(sites) || error("Site count mismatch")
    tensors = [canonical_mps_tensor(tt[n], n, length(tt)) for n in eachindex(tt.data)]
    output = zeros(Float64, Tuple(dim.(sites))...)

    if length(tensors) == 1
        for site_index in CartesianIndices(output)
            output[site_index] = tensors[1][1, site_index[1], 1]
        end
        return output
    end

    bond_dims = [size(tensors[n], 3) for n in 1:(length(tensors) - 1)]
    for site_index in CartesianIndices(output)
        site_values = Tuple(site_index)
        total = 0.0
        for bonds in Iterators.product((1:bond_dim for bond_dim in bond_dims)...)
            contribution = tensors[1][1, site_values[1], bonds[1]]
            for n in 2:(length(tensors) - 1)
                contribution *= tensors[n][bonds[n - 1], site_values[n], bonds[n]]
            end
            contribution *= tensors[end][bonds[end], site_values[end], 1]
            total += contribution
        end
        output[site_index] = total
    end
    return output
end

function dense_mpo(tt::TN.TensorTrain, input_sites::Vector{Index}, output_sites::Vector{Index})
    length(tt) == length(input_sites) == length(output_sites) || error("Site count mismatch")
    tensors = [canonical_mpo_tensor(tt[n], n, length(tt)) for n in eachindex(tt.data)]
    output = zeros(Float64, Tuple(dim.(input_sites))..., Tuple(dim.(output_sites))...)

    if length(tensors) == 1
        for input_index in CartesianIndices(Tuple(dim.(input_sites)))
            for output_index in CartesianIndices(Tuple(dim.(output_sites)))
                output[Tuple(input_index)..., Tuple(output_index)...] =
                    tensors[1][1, input_index[1], output_index[1], 1]
            end
        end
        return output
    end

    bond_dims = [size(tensors[n], 4) for n in 1:(length(tensors) - 1)]
    for input_index in CartesianIndices(Tuple(dim.(input_sites)))
        input_values = Tuple(input_index)
        for output_index in CartesianIndices(Tuple(dim.(output_sites)))
            output_values = Tuple(output_index)
            total = 0.0
            for bonds in Iterators.product((1:bond_dim for bond_dim in bond_dims)...)
                contribution = tensors[1][1, input_values[1], output_values[1], bonds[1]]
                for n in 2:(length(tensors) - 1)
                    contribution *= tensors[n][bonds[n - 1], input_values[n], output_values[n], bonds[n]]
                end
                contribution *= tensors[end][bonds[end], input_values[end], output_values[end], 1]
                total += contribution
            end
            output[Tuple(input_index)..., Tuple(output_index)...] = total
        end
    end
    return output
end

function sparse_mps_fixture()
    full_sites = [Index(2; tags=["x", "x=$n"]) for n in 1:3]
    l0 = Index(1; tags=["Link", "l=0"])
    l1 = Index(2; tags=["Link", "l=1"])
    l2 = Index(1; tags=["Link", "l=2"])

    tt = TN.TensorTrain(
        Tensor[
            Tensor(reshape([1.0, 2.0, 3.0, 4.0], 1, 2, 2), [l0, full_sites[1], l1]),
            Tensor(reshape([5.0, 6.0, 7.0, 8.0], 2, 2, 1), [l1, full_sites[3], l2]),
        ],
        0,
        3,
    )

    return (; tt, sparse_sites=[full_sites[1], full_sites[3]], full_sites)
end

function sparse_mpo_fixture()
    full_input = [Index(2; tags=["xin", "xin=$n"]) for n in 1:4]
    full_output = [Index(2; tags=["xout", "xout=$n"]) for n in 1:4]
    l0 = Index(1; tags=["Link", "l=0"])
    l1 = Index(2; tags=["Link", "l=1"])
    l2 = Index(1; tags=["Link", "l=2"])

    tt = TN.TensorTrain(
        Tensor[
            Tensor(reshape(collect(1.0:8.0), 1, 2, 2, 2), [l0, full_input[1], full_output[1], l1]),
            Tensor(reshape(collect(9.0:16.0), 2, 2, 2, 1), [l1, full_input[4], full_output[4], l2]),
        ],
        0,
        3,
    )

    return (; tt, sparse_input=[full_input[1], full_input[4]], sparse_output=[full_output[1], full_output[4]], full_input, full_output)
end

@testset "TensorNetworks.matchsiteinds" begin
    @testset "MPS-like sparse site set is embedded with constant-one legs" begin
        fixture = sparse_mps_fixture()
        ψ_full = TN.matchsiteinds(fixture.tt, fixture.full_sites)
        dense_sparse = dense_mps(fixture.tt, fixture.sparse_sites)
        expected = repeat(
            reshape(dense_sparse, dim(fixture.full_sites[1]), 1, dim(fixture.full_sites[3])),
            1,
            dim(fixture.full_sites[2]),
            1,
        )

        @test length(ψ_full) == 3
        @test dense_mps(ψ_full, fixture.full_sites) == expected
    end

    @testset "MPO-like sparse site set is embedded with identity legs" begin
        fixture = sparse_mpo_fixture()
        M_full = TN.matchsiteinds(fixture.tt, fixture.full_input, fixture.full_output)
        dense_sparse = dense_mpo(fixture.tt, fixture.sparse_input, fixture.sparse_output)
        expected = zeros(
            Float64,
            Tuple(dim.(fixture.full_input))...,
            Tuple(dim.(fixture.full_output))...,
        )

        for input_index in CartesianIndices(Tuple(dim.(fixture.full_input)))
            input_values = Tuple(input_index)
            for output_index in CartesianIndices(Tuple(dim.(fixture.full_output)))
                output_values = Tuple(output_index)
                expected[input_values..., output_values...] =
                    dense_sparse[input_values[1], input_values[4], output_values[1], output_values[4]] *
                    (input_values[2] == output_values[2] ? 1.0 : 0.0) *
                    (input_values[3] == output_values[3] ? 1.0 : 0.0)
            end
        end

        @test length(M_full) == 4
        @test dense_mpo(M_full, fixture.full_input, fixture.full_output) == expected
    end
end
