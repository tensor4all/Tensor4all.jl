using Test
using Tensor4all

const TN_OPMUT = Tensor4all.TensorNetworks

function _opmutation_siteinds_by_tensor(tt::TN_OPMUT.TensorTrain)
    counts = Dict{Index, Int}()
    for tensor in tt
        for index in inds(tensor)
            counts[index] = get(counts, index, 0) + 1
        end
    end
    return [
        [index for index in inds(tensor) if get(counts, index, 0) == 1 && !hastag(index, "Link")]
        for tensor in tt
    ]
end

function _opmutation_identity_op(n::Integer=2)
    input = [Index(2; tags=["in", "in=$i"]) for i in 1:n]
    output = [Index(2; tags=["out", "out=$i"]) for i in 1:n]
    true_input = [Index(2; tags=["tin", "tin=$i"]) for i in 1:n]
    true_output = [Index(2; tags=["tout", "tout=$i"]) for i in 1:n]
    links = [Index(1; tags=["Link", "op-l=$i"]) for i in 1:(n - 1)]

    tensors = Tensor[]
    for i in 1:n
        data = reshape([1.0, 0.0, 0.0, 1.0], 2, 2)
        if n == 1
            push!(tensors, Tensor(data, [input[i], output[i]]))
        elseif i == 1
            push!(tensors, Tensor(reshape(data, 2, 2, 1), [input[i], output[i], links[i]]))
        elseif i == n
            push!(tensors, Tensor(reshape(data, 1, 2, 2), [links[i - 1], input[i], output[i]]))
        else
            push!(tensors, Tensor(reshape(data, 1, 2, 2, 1), [links[i - 1], input[i], output[i], links[i]]))
        end
    end

    return TN_OPMUT.LinearOperator(;
        mpo=TN_OPMUT.TensorTrain(tensors),
        input_indices=input,
        output_indices=output,
        true_input=Union{Index, Nothing}[true_input...],
        true_output=Union{Index, Nothing}[true_output...],
    )
end

@testset "LinearOperator topology mutations" begin
    @testset "constructor validates bound-space metadata lengths" begin
        input = [Index(2; tags=["in"])]
        output = [Index(2; tags=["out"])]

        @test_throws DimensionMismatch TN_OPMUT.LinearOperator(;
            input_indices=input,
            output_indices=output,
            true_input=Union{Index, Nothing}[],
            true_output=Union{Index, Nothing}[output[1], output[1]],
        )
    end

    @testset "insert and delete identity sites keep metadata synchronized" begin
        op = _opmutation_identity_op()
        new_input = Index(2; tags=["in", "in=mid"])
        new_output = Index(2; tags=["out", "out=mid"])
        true_input = Index(2; tags=["tin", "tin=mid"])
        true_output = Index(2; tags=["tout", "tout=mid"])

        @test TN_OPMUT.insert_operator_identity!(
            op,
            2,
            new_input,
            new_output;
            true_input,
            true_output,
        ) === op

        @test length(op.mpo) == 3
        @test length(op.mpo) == length(op.input_indices) == length(op.output_indices)
        @test length(op.input_indices) == length(op.true_input) == length(op.true_output)
        @test op.input_indices[2] == new_input
        @test op.output_indices[2] == new_output
        @test op.true_input[2] == true_input
        @test op.true_output[2] == true_output
        @test _opmutation_siteinds_by_tensor(op.mpo)[2] == [new_input, new_output]
        @test op.mpo.llim == 0
        @test op.mpo.rlim == length(op.mpo) + 1

        @test TN_OPMUT.delete_operator_site!(op, 2) === op
        @test length(op.mpo) == 2
        @test all(!=(new_input), op.input_indices)
        @test all(!=(new_output), op.output_indices)
        @test length(op.input_indices) == length(op.output_indices) == length(op.true_input) == length(op.true_output)
    end

    @testset "permutation and index replacement update operator metadata" begin
        op = _opmutation_identity_op(3)
        original_input = copy(op.input_indices)
        original_output = copy(op.output_indices)
        original_true_input = copy(op.true_input)
        original_true_output = copy(op.true_output)
        original_dense = TN_OPMUT.to_dense(op.mpo)
        order = [3, 1, 2]

        @test TN_OPMUT.permute_operator_sites!(op, order) === op
        @test length(op.mpo) == 3
        @test op.input_indices == original_input[order]
        @test op.output_indices == original_output[order]
        @test op.true_input == original_true_input[order]
        @test op.true_output == original_true_output[order]
        @test isapprox(TN_OPMUT.to_dense(op.mpo), original_dense)

        old_input = op.input_indices[1]
        new_input = Index(dim(old_input); tags=["renamed-in"])
        old_true_input = op.true_input[1]
        @test TN_OPMUT.replace_operator_input_indices!(op, [old_input], [new_input]) === op
        @test op.input_indices[1] == new_input
        @test op.true_input[1] == old_true_input
        @test new_input in _opmutation_siteinds_by_tensor(op.mpo)[1]

        old_output = op.output_indices[end]
        new_output = Index(dim(old_output); tags=["renamed-out"])
        @test TN_OPMUT.replace_operator_output_indices!(op, [old_output], [new_output]) === op
        @test op.output_indices[end] == new_output
        @test new_output in _opmutation_siteinds_by_tensor(op.mpo)[end]
    end

    @testset "index replacement matches operator metadata by identity" begin
        op = _opmutation_identity_op(2)
        old_input = op.input_indices[2]
        metadata_copy = Index(
            dim(old_input);
            tags=["metadata-only"],
            plev=plev(old_input),
            id=id(old_input),
        )
        new_input = Index(dim(old_input); tags=["identity-renamed-in"])

        @test TN_OPMUT.replace_operator_input_indices!(op, [metadata_copy], [new_input]) === op
        @test op.input_indices[2] == new_input
        @test new_input in _opmutation_siteinds_by_tensor(op.mpo)[2]
    end
end
