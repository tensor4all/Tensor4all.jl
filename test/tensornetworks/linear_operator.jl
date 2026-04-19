using Test
using Tensor4all

const TN = Tensor4all.TensorNetworks

@testset "Base.transpose(::LinearOperator)" begin
    # Build a minimal LinearOperator fixture with distinguishable indices.
    s_in = Index(2; tags=["in"])
    s_out = Index(3; tags=["out"])
    true_in = Index(2; tags=["true_in"])
    true_out = Index(3; tags=["true_out"])
    op = TN.LinearOperator(;
        mpo=nothing,
        input_indices=[s_in],
        output_indices=[s_out],
        true_input=Union{Index, Nothing}[true_in],
        true_output=Union{Index, Nothing}[true_out],
        metadata=(; kind=:test),
    )

    @testset "swap" begin
        t = transpose(op)
        @test t.input_indices == [s_out]
        @test t.output_indices == [s_in]
        @test t.true_input == [true_out]
        @test t.true_output == [true_in]
        @test t.metadata == op.metadata
    end

    @testset "involution" begin
        tt = transpose(transpose(op))
        @test tt.input_indices == op.input_indices
        @test tt.output_indices == op.output_indices
        @test tt.true_input == op.true_input
        @test tt.true_output == op.true_output
        @test tt.metadata == op.metadata
    end
end
