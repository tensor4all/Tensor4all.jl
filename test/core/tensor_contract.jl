using Test
using Tensor4all

@testset "Tensor contraction via C API" begin
    i = Index(2; tags=["i"])
    j = Index(3; tags=["j"])
    k = Index(4; tags=["k"])

    @testset "matrix-vector contraction" begin
        A = Tensor(reshape(collect(1.0:6.0), 2, 3), [i, j])
        v = Tensor(collect(1.0:3.0), [j])
        result = contract(A, v)
        @test rank(result) == 1
        @test dims(result) == (2,)
        expected = reshape(collect(1.0:6.0), 2, 3) * collect(1.0:3.0)
        @test copy_data(result) ≈ expected
    end

    @testset "matrix-matrix contraction" begin
        A = Tensor(reshape(collect(1.0:6.0), 2, 3), [i, j])
        B = Tensor(reshape(collect(1.0:12.0), 3, 4), [j, k])
        result = contract(A, B)
        @test rank(result) == 2
        expected = reshape(collect(1.0:6.0), 2, 3) * reshape(collect(1.0:12.0), 3, 4)
        @test copy_data(result) ≈ expected
    end

    @testset "backend contraction result materializes lazily" begin
        A = Tensor(reshape(collect(1.0:6.0), 2, 3), [i, j])
        B = Tensor(reshape(collect(1.0:12.0), 3, 4), [j, k])

        result = contract(A, B)

        @test fieldnames(typeof(result)) == (:handle, :inds)
        @test result.backend_handle !== nothing
        @test dims(result) == (2, 4)
        materialized = Array(result, i, k)
        @test materialized ≈ reshape(collect(1.0:6.0), 2, 3) * reshape(collect(1.0:12.0), 3, 4)
        materialized[1, 1] = -1.0
        @test Array(result, i, k) ≈ reshape(collect(1.0:6.0), 2, 3) * reshape(collect(1.0:12.0), 3, 4)
        @test fieldnames(typeof(result)) == (:handle, :inds)
        @test structured_storage_info(result).kind == :dense
        @test fieldnames(typeof(result)) == (:handle, :inds)
    end

    @testset "selectinds fixes multiple indices lazily" begin
        data = reshape(collect(1.0:24.0), 2, 3, 4)
        t = Tensor(data, [i, j, k])

        selected = selectinds(t, j => 2, k => 3)

        @test fieldnames(typeof(selected)) == (:handle, :inds)
        @test selected.backend_handle !== nothing
        @test inds(selected) == [i]
        @test dims(selected) == (2,)
        selected_data = copy_data(selected, i)
        @test selected_data ≈ data[:, 2, 3]
        selected_data[1] = -1.0
        @test copy_data(selected, i) ≈ data[:, 2, 3]
        @test fieldnames(typeof(selected)) == (:handle, :inds)
    end

    @testset "outer product (no shared indices)" begin
        a = Tensor(collect(1.0:2.0), [i])
        b = Tensor(collect(1.0:3.0), [j])
        result = contract(a, b)
        @test rank(result) == 2
        @test inds(result) == [i, j]
        @test copy_data(result) ≈ collect(1.0:2.0) * transpose(collect(1.0:3.0))
    end

    @testset "index tags round-trip through backend with ITensors rules" begin
        csv_tag = Index(2; tags=["d=1,r=1"])
        unicode_space_tag = Index(3, "a\u3000b")
        a = Tensor(collect(1.0:2.0), [csv_tag])
        b = Tensor(collect(1.0:3.0), [unicode_space_tag])

        result = contract(a, b)
        @test inds(result) == [csv_tag, unicode_space_tag]
        @test tags(inds(result)[1]) == ["d=1", "r=1"]
        @test tags(inds(result)[2]) == ["a\u3000b"]
    end
end
