using Test
using Tensor4all

@testset "bootstrap errors and lazy backend helpers" begin
    @test Tensor4all.SKELETON_PHASE === true
    @test isdefined(Tensor4all, :SkeletonPhaseError)
    @test isdefined(Tensor4all, :SkeletonNotImplemented)
    @test isdefined(Tensor4all, :BackendUnavailableError)
    @test isdefined(Tensor4all, :backend_library_path)
    @test isdefined(Tensor4all, :require_backend)

    placeholder = Tensor4all.SkeletonNotImplemented(:contract, :core)
    @test sprint(showerror, placeholder) ==
        "Tensor4all skeleton phase: `contract` is planned in the `core` layer but not implemented yet."

    missing = Tensor4all.BackendUnavailableError("backend missing")
    @test sprint(showerror, missing) == "backend missing"
    @test Tensor4all.backend_library_path() isa String
end

@testset "backend inject helper surface" begin
    interface = Tensor4all._inject_blas_interface()
    @test interface in (:lp64, :ilp64)
    @test :dgemm in Tensor4all._CBLAS_GEMM_INJECT_SYMBOLS
    @test :zgemm in Tensor4all._CBLAS_GEMM_INJECT_SYMBOLS
    @test :dgesvd in Tensor4all._LAPACK_INJECT_SYMBOLS
    @test :zgesvd in Tensor4all._LAPACK_INJECT_SYMBOLS
    @test :dgetc2 in Tensor4all._LAPACK_INJECT_SYMBOLS
    @test :zgesc2 in Tensor4all._LAPACK_INJECT_SYMBOLS
    @test !Tensor4all._inject_missing_provider_pointer(
        Tensor4all._inject_provider_pointer(:dgemm, interface),
    )
    @test Tensor4all._inject_missing_provider_pointer(C_NULL)
    @test Tensor4all._inject_missing_provider_pointer(Ptr{Cvoid}(typemax(UInt)))
end
