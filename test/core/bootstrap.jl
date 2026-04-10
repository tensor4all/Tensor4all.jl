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
