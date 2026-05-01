using Test

function _load_build_script_for_config_tests()
    mod = Module(:Tensor4allBuildConfigUnderTest)
    build_jl = normpath(joinpath(@__DIR__, "..", "..", "deps", "build.jl"))
    withenv("TENSOR4ALL_BUILD_CONFIG_ONLY" => "1") do
        Base.include(mod, build_jl)
    end
    return mod
end

const BuildScript = _load_build_script_for_config_tests()

@testset "deps build linalg backend selection" begin
    withenv("TENSOR4ALL_LINALG_BACKEND" => nothing, "TENSOR4ALL_RS_FEATURES" => nothing) do
        @test BuildScript.selected_linalg_backend() == :julia_blas
        @test BuildScript.cargo_feature_args() ==
              ["--no-default-features", "--features", "tenferro-provider-inject"]
    end

    withenv("TENSOR4ALL_LINALG_BACKEND" => "faer", "TENSOR4ALL_RS_FEATURES" => nothing) do
        @test BuildScript.selected_linalg_backend() == :faer
        @test BuildScript.cargo_feature_args() == String[]
    end

    withenv("TENSOR4ALL_LINALG_BACKEND" => "faer", "TENSOR4ALL_RS_FEATURES" => "foo,bar baz") do
        @test BuildScript.cargo_feature_args() ==
              ["--no-default-features", "--features", "tenferro-cpu-faer,foo,bar,baz"]
    end

    withenv("TENSOR4ALL_LINALG_BACKEND" => "julia-blas", "TENSOR4ALL_RS_FEATURES" => nothing) do
        @test BuildScript.selected_linalg_backend() == :julia_blas
        @test BuildScript.cargo_feature_args() ==
              ["--no-default-features", "--features", "tenferro-provider-inject"]
    end

    withenv("TENSOR4ALL_LINALG_BACKEND" => "inject", "TENSOR4ALL_RS_FEATURES" => "extra") do
        @test BuildScript.selected_linalg_backend() == :julia_blas
        @test BuildScript.cargo_feature_args() ==
              ["--no-default-features", "--features", "tenferro-provider-inject,extra"]
    end

    withenv("TENSOR4ALL_LINALG_BACKEND" => "unknown", "TENSOR4ALL_RS_FEATURES" => nothing) do
        @test_throws ErrorException BuildScript.selected_linalg_backend()
    end
end
