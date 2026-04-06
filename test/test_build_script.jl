using Test

@testset "build.jl" begin
    script = read(joinpath(dirname(@__DIR__), "deps", "build.jl"), String)

    @test occursin(r"const TENSOR4ALL_RS_FALLBACK_COMMIT = \"[0-9a-f]{40}\"", script)
    @test occursin("checkout --detach", script)
    @test !occursin("--branch", script)
end
