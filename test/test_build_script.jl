using Test

@testset "build.jl" begin
    script = read(joinpath(dirname(@__DIR__), "deps", "build.jl"), String)

    @test occursin("const TENSOR4ALL_RS_FALLBACK_COMMIT = \"44ddedb3ff801f80c2f8d1609bf43eb28081a9f1\"", script)
    @test occursin("checkout --detach", script)
    @test !occursin("--branch", script)
end
