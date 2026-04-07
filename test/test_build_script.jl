using Test

@testset "build.jl" begin
    script = read(joinpath(dirname(@__DIR__), "deps", "build.jl"), String)

    @test occursin("const TENSOR4ALL_RS_FALLBACK_COMMIT = \"4ee57fee0a71d385576c11d42850304548c6949d\"", script)
    @test occursin("checkout --detach", script)
    @test !occursin("--branch", script)
end
