using Test

@testset "build.jl" begin
    script = read(joinpath(dirname(@__DIR__), "deps", "build.jl"), String)

    @test occursin("const TENSOR4ALL_RS_FALLBACK_COMMIT = \"3f05ea81177c64b5f351b99fdfd23325e732fc62\"", script)
    @test occursin("checkout --detach", script)
    @test !occursin("--branch", script)
end
