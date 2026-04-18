using Test
using Tensor4all

const TN_TP = Tensor4all.TensorNetworks

@testset "SvdTruncationPolicy" begin
    @testset "default construction" begin
        p = TN_TP.SvdTruncationPolicy()
        @test p.threshold == 0.0
        @test p.scale === :relative
        @test p.measure === :value
        @test p.rule === :per_value
    end

    @testset "keyword construction" begin
        p = TN_TP.SvdTruncationPolicy(;
            threshold=1e-8,
            scale=:absolute,
            measure=:squared_value,
            rule=:discarded_tail_sum,
        )
        @test p.threshold == 1e-8
        @test p.scale === :absolute
        @test p.measure === :squared_value
        @test p.rule === :discarded_tail_sum
    end

    @testset "invalid symbols raise ArgumentError" begin
        @test_throws ArgumentError TN_TP.SvdTruncationPolicy(; scale=:bogus)
        @test_throws ArgumentError TN_TP.SvdTruncationPolicy(; measure=:bogus)
        @test_throws ArgumentError TN_TP.SvdTruncationPolicy(; rule=:bogus)
        @test_throws ArgumentError TN_TP.SvdTruncationPolicy(; threshold=-1.0)
    end
end

@testset "_resolve_svd_policy" begin
    @testset "no truncation returns nothing" begin
        @test TN_TP._resolve_svd_policy(; rtol=0.0, cutoff=0.0, svd_policy=nothing) === nothing
    end

    @testset "rtol convenience path" begin
        ffi = TN_TP._resolve_svd_policy(; rtol=1e-6, cutoff=0.0, svd_policy=nothing)
        @test ffi !== nothing
        @test ffi.threshold == 1e-6
        @test ffi.scale == TN_TP._T4A_THRESHOLD_SCALE_RELATIVE
        @test ffi.measure == TN_TP._T4A_SINGULAR_VALUE_MEASURE_VALUE
        @test ffi.rule == TN_TP._T4A_TRUNCATION_RULE_PER_VALUE
    end

    @testset "cutoff takes precedence and applies sqrt" begin
        ffi = TN_TP._resolve_svd_policy(; rtol=0.1, cutoff=0.04, svd_policy=nothing)
        @test ffi !== nothing
        @test ffi.threshold ≈ 0.2  # sqrt(0.04)
        @test ffi.scale == TN_TP._T4A_THRESHOLD_SCALE_RELATIVE
        @test ffi.measure == TN_TP._T4A_SINGULAR_VALUE_MEASURE_VALUE
        @test ffi.rule == TN_TP._T4A_TRUNCATION_RULE_PER_VALUE
    end

    @testset "svd_policy passthrough" begin
        p = TN_TP.SvdTruncationPolicy(;
            threshold=1e-4,
            scale=:absolute,
            measure=:squared_value,
            rule=:discarded_tail_sum,
        )
        ffi = TN_TP._resolve_svd_policy(; rtol=0.0, cutoff=0.0, svd_policy=p)
        @test ffi !== nothing
        @test ffi.threshold == 1e-4
        @test ffi.scale == TN_TP._T4A_THRESHOLD_SCALE_ABSOLUTE
        @test ffi.measure == TN_TP._T4A_SINGULAR_VALUE_MEASURE_SQUARED_VALUE
        @test ffi.rule == TN_TP._T4A_TRUNCATION_RULE_DISCARDED_TAIL_SUM
    end

    @testset "ambiguity error" begin
        p = TN_TP.SvdTruncationPolicy(; threshold=1e-6)
        @test_throws ArgumentError TN_TP._resolve_svd_policy(;
            rtol=1e-8, cutoff=0.0, svd_policy=p,
        )
        @test_throws ArgumentError TN_TP._resolve_svd_policy(;
            rtol=0.0, cutoff=1e-8, svd_policy=p,
        )
    end
end

@testset "policy FFI enum constants" begin
    @test TN_TP._T4A_THRESHOLD_SCALE_RELATIVE === Cint(0)
    @test TN_TP._T4A_THRESHOLD_SCALE_ABSOLUTE === Cint(1)
    @test TN_TP._T4A_SINGULAR_VALUE_MEASURE_VALUE === Cint(0)
    @test TN_TP._T4A_SINGULAR_VALUE_MEASURE_SQUARED_VALUE === Cint(1)
    @test TN_TP._T4A_TRUNCATION_RULE_PER_VALUE === Cint(0)
    @test TN_TP._T4A_TRUNCATION_RULE_DISCARDED_TAIL_SUM === Cint(1)
end

@testset "QTT layout enum renumbered to match tensor4all-rs fd7180c" begin
    @test TN_TP._T4A_QTT_LAYOUT_INTERLEAVED === Cint(0)
    @test TN_TP._T4A_QTT_LAYOUT_FUSED === Cint(1)
end
