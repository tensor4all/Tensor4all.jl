using Test
using Tensor4all

const TN_TP = Tensor4all.TensorNetworks

@testset "SvdTruncationPolicy (strategy-only)" begin
    @testset "default construction" begin
        p = TN_TP.SvdTruncationPolicy()
        @test p.scale === :relative
        @test p.measure === :value
        @test p.rule === :per_value
        # No threshold field on the type any more.
        @test !hasfield(TN_TP.SvdTruncationPolicy, :threshold)
    end

    @testset "keyword construction" begin
        p = TN_TP.SvdTruncationPolicy(;
            scale=:absolute,
            measure=:squared_value,
            rule=:discarded_tail_sum,
        )
        @test p.scale === :absolute
        @test p.measure === :squared_value
        @test p.rule === :discarded_tail_sum
    end

    @testset "invalid symbols raise ArgumentError" begin
        @test_throws ArgumentError TN_TP.SvdTruncationPolicy(; scale=:bogus)
        @test_throws ArgumentError TN_TP.SvdTruncationPolicy(; measure=:bogus)
        @test_throws ArgumentError TN_TP.SvdTruncationPolicy(; rule=:bogus)
    end
end

@testset "_resolve_svd_policy" begin
    @testset "threshold==0 returns nothing" begin
        @test TN_TP._resolve_svd_policy(; threshold=0.0, svd_policy=nothing) === nothing
    end

    @testset "threshold>0 with nothing uses default" begin
        ffi = TN_TP._resolve_svd_policy(; threshold=1e-6, svd_policy=nothing)
        @test ffi !== nothing
        @test ffi.threshold == 1e-6
        @test ffi.scale == TN_TP._T4A_THRESHOLD_SCALE_RELATIVE
        @test ffi.measure == TN_TP._T4A_SINGULAR_VALUE_MEASURE_VALUE
        @test ffi.rule == TN_TP._T4A_TRUNCATION_RULE_PER_VALUE
    end

    @testset "explicit svd_policy" begin
        p = TN_TP.SvdTruncationPolicy(;
            scale=:absolute, measure=:squared_value, rule=:discarded_tail_sum,
        )
        ffi = TN_TP._resolve_svd_policy(; threshold=1e-4, svd_policy=p)
        @test ffi !== nothing
        @test ffi.threshold == 1e-4
        @test ffi.scale == TN_TP._T4A_THRESHOLD_SCALE_ABSOLUTE
        @test ffi.measure == TN_TP._T4A_SINGULAR_VALUE_MEASURE_SQUARED_VALUE
        @test ffi.rule == TN_TP._T4A_TRUNCATION_RULE_DISCARDED_TAIL_SUM
    end

    @testset "negative threshold" begin
        @test_throws ArgumentError TN_TP._resolve_svd_policy(;
            threshold=-1.0, svd_policy=nothing,
        )
    end
end

@testset "nothing threshold and maxdim normalize to no truncation" begin
    @test TN_TP._normalize_threshold(nothing) == 0.0
    @test TN_TP._normalize_threshold(1e-8) == 1e-8
    @test TN_TP._normalize_maxdim(nothing) == 0
    @test TN_TP._normalize_maxdim(7) == 7
    @test TN_TP._resolve_svd_policy(; threshold=nothing, svd_policy=nothing) === nothing
    @test_throws ArgumentError TN_TP._normalize_threshold(-1.0)
    @test_throws ArgumentError TN_TP._normalize_maxdim(-1)
end

@testset "default policy registry" begin
    original = TN_TP.default_svd_policy()
    try
        @testset "built-in default" begin
            @test original.scale === :relative
            @test original.measure === :value
            @test original.rule === :per_value
        end

        @testset "set_default_svd_policy!" begin
            p = TN_TP.SvdTruncationPolicy(;
                measure=:squared_value, rule=:discarded_tail_sum,
            )
            TN_TP.set_default_svd_policy!(p)
            @test TN_TP.default_svd_policy() === p
        end

        @testset "with_svd_policy scoped override" begin
            TN_TP.set_default_svd_policy!(original)
            p = TN_TP.SvdTruncationPolicy(; scale=:absolute)
            result = TN_TP.with_svd_policy(p) do
                @test TN_TP.default_svd_policy() === p
                return 42
            end
            @test result == 42
            @test TN_TP.default_svd_policy() === original
        end

        @testset "with_svd_policy nesting" begin
            TN_TP.set_default_svd_policy!(original)
            outer = TN_TP.SvdTruncationPolicy(; measure=:squared_value)
            inner = TN_TP.SvdTruncationPolicy(; scale=:absolute)
            TN_TP.with_svd_policy(outer) do
                @test TN_TP.default_svd_policy() === outer
                TN_TP.with_svd_policy(inner) do
                    @test TN_TP.default_svd_policy() === inner
                end
                @test TN_TP.default_svd_policy() === outer
            end
            @test TN_TP.default_svd_policy() === original
        end

        @testset "scope observed by resolver" begin
            p = TN_TP.SvdTruncationPolicy(; rule=:discarded_tail_sum)
            TN_TP.with_svd_policy(p) do
                ffi = TN_TP._resolve_svd_policy(; threshold=1e-6, svd_policy=nothing)
                @test ffi.rule == TN_TP._T4A_TRUNCATION_RULE_DISCARDED_TAIL_SUM
            end
        end

        @testset "parallel tasks are isolated" begin
            TN_TP.set_default_svd_policy!(original)
            pa = TN_TP.SvdTruncationPolicy(; rule=:discarded_tail_sum)
            pb = TN_TP.SvdTruncationPolicy(; scale=:absolute)
            results = fill(TN_TP.SvdTruncationPolicy(), 2)
            @sync begin
                Threads.@spawn TN_TP.with_svd_policy(pa) do
                    sleep(0.01)
                    results[1] = TN_TP.default_svd_policy()
                end
                Threads.@spawn TN_TP.with_svd_policy(pb) do
                    sleep(0.01)
                    results[2] = TN_TP.default_svd_policy()
                end
            end
            @test results[1] === pa
            @test results[2] === pb
        end
    finally
        TN_TP.set_default_svd_policy!(original)
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

@testset "QTT layout enum" begin
    @test TN_TP._T4A_QTT_LAYOUT_INTERLEAVED === Cint(0)
    @test TN_TP._T4A_QTT_LAYOUT_FUSED === Cint(1)
end
