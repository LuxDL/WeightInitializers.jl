using WeightInitializers, Test, SafeTestsets, StableRNGs, Statistics

const rng = StableRNG(12345)

@testset "WeightInitializers.jl Tests" begin
    @testset "_nfan" begin
        # Fallback
        @test WeightInitializers._nfan() == (1, 1)
        # Vector
        @test WeightInitializers._nfan(4) == (1, 4)
        # Matrix
        @test WeightInitializers._nfan(4, 5) == (5, 4)
        # Tuple
        @test WeightInitializers._nfan((4, 5, 6)) == WeightInitializers._nfan(4, 5, 6)
        # Convolution
        @test WeightInitializers._nfan(4, 5, 6) == 4 .* (5, 6)
    end

    @testset "Sizes and Types: $init" for init in [
        zeros32,
        ones32,
        rand32,
        randn32,
        kaiming_uniform,
        kaiming_normal,
        glorot_uniform,
        glorot_normal,
        truncated_normal,
    ]
        # Sizes
        @test size(init(3)) == (3,)
        @test size(init(rng, 3)) == (3,)
        @test size(init(3, 4)) == (3, 4)
        @test size(init(rng, 3, 4)) == (3, 4)
        @test size(init(3, 4, 5)) == (3, 4, 5)
        @test size(init(rng, 3, 4, 5)) == (3, 4, 5)
        # Type
        @test eltype(init(rng, 4, 2)) == Float32
        @test eltype(init(4, 2)) == Float32
    end

    @testset "Closure: $init" for init in [
        kaiming_uniform,
        kaiming_normal,
        glorot_uniform,
        glorot_normal,
        truncated_normal,
    ]
        cl = init(;)
        # Sizes
        @test size(cl(3)) == (3,)
        @test size(cl(rng, 3)) == (3,)
        @test size(cl(3, 4)) == (3, 4)
        @test size(cl(rng, 3, 4)) == (3, 4)
        @test size(cl(3, 4, 5)) == (3, 4, 5)
        @test size(cl(rng, 3, 4, 5)) == (3, 4, 5)
        # Type
        @test eltype(cl(4, 2)) == Float32
        @test eltype(cl(rng, 4, 2)) == Float32
    end

    @testset "kaiming" begin
        # kaiming_uniform should yield a kernel in range [-sqrt(6/n_out), sqrt(6/n_out)]
        # and kaiming_normal should yield a kernel with stddev ~= sqrt(2/n_out)
        for (n_in, n_out) in [(100, 100), (100, 400)]
            v = kaiming_uniform(rng, n_in, n_out)
            σ2 = sqrt(6 / n_out)
            @test -1σ2 < minimum(v) < -0.9σ2
            @test 0.9σ2 < maximum(v) < 1σ2

            v = kaiming_normal(rng, n_in, n_out)
            σ2 = sqrt(2 / n_out)
            @test 0.9σ2 < std(v) < 1.1σ2
        end
        # Type
        @test eltype(kaiming_uniform(rng, 3, 4; gain=1.5)) == Float32
        @test eltype(kaiming_normal(rng, 3, 4; gain=1.5)) == Float32
    end

    @testset "glorot: $init" for init in [glorot_uniform, glorot_normal]
        # glorot_uniform and glorot_normal should both yield a kernel with
        # variance ≈ 2/(fan_in + fan_out)
        for dims in [(1000,), (100, 100), (100, 400), (2, 3, 32, 64), (2, 3, 4, 32, 64)]
            v = init(dims...)
            fan_in, fan_out = WeightInitializers._nfan(dims...)
            σ2 = 2 / (fan_in + fan_out)
            @test 0.9σ2 < var(v) < 1.1σ2
        end
        @test eltype(init(3, 4; gain=1.5)) == Float32
    end
end
