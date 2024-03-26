using WeightInitializers, Test, SafeTestsets, Statistics
using StableRNGs, Random, CUDA, LinearAlgebra

CUDA.allowscalar(false)

const GROUP = get(ENV, "GROUP", "All")

rngs_arrtypes = []

if GROUP == "All" || GROUP == "CPU"
    append!(rngs_arrtypes,
        [(StableRNG(12345), AbstractArray), (Random.default_rng(), AbstractArray)])
end

if GROUP == "All" || GROUP == "CUDA"
    append!(rngs_arrtypes, [(CUDA.default_rng(), CuArray)])
end

const res_size = 30
const in_size = 3
const radius = 1.0
const sparsity = 0.1
const weight = 0.2
const jump_size = 3
const rng = Random.default_rng()

function check_radius(matrix, target_radius; tolerance = 1e-5)
    eigenvalues = eigvals(matrix)
    spectral_radius = maximum(abs.(eigenvalues))
    return isapprox(spectral_radius, target_radius, atol = tolerance)
end

ft = [Float16, Float32, Float64]
reservoir_inits = [
    rand_sparse,
    delay_line,
    delay_line_backward,
    cycle_jumps,
    simple_cycle,
    pseudo_svd
]
input_inits = [
    scaled_rand,
    weighted_init,
    minimal_init,
    minimal_init(; sampling_type = :irrational)
]

@testset "Reservoirs rng = $(typeof(rng)) & arrtype = $arrtype" for (rng, arrtype) in rngs_arrtypes
    @testset "Sizes and types: $init" for init in reservoir_inits
        #sizes
        @test size(init(res_size, res_size)) == (res_size, res_size)
        @test size(init(rng, res_size, res_size)) == (res_size, res_size)
        #types
        @test eltype(init(res_size, res_size)) == Float32
        @test eltype(init(rng, res_size, res_size)) == Float32
        #closure
        cl = init(rng)
        @test cl(res_size, res_size) isa arrtype{Float32, 2}
    end

    @testset "AbstractArray Type: $init $T" for init in reservoir_inits, T in ft
        @test init(rng, T, res_size, res_size) isa arrtype{T, 2}

        cl = init(rng)
        @test cl(T, res_size, res_size) isa arrtype{T, 2}

        cl = init(rng, T)
        @test cl(res_size, res_size) isa arrtype{T, 2}
    end

    @testset "Kwargs types" for T in ft
        @test eltype(rand_sparse(T, res_size, res_size; radius=1.0, sparsity=0.2, std=1.0)) == T
        @test eltype(delay_line(T, res_size, res_size; weight=1.0)) == T
        @test eltype(delay_line_backward(T, res_size, res_size; weight=1.0, fb_weight=1.0)) == T
        @test eltype(cycle_jumps(T, res_size, res_size; cycle_weight=1.0, jump_weight=1.0)) == T
        @test eltype(delay_line(T, res_size, res_size; weight=1.0)) == T
        @test eltype(pseudo_svd(T, res_size, res_size; max_value=1.0)) == T
    end

    @testset "rand_sparse spectral radius" begin
        sp = rand_sparse(res_size, res_size)
        @test check_radius(sp, radius)
    end

    @testset "Minimum complexity: $init" for init in [
        delay_line,
        delay_line_backward,
        cycle_jumps,
        simple_cycle
    ]
        dl = init(res_size, res_size)
        if init === delay_line_backward
            @test unique(dl) == Float32.([0.0, 0.1, 0.2])
        else
            @test unique(dl) == Float32.([0.0, 0.1])
        end
    end
end

# TODO: @MartinuzziFrancesco Missing tests for informed_init
@testset "Input initializers rng = $(typeof(rng)) & arrtype = $arrtype" for (rng, arrtype) in rngs_arrtypes
    @testset "Sizes and types: $init" for init in input_inits
        #sizes
        @test size(init(res_size, in_size)) == (res_size, in_size)
        @test size(init(rng, res_size, in_size)) == (res_size, in_size)
        #types
        @test eltype(init(res_size, in_size)) == Float32
        @test eltype(init(rng, res_size, in_size)) == Float32
        #closure
        cl = init(rng)
        @test eltype(cl(res_size, in_size)) == arrtype{Float32, 2}
    end

    @testset "AbstractArray Type: $init $T" for init in input_inits, T in ft
        @test init(rng, T, res_size, in_size) isa arrtype{T, 2}

        cl = init(rng)
        @test cl(T, res_size, in_size) isa arrtype{T, 2}

        cl = init(rng, T)
        @test cl(res_size, in_size) isa arrtype{T, 2}
    end

    @testset "Kwargs types: $T" for T in ft
        @test eltype(scaled_rand(T, res_size, in_size; scaling=1.0)) == T
        @test eltype(weighted_init(T, res_size, in_size; scaling=1.0)) == T
        @test eltype(minimal_init(T, res_size, in_size; weight=1.0)) == T
    end

    @testset "Minimum complexity: $init" for init in [
        minimal_init,
        minimal_init(; sampling_type = :irrational)
    ]
        dl = init(res_size, in_size)
        @test sort(unique(dl)) == Float32.([-0.1, 0.1])
    end
end
