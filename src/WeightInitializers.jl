module WeightInitializers

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ChainRulesCore, PartialFunctions, Random, SpecialFunctions, Statistics,
          LinearAlgebra
end

include("utils.jl")
include("initializers.jl")
include("rc_initializers.jl")

# Mark the functions as non-differentiable
for f in [
    :zeros64,
    :ones64,
    :rand64,
    :randn64,
    :zeros32,
    :ones32,
    :rand32,
    :randn32,
    :zeros16,
    :ones16,
    :rand16,
    :randn16,
    :zerosC64,
    :onesC64,
    :randC64,
    :randnC64,
    :zerosC32,
    :onesC32,
    :randC32,
    :randnC32,
    :zerosC16,
    :onesC16,
    :randC16,
    :randnC16,
    :glorot_normal,
    :glorot_uniform,
    :kaiming_normal,
    :kaiming_uniform,
    :truncated_normal,
    :orthogonal,
    :sparse_init,
    :identity_init,
    :rand_sparse,
    :delay_line,
    :delay_line_backward,
    :cycle_jumps,
    :simple_cycle,
    :pseudo_svd,
    :scaled_rand,
    :weighted_init,
    :informed_init,
    :minimal_init
]
    @eval @non_differentiable $(f)(::Any...)
end

export zeros64, ones64, rand64, randn64, zeros32, ones32, rand32, randn32, zeros16, ones16,
       rand16, randn16
export zerosC64, onesC64, randC64, randnC64, zerosC32, onesC32, randC32, randnC32, zerosC16,
       onesC16, randC16, randnC16
export glorot_normal, glorot_uniform
export kaiming_normal, kaiming_uniform
export truncated_normal
export orthogonal
export sparse_init
export identity_init
export scaled_rand, weighted_init, informed_init, minimal_init
export rand_sparse, delay_line, delay_line_backward, cycle_jumps, simple_cycle, pseudo_svd

end
