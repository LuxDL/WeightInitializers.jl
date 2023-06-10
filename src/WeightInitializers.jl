module WeightInitializers
using Random
using Statistics

include("inits.jl")

export zeros32, ones32, rand32, randn32
export glorot_normal, glorot_uniform
export kaiming_normal, kaiming_uniform

end