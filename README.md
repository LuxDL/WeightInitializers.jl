# WeightInitializers

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://lux.csail.mit.edu/dev/api/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://lux.csail.mit.edu/stable/api/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

[![Build status](https://badge.buildkite.com/ffa2c8c3629cd58322446cddd3e8dcc4f121c28a574ee3e626.svg?branch=main)](https://buildkite.com/julialang/weightinitializers-dot-jl)
[![CI](https://github.com/LuxDL/WeightInitializers.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LuxDL/WeightInitializers.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/LuxDL/WeightInitializers.jl/branch/main/graph/badge.svg?token=1ZY0A2NPEM)](https://codecov.io/gh/LuxDL/WeightInitializers.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/WeightInitializers)](https://pkgs.genieframework.com?packages=WeightInitializers)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

This package is a light dependency providing common weight initialization schemes for deep
learning models. It provides a flexible interface that accepts `AbstractRNG`s and allows
for multiple output types (`Float64`, `Float32`, `Float16` etc...)

## Example

These code snippets are just provided to give a high level overview of the functionalities
of the package.

```julia
using WeightInitializers, Random

# Fixing rng
rng = MersenneTwister(42)

# Explicit rng and Float32 call
weights = kaiming_normal(rng, Float32, 2, 5)
#2×5 Matrix{Float32}:
# -0.351662   0.0171745   1.12442   -0.296372   -1.67094
# -0.281053  -0.18941    -0.724099   0.0987538   0.634549

# Explicit rng call
weights = kaiming_normal(rng, 2, 5)
#2×5 Matrix{Float32}:
# 0.684558  0.327706   0.232467   0.432957   0.25972
# 0.118287  0.943231  -0.560485  -1.00597   -0.541603

# Explicit Float64 call
weights = kaiming_normal(Float64, 2, 5)
#2×5 Matrix{Float64}:
#  0.613897   0.570387   -0.379974  1.71233   -0.130052
# -0.619312  -0.0207465  -0.91401   0.964145   0.487435

# Default rng call
weights = kaiming_normal(2, 5)
#2×5 Matrix{Float32}:
# -0.227513  -0.265372   0.265788  1.29955  -0.192836
#  0.687611   0.454679  -0.433656  0.20548   0.292002

# Passing kwargs (if needed) with explicit rng call
weights_cl = kaiming_normal(rng; gain=1.0)
weights = weights_cl(2, 5)
#2×5 Matrix{Float64}:
# -0.470016  -0.096709  -1.93977   -0.635964   0.165026
#  0.224537  -0.315923   0.254835  -0.166543  -0.00340463

# Passing kwargs (if needed) with explicit Float16 call
weights_cl = kaiming_normal(Float16; gain=1.0)
weights = weights_cl(2, 5)
2×5 Matrix{Float64}:
 -0.160808  -0.187664   0.187882  0.918777  -0.136354
  0.486026   0.321397  -0.30655   0.145306   0.206441

# Passing kwargs (if needed) with default call
weights_cl = kaiming_normal(; gain=1.0)
weights = weights_cl(2, 5)
#2×5 Matrix{Float32}:
# -0.160876  -0.187646   0.18794   0.918918  -0.136356
#  0.486214   0.321506  -0.306641  0.145296   0.206476
```

## API

The package is meant to be working with deep learning libraries such as F/Lux. All the
methods take as input the chosen `rng` type and the dimension for the AbstractArray.

```julia
weights = init(rng, dims...)
```

The `rng` is optional, if not specified a default one will be used.

```julia
weights = init(dims...)
```

If there is the need to use keyword arguments the methods can be called with just the `rng`
(optionally) and the keywords to get in return a function behaving like the two examples
above.

```julia
weights_init = init(rng; kwargs...)
weights = weights_init(rng, dims...)
# or
weights_init = init(; kwargs...)
weights = weights_init(dims...)
```
