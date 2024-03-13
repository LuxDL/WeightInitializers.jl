module WeightInitializersCUDAExt

using WeightInitializers, CUDA
using Random
import WeightInitializers: __partial_apply, NUM_TO_FPOINT, identity_init, sparse_init,
                           orthogonal, delay_line, delay_line_backward

const AbstractCuRNG = Union{CUDA.RNG, CURAND.RNG}

for T in ("16", "32", "64", "C16", "C32", "C64"), fname in (:ones, :zeros)
    name = Symbol(fname, T)
    TP = NUM_TO_FPOINT[Symbol(T)]
    @eval begin
        function WeightInitializers.$(name)(rng::AbstractCuRNG, dims::Integer...; kwargs...)
            return CUDA.$(fname)($TP, dims...; kwargs...)
        end
    end

    @eval function WeightInitializers.$(name)(rng::AbstractCuRNG; kwargs...)
        return __partial_apply($name, (rng, (; kwargs...)))
    end
end

function sparse_init(rng::AbstractCuRNG, ::Type{T}, dims::Integer...;
        sparsity::Number, std::Number=T(0.01)) where {T <: Number}
    if length(dims) != 2
        throw(ArgumentError("Only 2-dimensional outputs are supported for sparse initialization."))
    end

    rows, cols = dims
    prop_zero = min(1.0, sparsity)
    num_zeros = ceil(Integer, prop_zero * rows)
    sparse_array = randn(rng, T, dims...) .* T(std)
    sparse_array[1:num_zeros, :] .= CUDA.zero(T)

    return CUDA.@allowscalar mapslices(shuffle, sparse_array, dims=1)
end

function identity_init(rng::AbstractCuRNG, ::Type{T}, dims::Integer...;
        gain::Number=1, shift::Integer=0) where {T <: Number}
    if length(dims) == 1
        # Bias initialization
        return CUDA.zeros(T, dims...)
    elseif length(dims) == 2
        # Matrix multiplication
        rows, cols = dims
        mat = CUDA.zeros(T, rows, cols)
        diag_indices = 1:min(rows, cols)
        CUDA.fill!(view(mat, diag_indices, diag_indices), T(gain))
        return CUDA.circshift(mat, shift)
    else
        # Convolution or more dimensions
        nin, nout = dims[end - 1], dims[end]
        centers = map(d -> cld(d, 2), dims[1:(end - 2)])
        weights = CUDA.zeros(T, dims...)
        #we should really find a better way to do this
        CUDA.@allowscalar for i in 1:min(nin, nout)
            index = (centers..., i, i)
            weights[index...] = T(gain)
        end
        return CUDA.circshift(weights, (ntuple(d -> 0, length(dims) - 2)..., shift, shift))
    end
end

# rc initializers

function delay_line(rng::AbstractCuRNG,
    ::Type{T},
    dims::Integer...;
    weight = T(0.1)) where {T <: Number}
    reservoir_matrix = CUDA.zeros(T, dims...)
    @assert length(dims) == 2&&dims[1] == dims[2] "The dimensions must define a square matrix (e.g., (100, 100))"

    for i in 1:(dims[1] - 1)
        reservoir_matrix[i + 1, i] = T(weight)
    end

    return reservoir_matrix
end

function delay_line_backward(rng::AbstractCuRNG,
    ::Type{T},
    dims::Integer...;
    weight = T(0.1),
    fb_weight = T(0.2)) where {T <: Number}
    res_size = first(dims)
    reservoir_matrix = CUDA.zeros(T, dims...)

    for i in 1:(res_size - 1)
        reservoir_matrix[i + 1, i] = T(weight)
        reservoir_matrix[i, i + 1] = T(fb_weight)
    end

    return reservoir_matrix
end

for initializer in (:sparse_init, :identity_init, :delay_line, :delay_line_backward)
    @eval function ($initializer)(rng::AbstractCuRNG, dims::Integer...; kwargs...)
        return $initializer(rng, Float32, dims...; kwargs...)
    end

    @eval function ($initializer)(rng::AbstractCuRNG; kwargs...)
        return __partial_apply($initializer, (rng, (; kwargs...)))
    end
    @eval function ($initializer)(rng::AbstractCuRNG,
            ::Type{T}; kwargs...) where {T <: Number}
        return __partial_apply($initializer, ((rng, T), (; kwargs...)))
    end
end

end
