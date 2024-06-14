module WeightInitializersCUDAExt

using WeightInitializers, CUDA
using Random
import WeightInitializers: __partial_apply, NUM_TO_FPOINT, identity_init, sparse_init,
                           orthogonal, delay_line, delay_line_backward, rand_sparse,
                           cycle_jumps, simple_cycle, pseudo_svd

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

function rand_sparse(rng::AbstractCuRNG,
    ::Type{T},
    dims::Integer...;
    radius = T(1.0),
    sparsity = T(0.1),
    std = T(1.0)) where {T <: Number}
    lcl_sparsity = T(1) - T(sparsity) #consistency with current implementations
    reservoir_matrix = sparse_init(rng, T, dims...;
        sparsity = lcl_sparsity, std = T(std))
    CUDA.@allowscalar rho_w = maximum(abs.(eigvals(reservoir_matrix)))
    reservoir_matrix .*= T(radius) / rho_w
    if Inf in unique(reservoir_matrix) || -Inf in unique(reservoir_matrix)
        error("Sparsity too low for size of the matrix.
            Increase res_size or increase sparsity")
    end
    return reservoir_matrix
end

function delay_line(rng::AbstractCuRNG,
    ::Type{T},
    dims::Integer...;
    weight = T(0.1)) where {T <: Number}
    reservoir_matrix = CUDA.zeros(T, dims...)
    @assert length(dims) == 2&&dims[1] == dims[2] "The dimensions must define a square matrix (e.g., (100, 100))"

    CUDA.@allowscalar for i in 1:(dims[1] - 1)
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

    CUDA.@allowscalar for i in 1:(res_size - 1)
        reservoir_matrix[i + 1, i] = T(weight)
        reservoir_matrix[i, i + 1] = T(fb_weight)
    end

    return reservoir_matrix
end

function cycle_jumps(rng::AbstractCuRNG,
    ::Type{T},
    dims::Integer...;
    cycle_weight::Number = T(0.1),
    jump_weight::Number = T(0.1),
    jump_size::Int = 3) where {T <: Number}
    res_size = first(dims)
    reservoir_matrix = CUDA.zeros(T, dims...)

    CUDA.@allowscalar for i in 1:(res_size - 1)
        reservoir_matrix[i + 1, i] = T(cycle_weight)
    end

    CUDA.@allowscalar reservoir_matrix[1, res_size] = T(cycle_weight)

    CUDA.@allowscalar for i in 1:jump_size:(res_size - jump_size)
        tmp = (i + jump_size) % res_size
        if tmp == 0
            tmp = res_size
        end
        reservoir_matrix[i, tmp] = T(jump_weight)
        reservoir_matrix[tmp, i] = T(jump_weight)
    end

    return reservoir_matrix
end

function simple_cycle(rng::AbstractCuRNG,
    ::Type{T},
    dims::Integer...;
    weight = T(0.1)) where {T <: Number}
    reservoir_matrix = CUDA.zeros(T, dims...)

    CUDA.@allowscalar for i in 1:(dims[1] - 1)
        reservoir_matrix[i + 1, i] = T(weight)
    end

    CUDA.@allowscalar reservoir_matrix[1, dims[1]] = T(weight)
    return reservoir_matrix
end

function pseudo_svd(rng::AbstractCuRNG,
    ::Type{T},
    dims::Integer...;
    max_value::Number = T(1.0),
    sparsity::Number = 0.1,
    sorted::Bool = true,
    reverse_sort::Bool = false) where {T <: Number}
    CUDA.@allowscalar reservoir_matrix = WeightInitializers.create_diag(dims[1],
        max_value,
        T;
        sorted = sorted,
        reverse_sort = reverse_sort)
    tmp_sparsity = WeightInitializers.get_sparsity(reservoir_matrix, dims[1])

    CUDA.@allowscalar while tmp_sparsity <= sparsity
        reservoir_matrix *= WeightInitializers.create_qmatrix(dims[1],
            rand(1:dims[1]),
            rand(1:dims[1]),
            rand(T) * T(2) - T(1),
            T)
        tmp_sparsity = WeightInitializers.get_sparsity(reservoir_matrix, dims[1])
    end

    return reservoir_matrix
end

for initializer in (:sparse_init, :identity_init, :delay_line, :delay_line_backward,
        :rand_sparse, :cycle_jumps, :simple_cycle, :pseudo_svd)
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
