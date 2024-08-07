@inline _nfan() = 1, 1 # fan_in, fan_out
@inline _nfan(n) = 1, n # A vector is treated as a n×1 matrix
@inline _nfan(n_out, n_in) = n_in, n_out # In case of Dense kernels: arranged as matrices
@inline _nfan(dims::Tuple) = _nfan(dims...)
@inline _nfan(dims...) = prod(dims[1:(end - 2)]) .* (dims[end - 1], dims[end]) # In case of convolution kernels
@inline _norm_cdf(x::T) where {T} = T(0.5) * (1 + T(erf(x / √2))) # erf often doesn't respect the type

@inline _default_rng() = Xoshiro(1234)

const NAME_TO_DIST = Dict(
    :zeros => "an AbstractArray of zeros", :ones => "an AbstractArray of ones",
    :randn => "random numbers from a standard normal distribution",
    :rand => "random numbers from a uniform distribution")
const NUM_TO_FPOINT = Dict(
    Symbol(16) => Float16, Symbol(32) => Float32, Symbol(64) => Float64,
    :C16 => ComplexF16, :C32 => ComplexF32, :C64 => ComplexF64)

@inline function __funcname(fname::String)
    fp = fname[(end - 2):end]
    Symbol(fp) in keys(NUM_TO_FPOINT) && return fname[1:(end - 3)], fp
    return fname[1:(end - 2)], fname[(end - 1):end]
end

@inline function __generic_docstring(fname::String)
    funcname, fp = __funcname(fname)
    name = NAME_TO_DIST[Symbol(funcname)]
    dist_type = NUM_TO_FPOINT[Symbol(fp)]
    return """
        $fname([::AbstractRNG=_default_rng()], size...;
            kwargs...) -> AbstractArray{$(dist_type), length(size)}

    Return an `AbstractArray{$(dist_type)}` of the given `size` containing $(name).
    """
end

# Helpers for device agnostic initializers
@inline function __zeros(::AbstractRNG, ::Type{T}, dims::Integer...) where {T <: Number}
    return zeros(T, dims...)
end
@inline function __ones(::AbstractRNG, ::Type{T}, dims::Integer...) where {T <: Number}
    return ones(T, dims...)
end
@inline function __rand(rng::AbstractRNG, ::Type{T}, args::Integer...) where {T <: Number}
    return rand(rng, T, args...)
end
@inline function __randn(rng::AbstractRNG, ::Type{T}, args::Integer...) where {T <: Number}
    return randn(rng, T, args...)
end

## Certain backends don't support sampling Complex numbers, so we avoid hitting those
## dispatches
for f in (:__rand, :__randn)
    @eval @inline function $(f)(
            rng::AbstractRNG, ::Type{<:Complex{T}}, args::Integer...) where {T <: Number}
        real_part = $(f)(rng, T, args...)
        imag_part = $(f)(rng, T, args...)
        return Complex{T}.(real_part, imag_part)
    end
end
