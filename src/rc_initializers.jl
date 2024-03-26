### input layers
"""
    scaled_rand([::AbstractRNG=_default_rng()], [T=Float32], dims::Integer...;
    scaling=T(0.1))

Create and return a matrix with random values, uniformly distributed within a
range defined by `scaling`. This function is useful for initializing matrices,
such as the layers of a neural network, with scaled random values.

# Arguments

  - `rng`: An instance of `AbstractRNG` for random number generation.
  - `T`: The data type for the elements of the matrix.
  - `dims`: Dimensions of the matrix. It must be a 2-element tuple specifying the
  number of rows and columns (e.g., `(res_size, in_size)`).
  - `scaling`: A scaling factor to define the range of the uniform distribution.
  The matrix elements will be randomly chosen from the range `[-scaling, scaling]`.
  Defaults to `0.1`.

# Returns

A matrix of type with dimensions specified by `dims`.
Each element of the matrix is a random number uniformly distributed
between `-scaling` and `scaling`.

# Example

```julia
rng = Random.default_rng()
matrix = scaled_rand(rng, Float64, (100, 50); scaling = 0.2)
```
"""
function scaled_rand(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        scaling = T(0.1)) where {T <: Number}
    res_size, in_size = dims
    layer_matrix = T(2)*T(scaling)*rand(rng, T, res_size, in_size) .- T(scaling)
    return layer_matrix
end

"""
    weighted_init([::AbstractRNG=_default_rng()], [T=Float32], dims::Integer...;
    scaling=T(0.1))

Create and return a matrix representing a weighted input layer for Echo State
Networks (ESNs). This initializer generates a weighted input matrix with random
non-zero elements distributed uniformly within the range [-`scaling`, `scaling`],
inspired by the approach in [^Lu].

# Arguments

  - `rng`: An instance of `AbstractRNG` for random number generation.
  - `T`: The data type for the elements of the matrix.
  - `dims`: A 2-element tuple specifying the approximate reservoir size and input
  size (e.g., `(approx_res_size, in_size)`).
  - `scaling`: The scaling factor for the weight distribution. Defaults to `0.1`.

# Returns

A matrix representing the weighted input layer as defined in [^Lu2017]. The matrix
dimensions will be adjusted to ensure each input unit connects to an equal number of
reservoir units.

# Example

```julia
rng = Random.default_rng()
input_layer = weighted_init(rng, Float64, (3, 300); scaling = 0.2)
```

# References

[^Lu2017]: Lu, Zhixin, et al.
    "Reservoir observers: Model-free inference of unmeasured variables in chaotic systems."
    Chaos: An Interdisciplinary Journal of Nonlinear Science 27.4 (2017): 041102.
"""
function weighted_init(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        scaling = T(0.1)) where {T <: Number}
    approx_res_size, in_size = dims
    res_size = Int(floor(approx_res_size / in_size) * in_size)
    layer_matrix = zeros(T, res_size, in_size)
    q = floor(Int, res_size / in_size)

    for i in 1:in_size
        layer_matrix[((i - 1) * q + 1):((i) * q), i] = T(2)*T(scaling)*rand(rng, T, q).-
            T(scaling)
    end

    return layer_matrix
end

"""
    informed_init([::AbstractRNG=_default_rng()], [T=Float32], dims::Integer...;
    scaling=T(0.1), model_in_size, gamma=T(0.5))

Create a layer of a neural network.

# Arguments

  - `rng::AbstractRNG`: The random number generator.
  - `T::Type`: The data type.
  - `dims::Integer...`: The dimensions of the layer.
  - `scaling::T = T(0.1)`: The scaling factor for the input matrix.
  - `model_in_size`: The size of the input model.
  - `gamma::T = T(0.5)`: The gamma value.

# Returns

  - `input_matrix`: The created input matrix for the layer.

# Example

```julia
rng = Random.default_rng()
dims = (100, 200)
model_in_size = 50
input_matrix = informed_init(rng, Float64, dims; model_in_size = model_in_size)
```
"""
function informed_init(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        scaling = T(0.1), model_in_size, gamma = T(0.5)) where {T <: Number}
    res_size, in_size = dims
    state_size = in_size - model_in_size

    if state_size <= 0
        throw(DimensionMismatch("in_size must be greater than model_in_size"))
    end

    input_matrix = zeros(T, res_size, in_size)
    zero_connections = zeros(in_size)
    num_for_state = floor(Int, res_size * T(gamma))
    num_for_model = floor(Int, res_size * (T(1) - T(gamma)))

    for i in 1:num_for_state
        idxs = findall(Bool[zero_connections .== input_matrix[i, :]
                            for i in 1:size(input_matrix, 1)])
        random_row_idx = idxs[rand(rng, 1:end)]
        random_clm_idx = range(1, state_size, step = 1)[rand(rng, 1:end)]
        input_matrix[random_row_idx, random_clm_idx] = T(2)*T(scaling)*rand(rng, T) .- T(scaling)
    end

    for i in 1:num_for_model
        idxs = findall(Bool[zero_connections .== input_matrix[i, :]
                            for i in 1:size(input_matrix, 1)])
        random_row_idx = idxs[rand(rng, 1:end)]
        random_clm_idx = range(state_size + 1, in_size, step = 1)[rand(rng, 1:end)]
        input_matrix[random_row_idx, random_clm_idx] = T(2)*T(scaling)*rand(rng, T) .- T(scaling)
    end

    return input_matrix
end

"""
    minimal_init([::AbstractRNG=_default_rng()], [T=Float32], dims::Integer...;
    weight = 0.1, sampling_type = :bernoulli)

Create a layer matrix using the provided random number generator and sampling parameters.

# Arguments

  - `rng::AbstractRNG`: The random number generator used to generate random numbers.
  - `dims::Integer...`: The dimensions of the layer matrix.
  - `weight`: The weight used to fill the layer matrix. Default is 0.1.
  - `sampling`: The sampling parameters used to generate the input matrix. Default is
  :bernoulli.

# Returns

The layer matrix generated using the provided random number generator and sampling
parameters.

# Example

```julia
using Random
rng = Random.default_rng()
dims = (3, 2)
weight = 0.5
layer_matrix = irrational_sample_init(rng, Float64, dims; weight = weight,
    sampling = IrrationalSample(irrational = sqrt(2), start = 1))
```
"""
function minimal_init(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        sampling_type::Symbol = :bernoulli,
        weight::Number = T(0.1),
        irrational::Real = pi,
        start::Int = 1,
        p::Number = T(0.5)) where {T <: Number}
    res_size, in_size = dims
    if sampling_type == :bernoulli
        layer_matrix = _create_bernoulli(p, res_size, in_size, weight, rng, T)
    elseif sampling_type == :irrational
        layer_matrix = _create_irrational(irrational,
            start,
            res_size,
            in_size,
            weight,
            rng,
            T)
    else
        error("Sampling type not allowed. Please use one of :bernoulli or :irrational")
    end
    return layer_matrix
end

function _generate_bernoulli(rng::AbstractRNG, p::Number)
    rand(rng) < p ? true : false
end

function _create_bernoulli(p::Number,
        res_size::Int,
        in_size::Int,
        weight::Number,
        rng::AbstractRNG,
        ::Type{T}) where {T <: Number}
    input_matrix = zeros(T, res_size, in_size)
    for i in 1:res_size
        for j in 1:in_size
            _generate_bernoulli(rng, p) ? (input_matrix[i, j] = weight) :
            (input_matrix[i, j] = -weight)
        end
    end
    return input_matrix
end

function _create_irrational(irrational::Irrational,
        start::Int,
        res_size::Int,
        in_size::Int,
        weight::T,
        rng::AbstractRNG,
        ::Type{T}) where {T <: Number}
    setprecision(BigFloat, Int(ceil(log2(10) * (res_size * in_size + start + 1))))
    ir_string = string(BigFloat(irrational)) |> collect
    deleteat!(ir_string, findall(x -> x == '.', ir_string))
    ir_array = zeros(length(ir_string))
    input_matrix = zeros(T, res_size, in_size)

    for i in 1:length(ir_string)
        ir_array[i] = parse(Int, ir_string[i])
    end

    for i in 1:res_size
        for j in 1:in_size
            random_number = rand(rng, T)
            input_matrix[i, j] = random_number < 0.5 ? -weight : weight
        end
    end

    return input_matrix
end

### reservoirs

"""
    rand_sparse([::AbstractRNG=_default_rng()], [T=Float32], dims::Integer...;
    radius=1.0, sparsity=0.1)

Create and return a random sparse reservoir matrix for use in Echo State Networks (ESNs).
The matrix will be of size specified by `dims`, with specified `sparsity` and scaled
spectral radius according to `radius`.

# Arguments

  - `rng`: An instance of `AbstractRNG` for random number generation.
  - `T`: The data type for the elements of the matrix.
  - `dims`: Dimensions of the reservoir matrix.
  - `radius`: The desired spectral radius of the reservoir. Defaults to 1.0.
  - `sparsity`: The sparsity level of the reservoir matrix, controlling the fraction
  of zero elements. Defaults to 0.1.

# Returns

A matrix representing the random sparse reservoir.

# References

This type of reservoir initialization is commonly used in ESNs for capturing temporal
dependencies in data.
"""
function rand_sparse(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        radius = T(1.0),
        sparsity = T(0.1),
        std = T(1.0)) where {T <: Number}
    lcl_sparsity = T(1) - T(sparsity) #consistency with current implementations
    reservoir_matrix = sparse_init(rng, T, dims...;
        sparsity = lcl_sparsity, std = T(std))
    rho_w = maximum(abs.(eigvals(reservoir_matrix)))
    reservoir_matrix .*= T(radius) / rho_w
    if Inf in unique(reservoir_matrix) || -Inf in unique(reservoir_matrix)
        error("Sparsity too low for size of the matrix.
            Increase res_size or increase sparsity")
    end
    return reservoir_matrix
end

"""
    delay_line([::AbstractRNG=_default_rng()], [T=Float32], dims::Integer...;
    weight=0.1)

Create and return a delay line reservoir matrix for use in Echo State Networks (ESNs).
A delay line reservoir is a deterministic structure where each unit is connected only
to its immediate predecessor with a specified weight. This method is particularly useful
for tasks that require specific temporal processing.

# Arguments

  - `rng`: An instance of `AbstractRNG` for random number generation. This argument is
  not used in the current implementation but is included for consistency with other
  initialization functions.
  - `T`: The data type for the elements of the matrix.
  - `dims`: Dimensions of the reservoir matrix. Typically, this should be a tuple of
  two equal integers representing a square matrix.
  - `weight`: The weight determines the absolute value of all connections in the reservoir.
  Defaults to 0.1.

# Returns

A delay line reservoir matrix with dimensions specified by `dims`.
The matrix is initialized such that each element in the `i+1`th row and `i`th
column is set to `weight`, and all other elements are zeros.

# Example

```julia
reservoir = delay_line(Float64, 100, 100; weight = 0.2)
```

# References

This type of reservoir initialization is described in:
Rodan, Ali, and Peter Tino. "Minimum complexity echo state network."
IEEE Transactions on Neural Networks 22.1 (2010): 131-144.
"""
function delay_line(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        weight = T(0.1)) where {T <: Number}
    reservoir_matrix = zeros(T, dims...)
    @assert length(dims) == 2&&dims[1] == dims[2] "The dimensions must define a square matrix (e.g., (100, 100))"

    for i in 1:(dims[1] - 1)
        reservoir_matrix[i + 1, i] = T(weight)
    end

    return reservoir_matrix
end

"""
    delay_line_backward([::AbstractRNG=_default_rng()], [T=Float32], dims::Integer...;
    weight = T(0.1), fb_weight = T(0.2)) where {T <: Number}

Create a delay line backward reservoir with the specified by `dims` and weights.
Creates a matrix with backward connections as described in [^Rodan2010].
The `weight` and `fb_weight` can be passed as either arguments or keyword arguments,
and they determine the absolute values of the connections in the reservoir.

# Arguments

  - `rng::AbstractRNG`: Random number generator.
  - `T::Type`: Type of the elements in the reservoir matrix.
  - `dims::Integer...`: Dimensions of the reservoir matrix.
  - `weight::T`: The weight determines the absolute value of forward connections
  in the reservoir, and is set to 0.1 by default.
  - `fb_weight::T`: The `fb_weight` determines the absolute value of backward
  connections in the reservoir, and is set to 0.2 by default.

# Returns

Reservoir matrix with the dimensions specified by `dims` and weights.

# References

[^Rodan2010]: Rodan, Ali, and Peter Tino. "Minimum complexity echo state network."
    IEEE transactions on neural networks 22.1 (2010): 131-144.
"""
function delay_line_backward(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        weight = T(0.1),
        fb_weight = T(0.2)) where {T <: Number}
    res_size = first(dims)
    reservoir_matrix = zeros(T, dims...)

    for i in 1:(res_size - 1)
        reservoir_matrix[i + 1, i] = T(weight)
        reservoir_matrix[i, i + 1] = T(fb_weight)
    end

    return reservoir_matrix
end

"""
    cycle_jumps([::AbstractRNG=_default_rng()], [T=Float32], dims::Integer...;
    cycle_weight = T(0.1), jump_weight = T(0.1), jump_size = 3)

Create a cycle jumps reservoir with the specified dimensions, cycle weight,
jump weight, and jump size.

# Arguments

  - `rng::AbstractRNG`: Random number generator.
  - `T::Type`: Type of the elements in the reservoir matrix.
  - `dims::Integer...`: Dimensions of the reservoir matrix.
  - `cycle_weight::T = T(0.1)`:  The weight of cycle connections.
  - `jump_weight::T = T(0.1)`: The weight of jump connections.
  - `jump_size::Int = 3`:  The number of steps between jump connections.

# Returns

Reservoir matrix with the specified dimensions, cycle weight, jump weight, and jump size.

# References

[^Rodan2012]: Rodan, Ali, and Peter Tiňo. "Simple deterministically constructed cycle
reservoirs with regular jumps." Neural computation 24.7 (2012): 1822-1852.
"""
function cycle_jumps(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        cycle_weight::Number = T(0.1),
        jump_weight::Number = T(0.1),
        jump_size::Int = 3) where {T <: Number}
    res_size = first(dims)
    reservoir_matrix = zeros(T, dims...)

    for i in 1:(res_size - 1)
        reservoir_matrix[i + 1, i] = T(cycle_weight)
    end

    reservoir_matrix[1, res_size] = T(cycle_weight)

    for i in 1:jump_size:(res_size - jump_size)
        tmp = (i + jump_size) % res_size
        if tmp == 0
            tmp = res_size
        end
        reservoir_matrix[i, tmp] = T(jump_weight)
        reservoir_matrix[tmp, i] = T(jump_weight)
    end

    return reservoir_matrix
end

"""
    simple_cycle([::AbstractRNG=_default_rng()], [T=Float32], dims::Integer...;
        weight = T(0.1))

Create a simple cycle reservoir with the specified dimensions and weight.

# Arguments

  - `rng::AbstractRNG`: Random number generator.
  - `T::Type`: Type of the elements in the reservoir matrix.
  - `dims::Integer...`: Dimensions of the reservoir matrix.
  - `weight::T = T(0.1)`: Weight of the connections in the reservoir matrix.

# Returns

Reservoir matrix with the dimensions specified by `dims` and weights.

# References

[^Rodan2010]: Rodan, Ali, and Peter Tino. "Minimum complexity echo state network."
    IEEE transactions on neural networks 22.1 (2010): 131-144.
"""
function simple_cycle(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        weight = T(0.1)) where {T <: Number}
    reservoir_matrix = zeros(T, dims...)

    for i in 1:(dims[1] - 1)
        reservoir_matrix[i + 1, i] = T(weight)
    end

    reservoir_matrix[1, dims[1]] = T(weight)
    return reservoir_matrix
end

"""
    pseudo_svd([::AbstractRNG=_default_rng()], [T=Float32], dims::Integer...; 
        max_value, sparsity, sorted = true, reverse_sort = false)

Returns an initializer to build a sparse reservoir matrix with the given
`sparsity` by using a pseudo-SVD approach as described in [^yang].

# Arguments

  - `rng::AbstractRNG`: Random number generator.
  - `T::Type`: Type of the elements in the reservoir matrix.
  - `dims::Integer...`: Dimensions of the reservoir matrix.
  - `max_value`: The maximum absolute value of elements in the matrix.
  - `sparsity`: The desired sparsity level of the reservoir matrix.
  - `sorted`: A boolean indicating whether to sort the singular values before
  creating the diagonal matrix. By default, it is set to `true`.
  - `reverse_sort`: A boolean indicating whether to reverse the sorted singular
  values. By default, it is set to `false`.

# Returns

Reservoir matrix with the specified dimensions, max value, and sparsity.

# References

This reservoir initialization method, based on a pseudo-SVD approach, is
inspired by the work in [^yang], which focuses on designing polynomial echo
state networks for time series prediction.

[^yang]: Yang, Cuili, et al. "_Design of polynomial echo state networks
for time series prediction._" Neurocomputing 290 (2018): 148-160.
"""
function pseudo_svd(rng::AbstractRNG,
        ::Type{T},
        dims::Integer...;
        max_value::Number = T(1.0),
        sparsity::Number = 0.1,
        sorted::Bool = true,
        reverse_sort::Bool = false) where {T <: Number}
    reservoir_matrix = create_diag(dims[1],
        max_value,
        T;
        sorted = sorted,
        reverse_sort = reverse_sort)
    tmp_sparsity = get_sparsity(reservoir_matrix, dims[1])

    while tmp_sparsity <= sparsity
        reservoir_matrix *= create_qmatrix(dims[1],
            rand(1:dims[1]),
            rand(1:dims[1]),
            rand(T) * T(2) - T(1),
            T)
        tmp_sparsity = get_sparsity(reservoir_matrix, dims[1])
    end

    return reservoir_matrix
end

function create_diag(dim::Number, max_value::Number, ::Type{T};
        sorted::Bool = true, reverse_sort::Bool = false) where {T <: Number}
    diagonal_matrix = zeros(T, dim, dim)
    if sorted == true
        if reverse_sort == true
            diagonal_values = sort(rand(T, dim) .* max_value, rev = true)
            diagonal_values[1] = max_value
        else
            diagonal_values = sort(rand(T, dim) .* max_value)
            diagonal_values[end] = max_value
        end
    else
        diagonal_values = rand(T, dim) .* max_value
    end

    for i in 1:dim
        diagonal_matrix[i, i] = diagonal_values[i]
    end

    return diagonal_matrix
end

function create_qmatrix(dim::Number,
        coord_i::Number,
        coord_j::Number,
        theta::Number,
        ::Type{T}) where {T <: Number}
    qmatrix = zeros(T, dim, dim)

    for i in 1:dim
        qmatrix[i, i] = 1.0
    end

    qmatrix[coord_i, coord_i] = cos(theta)
    qmatrix[coord_j, coord_j] = cos(theta)
    qmatrix[coord_i, coord_j] = -sin(theta)
    qmatrix[coord_j, coord_i] = sin(theta)
    return qmatrix
end

function get_sparsity(M, dim)
    return size(M[M .!= 0], 1) / (dim * dim - size(M[M .!= 0], 1)) #nonzero/zero elements
end
