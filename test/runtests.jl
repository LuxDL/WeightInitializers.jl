using Test, SafeTestsets

@testset "Common Utilities" begin
    @safetestset "Quality Assurance" include("qa.jl")
    @safetestset "Utils" include("utils.jl")
end

@testset "Initializers" begin
    @safetestset "Standard initializers" include("initializers.jl")
    @safetestset "Reservoir Computing initializers" include("rc_initializers.jl")
end