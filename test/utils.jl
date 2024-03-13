using WeightInitializers

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