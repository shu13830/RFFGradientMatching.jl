@testset "T8: GeneralizedRFF integration" begin
    @testset "GeneralizedCauchy kernel evaluation" begin
        using GeneralizedRFF
        k = GeneralizedCauchyKernel(1.5, 1.5)
        @test k(0.0, 0.0) > 0
        @test k(0.0, 1.0) >= 0
    end
end
