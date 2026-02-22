@testset "T5: Gradient correctness" begin
    @testset "RFFGM ∇w_logpdf_x" begin
        gm = create_test_rffgm()
        @test validate_∇w_logpdf_x(gm)
    end

    @testset "RFFGM ∇w_logpdf_y" begin
        gm = create_test_rffgm()
        @test validate_∇w_logpdf_y(gm)
    end

    @testset "RFFGM ∇w_ulogpdf_e" begin
        gm = create_test_rffgm()
        gm.β[1] = 1.0
        @test validate_∇w_ulogpdf_e(gm)
    end

    @testset "RFFGM ∇tθ_logpdf_θ" begin
        gm = create_test_rffgm()
        @test validate_∇tθ_logpdf_θ(gm)
    end

    @testset "RFFGM ∇tθ_ulogpdf_e" begin
        gm = create_test_rffgm()
        gm.β[1] = 1.0
        @test validate_∇tθ_ulogpdf_e(gm)
    end

    @testset "RFFGM ∇tγ_logpdf_γ" begin
        gm = create_test_rffgm()
        @test validate_∇tγ_logpdf_γ(gm)
    end

    @testset "RFFGM ∇tγ_ulogpdf_e" begin
        gm = create_test_rffgm()
        gm.β[1] = 1.0
        @test validate_∇tγ_ulogpdf_e(gm)
    end

    @testset "RFFGM ∇tσ_logpdf_σ" begin
        gm = create_test_rffgm()
        @test validate_∇tσ_logpdf_σ(gm)
    end

    @testset "RFFGM ∇tσ_logpdf_y" begin
        gm = create_test_rffgm()
        @test validate_∇tσ_logpdf_y(gm)
    end

    @testset "RFFGM ∇y_logpdf_y" begin
        gm = create_test_rffgm()
        @test validate_∇y_logpdf_y(gm)
    end

    @testset "RFFGM ∇ulogpdf [:W]" begin
        gm = create_test_rffgm()
        gm.β[1] = 1.0
        @test validate_∇ulogpdf(gm, [:W])
    end

    @testset "RFFGM ∇ulogpdf [:θ]" begin
        gm = create_test_rffgm()
        gm.β[1] = 1.0
        @test validate_∇ulogpdf(gm, [:θ])
    end

    @testset "RFFGM ∇ulogpdf [:W, :θ]" begin
        gm = create_test_rffgm()
        gm.β[1] = 1.0
        @test validate_∇ulogpdf(gm, [:W, :θ])
    end

    @testset "GPGM ∇tx_logpdf_x" begin
        gm = create_test_gpgm()
        @test validate_∇tx_logpdf_x(gm)
    end

    @testset "GPGM ∇tx_logpdf_y" begin
        gm = create_test_gpgm()
        @test validate_∇tx_logpdf_y(gm)
    end

    @testset "GPGM ∇tx_ulogpdf_e" begin
        gm = create_test_gpgm()
        gm.β[1] = 1.0
        @test validate_∇tx_ulogpdf_e(gm)
    end

    @testset "GPGM ∇tθ_logpdf_θ" begin
        gm = create_test_gpgm()
        @test validate_∇tθ_logpdf_θ(gm)
    end

    @testset "GPGM ∇tθ_ulogpdf_e" begin
        gm = create_test_gpgm()
        gm.β[1] = 1.0
        @test validate_∇tθ_ulogpdf_e(gm)
    end

    @testset "GPGM ∇tγ_logpdf_γ" begin
        gm = create_test_gpgm()
        @test validate_∇tγ_logpdf_γ(gm)
    end

    @testset "GPGM ∇tγ_ulogpdf_e" begin
        gm = create_test_gpgm()
        gm.β[1] = 1.0
        @test validate_∇tγ_ulogpdf_e(gm)
    end

    @testset "GPGM ∇tσ_logpdf_σ" begin
        gm = create_test_gpgm()
        @test validate_∇tσ_logpdf_σ(gm)
    end

    @testset "GPGM ∇tσ_logpdf_y" begin
        gm = create_test_gpgm()
        @test validate_∇tσ_logpdf_y(gm)
    end

    @testset "GPGM ∇y_logpdf_y" begin
        gm = create_test_gpgm()
        @test validate_∇y_logpdf_y(gm)
    end

    @testset "GPGM ∇ulogpdf [:X, :θ]" begin
        gm = create_test_gpgm()
        gm.β[1] = 1.0
        @test validate_∇ulogpdf(gm, [:X, :θ])
    end
end
