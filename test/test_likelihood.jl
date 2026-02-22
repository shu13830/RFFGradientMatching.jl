@testset "T4: Likelihood" begin
    @testset "RFFGM ulogpdf merge vs decompose" begin
        gm = create_test_rffgm()
        gm.β[1] = 1.0
        pd = pack_param_dict(gm)
        total = ulogpdf(gm, pd; merge_output=true)
        parts = ulogpdf(gm, pd; merge_output=false)
        @test total ≈ sum(values(parts))
        @test isfinite(total)
    end

    @testset "GPGM ulogpdf merge vs decompose" begin
        gm = create_test_gpgm()
        gm.β[1] = 1.0
        pd = pack_param_dict(gm)
        total = ulogpdf(gm, pd; merge_output=true)
        parts = ulogpdf(gm, pd; merge_output=false)
        @test total ≈ sum(values(parts))
        @test isfinite(total)
    end

    @testset "RFFGM ulogpdf returns finite" begin
        gm = create_test_rffgm()
        @test isfinite(ulogpdf(gm))
    end

    @testset "GPGM ulogpdf returns finite" begin
        gm = create_test_gpgm()
        @test isfinite(ulogpdf(gm))
    end

    @testset "RFFGM ulogpdf via param_vec" begin
        gm = create_test_rffgm()
        gm.β[1] = 1.0
        sample_target = [:W, :θ]
        vec = pack_param_vec(gm, sample_target)
        lp = ulogpdf(vec, gm, sample_target)
        @test isfinite(lp)
    end

    @testset "GPGM ulogpdf via param_vec" begin
        gm = create_test_gpgm()
        gm.β[1] = 1.0
        sample_target = [:X, :θ]
        vec = pack_param_vec(gm, sample_target)
        lp = ulogpdf(vec, gm, sample_target)
        @test isfinite(lp)
    end
end
