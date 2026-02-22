@testset "T1: PriorTransformation" begin
    @testset "log transform roundtrip" begin
        pt = PriorTransformation(Normal(0, 1), log)
        for v in [0.1, 1.0, 5.0, 100.0]
            tv = calc_tvar(pt, v)
            @test calc_var(pt, tv) ≈ v
        end
    end

    @testset "jacobian correctness (dtv_dv)" begin
        pt = PriorTransformation(Normal(0, 1), log)
        for v in [0.5, 1.0, 2.0]
            fd_jac = ForwardDiff.derivative(log, v)
            @test pt.dtv_dv(v) ≈ fd_jac atol=1e-10
        end
    end

    @testset "identity transform roundtrip" begin
        pt = PriorTransformation(Normal(0, 1), identity)
        for v in [-1.0, 0.0, 1.0, 5.0]
            tv = calc_tvar(pt, v)
            @test calc_var(pt, tv) ≈ v
        end
    end

    @testset "vector calc_tvar / calc_var" begin
        pts = [PriorTransformation(Normal(0, 1), log) for _ in 1:3]
        vs = [0.5, 1.0, 2.0]
        tvs = calc_tvar(pts, vs)
        recovered = calc_var(pts, tvs)
        @test all(recovered .≈ vs)
    end

    @testset "rand_tvar and rand_var" begin
        pt = PriorTransformation(Normal(0, 1), log)
        tv = rand_tvar(pt)
        @test isa(tv, Float64)
        v = rand_var(pt)
        @test v > 0  # exp of any real number is positive
    end
end
