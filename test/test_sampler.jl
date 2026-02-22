@testset "T6: Sampler" begin
    @testset "BlockedSampler construction" begin
        gm = create_test_rffgm()
        gm.β[1] = 1.0
        block_W = HMCBlock(gm, [:W]; n_leapfrog=5, step_size=0.01)
        block_θ = HMCBlock(gm, [:θ]; n_leapfrog=5, step_size=0.01)
        bs = BlockedSampler([[block_W], [block_θ]], [0.5, 0.5])
        @test length(bs.blocks) == 2
        @test Set(bs.global_sample_target) == Set([:W, :θ])
    end

    @testset "pack/unpack roundtrip" begin
        gm = create_test_rffgm()
        sample_target = [:W, :θ]
        dict_orig = pack_param_dict(gm)
        vec = pack_param_vec(gm, sample_target)
        dict_reconstructed = pack_param_dict_from_vec(gm, vec, sample_target)
        @test dict_orig[:W] ≈ dict_reconstructed[:W]
        @test dict_orig[:θ] ≈ dict_reconstructed[:θ]
    end

    @testset "pack/unpack roundtrip GPGM" begin
        gm = create_test_gpgm()
        sample_target = [:X, :θ]
        dict_orig = pack_param_dict(gm)
        vec = pack_param_vec(gm, sample_target)
        dict_reconstructed = pack_param_dict_from_vec(gm, vec, sample_target)
        @test dict_orig[:X] ≈ dict_reconstructed[:X]
        @test dict_orig[:θ] ≈ dict_reconstructed[:θ]
    end

    @testset "update_model_with_vec! roundtrip" begin
        gm = create_test_rffgm()
        W_before = copy(get_W(gm))
        θ_before = copy(get_θ(gm))
        sample_target = [:W, :θ]
        vec = pack_param_vec(gm, sample_target)
        # Perturb and update
        vec_perturbed = vec .+ 0.01 .* randn(length(vec))
        update_model_with_vec!(gm, sample_target, vec_perturbed)
        # Values should have changed
        @test !(get_W(gm) ≈ W_before)
    end
end
