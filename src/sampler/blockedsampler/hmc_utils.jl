"""
Adjustment of step size ε:
ε is reduced when rejected for 3 consecutive times and increased when accepted for 10 consecutive times.
This process is only carried out during burn-in to avoid disrupting the Markov chain.
"""
function adjust_ϵ_heuristically!(burnin::Bool, blk::HMCBlock)
    if blk.type != :HMC
        return
    end

    trajectory = blk.sampler.κ.τ
    ϵ = trajectory.integrator.ϵ
    L = trajectory.termination_criterion.L

    if burnin
        if blk.accept_counter > 10
            ϵ *= 1.1
        end
        if blk.reject_counter > 3
            ϵ /= 1.1
        end
        if blk.accept_counter > 10
            ϵ *= 1.1
        end
        if blk.reject_counter > 3
            ϵ /= 1.1
        end
    else # NOTE: after burnin, keep L * ϵ constant not to break the Markov chain
        if blk.accept_counter > 6
            if L > 5  # NOTE: assume minimum L is 5
                # subtract 1 from L
                L -= 1
                # make ϵ larger to keep L * ϵ constant
                ϵ *= (L+1) / L
            end
        end
        if blk.reject_counter > 4
            if L < 30  # NOTE: assume maximum L is 30
                # add 1 to L
                L += 1
                # make ϵ smaller to keep L * ϵ constant
                ϵ *= (L-1) / L
            end
        end
    end

    # update sampler
    new_kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{EndPointTS}(Leapfrog(ϵ), AdvancedHMC.FixedNSteps(L)))
    new_sampler = HMCSampler(new_kernel, blk.sampler.metric, blk.sampler.adaptor)
    blk.sampler = new_sampler
end
