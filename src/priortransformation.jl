struct PriorTransformation
    prior::Distribution                        # prior for transformed variable t(v)
    t::Union{Function, Bijectors.Transform}    # transform from v to t(v)
    t⁻¹::Union{Function, Bijectors.Transform}  # inverse transform from t(v) to v
    dtv_dv::Function                           # dt(v)/dv. NOTE: input is not t(v) but v.

    function PriorTransformation(prior::Distribution, t::Union{Function, Bijectors.Transform})
        dtv_dv = v -> ForwardDiff.derivative(v -> t(v), v)  # dt(v)/dv
        # Check if `t` has an inverse; otherwise, raise an error or handle gracefully
        t_inv = Bijectors.inverse(t)
        if t_inv isa Union{Function, Bijectors.Transform}
            return new(prior, t, Bijectors.inverse(t), dtv_dv)
        else
            error("The transformation provided does not have a valid inverse.")
        end
    end
end

calc_tvar(pt::PriorTransformation, v::T) where {T<:Real} = pt.t(v)
calc_tvar(pt::Vector{PriorTransformation}, v::Vector{<:Real}) =
    [calc_tvar(pt_i, v_i) for (pt_i, v_i) in zip(pt, v)]

calc_var(pt::PriorTransformation, tv::T) where {T<:Real} = pt.t⁻¹(tv)
calc_var(pt::Vector{PriorTransformation}, tv::Vector{<:Real}) =
    [calc_var(pt_i, tv_i) for (pt_i, tv_i) in zip(pt, tv)]

rand_tvar(pt::PriorTransformation) = rand(pt.prior)
rand_var(pt::PriorTransformation) = calc_var(pt, rand_tvar(pt))
