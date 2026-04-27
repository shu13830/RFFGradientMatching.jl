# Hand-coded Jacobians for known ODE systems.
# Each ODE has allocating and in-place (!) versions.
# Returns (dẋdx!, dẋdθ!, K, P) or (nothing, nothing, 0, 0) if not available.
# K = state dimension, P = parameter dimension

"""
    get_manual_jacobians(probname::String)

Return hand-coded in-place (dẋdx!, dẋdθ!) for known ODE systems.
Signature: dẋdx!(J, x, θ) fills J in-place (K×K).
           dẋdθ!(J, x, θ) fills J in-place (K×P).
Returns (nothing, nothing) if not available.
"""
function get_manual_jacobians(probname::String)
    if probname == "LV"
        return _jac_lv_dx!, _jac_lv_dθ!
    elseif probname == "FN"
        return _jac_fn_dx!, _jac_fn_dθ!
    elseif probname == "PST"
        return _jac_pst_dx!, _jac_pst_dθ!
    elseif probname == "SIR"
        return _jac_sir_dx!, _jac_sir_dθ!
    else
        return nothing, nothing
    end
end

# ── Lotka-Volterra Predator-Prey ──
# du[1] = a*x - b*x*y
# du[2] = -c*y + d*x*y
function _jac_lv_dx!(J, u, p)
    x, y = u
    a, b, c, d = p
    @inbounds begin
        J[1,1] = a - b*y;   J[1,2] = -b*x
        J[2,1] = d*y;       J[2,2] = -c + d*x
    end
    return J
end

function _jac_lv_dθ!(J, u, p)
    x, y = u
    @inbounds begin
        J[1,1] = x;    J[1,2] = -x*y;  J[1,3] = 0.0;   J[1,4] = 0.0
        J[2,1] = 0.0;  J[2,2] = 0.0;   J[2,3] = -y;    J[2,4] = x*y
    end
    return J
end

# ── FitzHugh-Nagumo ──
# du[1] = θ1*(V - V³/3 + R)
# du[2] = (1/θ1)*(V - θ2 + θ3*R)
function _jac_fn_dx!(J, u, p)
    V, R = u
    θ1, θ2, θ3 = p
    @inbounds begin
        J[1,1] = θ1*(1 - V^2);  J[1,2] = θ1
        J[2,1] = 1/θ1;          J[2,2] = θ3/θ1
    end
    return J
end

function _jac_fn_dθ!(J, u, p)
    V, R = u
    θ1, θ2, θ3 = p
    @inbounds begin
        J[1,1] = V - V^3/3 + R;                J[1,2] = 0.0;    J[1,3] = 0.0
        J[2,1] = -(1/θ1^2)*(V - θ2 + θ3*R);   J[2,2] = -1/θ1;  J[2,3] = R/θ1
    end
    return J
end

# ── Protein Signaling Transduction Cascade ──
function _jac_pst_dx!(J, u, p)
    S, dS, R, Rs, Rpp = u
    θ1, θ2, θ3, θ4, θ5, θ6 = p
    denom = (θ6 + Rpp)^2
    fill!(J, 0.0)
    @inbounds begin
        J[1,1] = -θ1 - θ2*R;  J[1,3] = -θ2*S;  J[1,4] = θ3
        J[2,1] = θ1
        J[3,1] = -θ2*R;  J[3,3] = -θ2*S;  J[3,4] = θ3;  J[3,5] = θ5*θ6/denom
        J[4,1] = θ2*R;   J[4,3] = θ2*S;    J[4,4] = -θ3 - θ4
        J[5,4] = θ4;     J[5,5] = -θ5*θ6/denom
    end
    return J
end

function _jac_pst_dθ!(J, u, p)
    S, dS, R, Rs, Rpp = u
    θ1, θ2, θ3, θ4, θ5, θ6 = p
    denom = θ6 + Rpp
    fill!(J, 0.0)
    @inbounds begin
        J[1,1] = -S;   J[1,2] = -S*R;  J[1,3] = Rs
        J[2,1] = S
        J[3,2] = -S*R; J[3,3] = Rs;    J[3,5] = Rpp/denom;  J[3,6] = -θ5*Rpp/denom^2
        J[4,2] = S*R;  J[4,3] = -Rs;   J[4,4] = -Rs
        J[5,4] = Rs;   J[5,5] = -Rpp/denom;  J[5,6] = θ5*Rpp/denom^2
    end
    return J
end

# ── SIR ──
function _jac_sir_dx!(J, u, p)
    S, I, R = u
    a, b = p
    @inbounds begin
        J[1,1] = -a*I;  J[1,2] = -a*S;  J[1,3] = 0.0
        J[2,1] = a*I;   J[2,2] = a*S-b; J[2,3] = 0.0
        J[3,1] = 0.0;   J[3,2] = b;     J[3,3] = 0.0
    end
    return J
end

function _jac_sir_dθ!(J, u, p)
    S, I, R = u
    a, b = p
    @inbounds begin
        J[1,1] = -S*I;  J[1,2] = 0.0
        J[2,1] = S*I;   J[2,2] = -I
        J[3,1] = 0.0;   J[3,2] = I
    end
    return J
end
