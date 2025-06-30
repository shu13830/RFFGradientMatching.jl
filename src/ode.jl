#   - The Lotka-Volterra Predator Prey model. (Dondelinger 13, Barber 14, Wenk 19)
#   - The Lotka-Volterra Competition model.
#   - The SIR model.
#   - The PIF4/5 model. (Dondelinger 13)
#   - The signal transduction cascade. (Dondelinger 13, Barber 14, Wenk 19)
#   - FHN system (Wenk 19)
#   - The Lorenz96 model.

function lotkavolterrapredatorprey!(du, u, p, t)
    x, y = u
    a, b, c, d = p
    du[1] = a * x - b * x * y
    du[2] = - c * y + d * x * y
end

function lotkavolterracompetition!(du, u, p, t)
    K = length(u)  # number of species
    r = p[1:K]  # growth rate
    Ks = p[K+1:2K]  # carrying capacity
    α = reshape(p[2K+1:end], K, K)  # interaction matrix
    for i in 1:K
        interaction = sum(α[i, j] * u[j] for j in 1:K)  # interaction with other species
        du[i] = r[i] * u[i] * (1 - interaction / Ks[i])  # change in population
    end
end

function pif4and5!(du, u, p, t)
    error("Not implemented")  # TODO not sufficiently described in Dondelinger 13
    # PIF45: Phytochrome Interacting Factor 4/5
    # TOC1: Timing of CAB expression 1
    PIF45, TOC1 = u
    # s: promoter strength
    # Kd: rate constant
    # h: Hill coefficient
    # d: degradation rate of PIF4/5 mRNA
    s, Kd, h, d = p
    du[1] = s * Kd^h / (Kd^h + TOC1^h) - d * PIF45  # d[PIF4/5]/dt
    du[2] = s * Kd^h / (Kd^h + PIF45^h) - d * TOC1  # d[TOC1]/dt  # TODO: check this
end

function signaltransductioncascade!(du, u, p, t)
    # S: substrate
    # R: receptor
    # Rs: receptor-substrate complex
    # Rpp: phosphorylated receptor
    S, dS, R, Rs, Rpp = u  
    θ1, θ2, θ3, θ4, θ5, θ6 = p

    du[1] = -θ1 * S - θ2 * S * R + θ3 * Rs  # dS/dt
    du[2] = θ1 * S                          # ddS/dt
    du[3] = -θ2 * S * R + θ3 * Rs + θ5 * Rpp / (θ6 + Rpp)  # dR/dt
    du[4] = θ2 * S * R - θ3 * Rs - θ4 * Rs  # dRs/dt
    du[5] = θ4 * Rs - θ5 * Rpp / (θ6 + Rpp) # dRpp/dt
end

function fitzhughnagumo!(du, u, p, t)
    V, R = u
    θ1, θ2, θ3 = p

    du[1] = θ1 * (V - (V^3) / 3 + R)  # dV/dt
    du[2] = (1 / θ1) * (V - θ2 + θ3 * R)  # dR/dt
end

function lorenz96!(du, u, p, t)
    F = p[1]
    N = length(u)
    for i in 1:N
        du[i] = (u[mod1(i + 1, N)] - u[mod1(i - 2, N)]) * u[mod1(i - 1, N)] - u[i] + F
    end
end

function sir!(du, u, p, t)
    S, I, R = u
    a, b = p
    du[1] = -a * S * I
    du[2] = a * S * I - b * I
    du[3] = b * I
end

function log_lotkavolterrapredatorprey!(du, u, p, t)
    x, y = exp.(u)
    a, b, c, d = p
    du[1] = (a * x - b * x * y) / x
    du[2] = (- c * y + d * x * y) / y
end

function log_lotkavolterracompetition!(du, u, p, t)
    K = length(u)  # number of species
    r = p[1:K]  # growth rate
    Ks = p[K+1:2K]  # carrying capacity
    α = reshape(p[2K+1:end], K, K)  # interaction matrix
    expu = exp.(u)
    for i in 1:K
        interaction = sum(α[i, j] * expu[j] for j in 1:K)  # interaction with other species
        du[i] = r[i] * (1 - interaction / Ks[i])  # change in population
    end
end

function log_sir!(du, u, p, t)
    S, I, R = u
    a, b = p
    du[1] = -a * exp(I)
    du[2] = a * exp(S) - b
    du[3] = b * exp(I) / exp(R)
end
