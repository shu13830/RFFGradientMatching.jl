# ODE model definitions for MAGI baseline.
# These match the Julia implementations in src/ode.jl exactly.
#
# MAGI model list fields:
#   fOde(theta, x, tvec)       : dx/dt  (n_times x n_components)
#   fOdeDx(theta, x, tvec)     : d(dx/dt)/dx  (n_times x n_components x n_components)
#   fOdeDtheta(theta, x, tvec) : d(dx/dt)/dtheta  (n_times x n_params x n_components)
#   thetaLowerBound, thetaUpperBound : parameter bounds

# ── Lotka-Volterra Predator-Prey ───────────────────────────────────────────
# Julia: du[1] = a*x - b*x*y,  du[2] = -c*y + d*x*y
# theta = (a, b, c, d), true = (2, 1, 4, 1)

lvmodelODE <- function(theta, x, tvec) {
  H <- x[, 1]  # prey
  P <- x[, 2]  # predator
  result <- array(0, c(nrow(x), ncol(x)))
  result[, 1] <- theta[1] * H - theta[2] * H * P
  result[, 2] <- -theta[3] * P + theta[4] * H * P
  result
}

lvmodelDx <- function(theta, x, tvec) {
  H <- x[, 1]
  P <- x[, 2]
  resultDx <- array(0, c(nrow(x), ncol(x), ncol(x)))
  # d(dH/dt)/dH, d(dH/dt)/dP
  resultDx[, 1, 1] <- theta[1] - theta[2] * P
  resultDx[, 2, 1] <- -theta[2] * H
  # d(dP/dt)/dH, d(dP/dt)/dP
  resultDx[, 1, 2] <- theta[4] * P
  resultDx[, 2, 2] <- -theta[3] + theta[4] * H
  resultDx
}

lvmodelDtheta <- function(theta, x, tvec) {
  H <- x[, 1]
  P <- x[, 2]
  resultDtheta <- array(0, c(nrow(x), length(theta), ncol(x)))
  # d(dH/dt)/d(a,b,c,d)
  resultDtheta[, 1, 1] <- H
  resultDtheta[, 2, 1] <- -H * P
  # d(dP/dt)/d(a,b,c,d)
  resultDtheta[, 3, 2] <- -P
  resultDtheta[, 4, 2] <- H * P
  resultDtheta
}

lvmodel <- list(
  fOde = lvmodelODE,
  fOdeDx = lvmodelDx,
  fOdeDtheta = lvmodelDtheta,
  thetaLowerBound = c(0, 0, 0, 0),
  thetaUpperBound = c(Inf, Inf, Inf, Inf)
)


# ── FitzHugh-Nagumo ───────────────────────────────────────────────────────
# Julia: du[1] = theta1*(V - V^3/3 + R)
#         du[2] = (1/theta1)*(V - theta2 + theta3*R)
# theta = (theta1, theta2, theta3) = (c, a, b), true = (3.0, 0.2, 0.2)
#
# NOTE: This matches the Julia parameterization in src/ode.jl.
# MAGI's built-in fnmodelODE uses a different sign convention for dR/dt,
# so we define our own model here for consistency with the Julia data.

fnmodelODE_jl <- function(theta, x, tvec) {
  V <- x[, 1]
  R <- x[, 2]
  result <- array(0, c(nrow(x), ncol(x)))
  result[, 1] <- theta[1] * (V - V^3 / 3.0 + R)
  result[, 2] <- (1.0 / theta[1]) * (V - theta[2] + theta[3] * R)
  result
}

fnmodelDx_jl <- function(theta, x, tvec) {
  V <- x[, 1]
  resultDx <- array(0, c(nrow(x), ncol(x), ncol(x)))
  # d(dV/dt)/dV, d(dV/dt)/dR
  resultDx[, 1, 1] <- theta[1] * (1 - V^2)
  resultDx[, 2, 1] <- theta[1]
  # d(dR/dt)/dV, d(dR/dt)/dR
  resultDx[, 1, 2] <- 1.0 / theta[1]
  resultDx[, 2, 2] <- theta[3] / theta[1]
  resultDx
}

fnmodelDtheta_jl <- function(theta, x, tvec) {
  V <- x[, 1]
  R <- x[, 2]
  resultDtheta <- array(0, c(nrow(x), length(theta), ncol(x)))
  # d(dV/dt)/d(theta1) = V - V^3/3 + R
  resultDtheta[, 1, 1] <- V - V^3 / 3.0 + R
  # d(dR/dt)/d(theta1) = -(1/theta1^2)*(V - theta2 + theta3*R)
  resultDtheta[, 1, 2] <- -(1.0 / theta[1]^2) * (V - theta[2] + theta[3] * R)
  # d(dR/dt)/d(theta2) = -1/theta1
  resultDtheta[, 2, 2] <- -1.0 / theta[1]
  # d(dR/dt)/d(theta3) = R/theta1
  resultDtheta[, 3, 2] <- R / theta[1]
  resultDtheta
}

fnmodel_jl <- list(
  fOde = fnmodelODE_jl,
  fOdeDx = fnmodelDx_jl,
  fOdeDtheta = fnmodelDtheta_jl,
  thetaLowerBound = c(0, 0, 0),
  thetaUpperBound = c(Inf, Inf, Inf)
)


# ── Protein Signaling Transduction (PST) ──────────────────────────────────
# Julia: du[1] = -θ1*S - θ2*S*R + θ3*Rs
#         du[2] = θ1*S
#         du[3] = -θ2*S*R + θ3*Rs + θ5*Rpp/(θ6+Rpp)
#         du[4] = θ2*S*R - θ3*Rs - θ4*Rs
#         du[5] = θ4*Rs - θ5*Rpp/(θ6+Rpp)
# theta = (k1, k2, k3, k4, V, Km), true = (0.07, 0.6, 0.05, 0.3, 0.017, 0.3)
#
# This matches MAGI's built-in ptransmodelODE exactly.
# We use the built-in for correctness but define an alias for clarity.

# ptransmodel will be loaded from the magi package in the run scripts.
# If magi is loaded, ptransmodelODE, ptransmodelDx, ptransmodelDtheta are available.


# ── Lotka-Volterra Competition (off-diagonal alpha only) ─────────────────
# Julia: du[i] = r[i]*u[i]*(1 - sum(alpha[i,j]*u[j])/K[i])
# Parameters: theta = off-diagonal alpha entries (K*(K-1) values),
#             packed row-major: [alpha_{1,2}, ..., alpha_{1,K}, alpha_{2,1}, ...]
# Fixed: r (growth rates), Ks (carrying capacities), diag_alpha = 1
#
# Usage: lvc_model <- make_lvc_model(K, r_true, Ks_true)

make_lvc_model <- function(K, r_true, Ks_true, diag_alpha = rep(1, K)) {

  # Reconstruct full alpha matrix from off-diagonal theta
  unpack_alpha <- function(theta) {
    alpha <- matrix(0, K, K)
    idx <- 1
    for (i in 1:K) {
      for (j in 1:K) {
        if (i == j) {
          alpha[i, j] <- diag_alpha[i]
        } else {
          alpha[i, j] <- theta[idx]
          idx <- idx + 1
        }
      }
    }
    alpha
  }

  fOde <- function(theta, x, tvec) {
    alpha <- unpack_alpha(theta)
    n_t <- nrow(x)
    result <- array(0, c(n_t, K))
    for (i in 1:K) {
      interaction <- x %*% alpha[i, ]  # n_t x 1
      result[, i] <- r_true[i] * x[, i] * (1 - interaction / Ks_true[i])
    }
    result
  }

  fOdeDx <- function(theta, x, tvec) {
    alpha <- unpack_alpha(theta)
    n_t <- nrow(x)
    resultDx <- array(0, c(n_t, K, K))
    # d(du[i]/dt)/dx[j]
    for (i in 1:K) {
      interaction <- x %*% alpha[i, ]  # n_t x 1
      for (j in 1:K) {
        if (i == j) {
          # d/dx_i of r_i * x_i * (1 - sum_j alpha_ij x_j / K_i)
          #   = r_i * (1 - sum_j alpha_ij x_j / K_i) + r_i * x_i * (-alpha_ii / K_i)
          resultDx[, j, i] <- r_true[i] * (1 - interaction / Ks_true[i]) -
            r_true[i] * x[, i] * alpha[i, j] / Ks_true[i]
        } else {
          # d/dx_j of r_i * x_i * (1 - sum_k alpha_ik x_k / K_i)
          #   = r_i * x_i * (-alpha_ij / K_i)
          resultDx[, j, i] <- -r_true[i] * x[, i] * alpha[i, j] / Ks_true[i]
        }
      }
    }
    resultDx
  }

  fOdeDtheta <- function(theta, x, tvec) {
    n_t <- nrow(x)
    n_theta <- K * (K - 1)
    resultDtheta <- array(0, c(n_t, n_theta, K))
    # d(du[i]/dt)/d(alpha_{p,q}) where (p,q) are off-diagonal
    # du[i]/dt = r_i * x_i * (1 - sum_j alpha_ij x_j / K_i)
    # d/d(alpha_{p,q}) = -r_p * x_p * x_q / K_p   if i == p
    #                  = 0                           otherwise
    idx <- 1
    for (p in 1:K) {
      for (q in 1:K) {
        if (p != q) {
          # This off-diagonal alpha_{p,q} only affects du[p]/dt
          resultDtheta[, idx, p] <- -r_true[p] * x[, p] * x[, q] / Ks_true[p]
          idx <- idx + 1
        }
      }
    }
    resultDtheta
  }

  list(
    fOde = fOde,
    fOdeDx = fOdeDx,
    fOdeDtheta = fOdeDtheta,
    thetaLowerBound = rep(0, K * (K - 1)),
    thetaUpperBound = rep(Inf, K * (K - 1))
  )
}
