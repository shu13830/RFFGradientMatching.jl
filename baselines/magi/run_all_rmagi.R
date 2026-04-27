#!/usr/bin/env Rscript
# R MAGI comparison: LV, PST, FN × 3 θ patterns × 3 N × 5 seeds
# Uses default R MAGI settings (niterHmc=20000, nstepsHmc=200, burninRatio=0.5)

library(magi)
library(coda)

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_dir <- dirname(normalizePath(script_path))
data_dir <- file.path(script_dir, "..", "data")
results_dir <- file.path(script_dir, "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

# ── ODE Models ──

# Lotka-Volterra
lv_ode <- function(theta, x, tvec) {
  a <- theta[1]; b <- theta[2]; c <- theta[3]; d <- theta[4]
  dx1 <- a * x[,1] - b * x[,1] * x[,2]
  dx2 <- -c * x[,2] + d * x[,1] * x[,2]
  cbind(dx1, dx2)
}
lv_dx <- function(theta, x, tvec) {
  a <- theta[1]; b <- theta[2]; c <- theta[3]; d <- theta[4]
  n <- nrow(x)
  result <- array(0, dim = c(n, 2, 2))
  result[,1,1] <- a - b * x[,2]
  result[,2,1] <- -b * x[,1]
  result[,1,2] <- d * x[,2]
  result[,2,2] <- -c + d * x[,1]
  result
}
lv_dtheta <- function(theta, x, tvec) {
  n <- nrow(x)
  result <- array(0, dim = c(n, 4, 2))
  result[,1,1] <- x[,1]
  result[,2,1] <- -x[,1] * x[,2]
  result[,3,2] <- -x[,2]
  result[,4,2] <- x[,1] * x[,2]
  result
}
lvmodel <- list(fOde=lv_ode, fOdeDx=lv_dx, fOdeDtheta=lv_dtheta,
                thetaLowerBound=rep(0, 4), thetaUpperBound=rep(Inf, 4))

# FitzHugh-Nagumo
fn_ode <- function(theta, x, tvec) {
  th1 <- theta[1]; th2 <- theta[2]; th3 <- theta[3]
  dV <- th1 * (x[,1] - x[,1]^3/3 + x[,2])
  dR <- (1/th1) * (x[,1] - th2 + th3 * x[,2])
  cbind(dV, dR)
}
fn_dx <- function(theta, x, tvec) {
  th1 <- theta[1]; th3 <- theta[3]
  n <- nrow(x)
  result <- array(0, dim = c(n, 2, 2))
  result[,1,1] <- th1 * (1 - x[,1]^2)
  result[,2,1] <- th1
  result[,1,2] <- 1/th1
  result[,2,2] <- th3/th1
  result
}
fn_dtheta <- function(theta, x, tvec) {
  th1 <- theta[1]; th2 <- theta[2]; th3 <- theta[3]
  n <- nrow(x)
  result <- array(0, dim = c(n, 3, 2))
  result[,1,1] <- x[,1] - x[,1]^3/3 + x[,2]
  result[,1,2] <- -(1/th1^2) * (x[,1] - th2 + th3 * x[,2])
  result[,2,2] <- -1/th1
  result[,3,2] <- x[,2]/th1
  result
}
fnmodel <- list(fOde=fn_ode, fOdeDx=fn_dx, fOdeDtheta=fn_dtheta,
                thetaLowerBound=rep(0, 3), thetaUpperBound=rep(Inf, 3))

# Protein Signaling Transduction
pst_ode <- function(theta, x, tvec) {
  th <- theta
  S <- x[,1]; dS <- x[,2]; R <- x[,3]; Rs <- x[,4]; Rpp <- x[,5]
  dx1 <- -th[1]*S - th[2]*S*R + th[3]*Rs
  dx2 <- th[1]*S
  dx3 <- -th[2]*S*R + th[3]*Rs + th[5]*Rpp/(th[6]+Rpp)
  dx4 <- th[2]*S*R - th[3]*Rs - th[4]*Rs
  dx5 <- th[4]*Rs - th[5]*Rpp/(th[6]+Rpp)
  cbind(dx1, dx2, dx3, dx4, dx5)
}
pst_dx <- function(theta, x, tvec) {
  th <- theta
  n <- nrow(x)
  S <- x[,1]; R <- x[,3]; Rs <- x[,4]; Rpp <- x[,5]
  denom <- (th[6]+Rpp)^2
  result <- array(0, dim = c(n, 5, 5))
  result[,1,1] <- -th[1] - th[2]*R; result[,3,1] <- -th[2]*S; result[,4,1] <- th[3]
  result[,1,2] <- th[1]
  result[,1,3] <- -th[2]*R; result[,3,3] <- -th[2]*S; result[,4,3] <- th[3]; result[,5,3] <- th[5]*th[6]/denom
  result[,1,4] <- th[2]*R; result[,3,4] <- th[2]*S; result[,4,4] <- -th[3]-th[4]
  result[,4,5] <- th[4]; result[,5,5] <- -th[5]*th[6]/denom
  result
}
pst_dtheta <- function(theta, x, tvec) {
  th <- theta
  n <- nrow(x)
  S <- x[,1]; R <- x[,3]; Rs <- x[,4]; Rpp <- x[,5]
  denom <- th[6]+Rpp
  result <- array(0, dim = c(n, 6, 5))
  result[,1,1] <- -S; result[,2,1] <- -S*R; result[,3,1] <- Rs
  result[,1,2] <- S
  result[,2,3] <- -S*R; result[,3,3] <- Rs; result[,5,3] <- Rpp/denom; result[,6,3] <- -th[5]*Rpp/denom^2
  result[,2,4] <- S*R; result[,3,4] <- -Rs; result[,4,4] <- -Rs
  result[,4,5] <- Rs; result[,5,5] <- -Rpp/denom; result[,6,5] <- th[5]*Rpp/denom^2
  result
}
pstmodel <- list(fOde=pst_ode, fOdeDx=pst_dx, fOdeDtheta=pst_dtheta,
                 thetaLowerBound=rep(0, 6), thetaUpperBound=rep(Inf, 6))

# SIR
sir_ode <- function(theta, x, tvec) {
  a <- theta[1]; b <- theta[2]
  S <- x[,1]; I <- x[,2]; R <- x[,3]
  dS <- -a * S * I
  dI <- a * S * I - b * I
  dR <- b * I
  cbind(dS, dI, dR)
}
sir_dx <- function(theta, x, tvec) {
  a <- theta[1]; b <- theta[2]
  n <- nrow(x)
  S <- x[,1]; I <- x[,2]; R <- x[,3]
  result <- array(0, dim = c(n, 3, 3))
  result[,1,1] <- -a * I;  result[,2,1] <- -a * S
  result[,1,2] <- a * I;   result[,2,2] <- a * S - b
  result[,2,3] <- b
  result
}
sir_dtheta <- function(theta, x, tvec) {
  n <- nrow(x)
  S <- x[,1]; I <- x[,2]; R <- x[,3]
  result <- array(0, dim = c(n, 2, 3))
  result[,1,1] <- -S * I
  result[,1,2] <- S * I;   result[,2,2] <- -I
  result[,2,3] <- I
  result
}
sirmodel <- list(fOde=sir_ode, fOdeDx=sir_dx, fOdeDtheta=sir_dtheta,
                 thetaLowerBound=rep(0, 2), thetaUpperBound=rep(Inf, 2))

# ── Config ──
models <- list(lv=lvmodel, fn=fnmodel, pst=pstmodel, sir=sirmodel)
ode_names <- c("lv", "fn", "pst", "sir")
comp_names <- list(lv=c("prey","predator"), fn=c("V","R"), pst=c("S","dS","R","Rs","Rpp"), sir=c("S","I","R"))

args <- commandArgs(trailingOnly = TRUE)
target_ode <- if (length(args) >= 1) args[1] else "all"
target_N <- if (length(args) >= 2) as.integer(args[2]) else 25
target_tid <- if (length(args) >= 3) as.integer(args[3]) else 0  # 0 = all

N_values <- target_N
theta_ids <- if (target_tid == 0) c(1, 2, 3) else target_tid
seeds <- c(42, 123, 456, 789, 1234)
niter <- 20000
nsteps <- 200

# Read theta_true
theta_df <- read.csv(file.path(data_dir, "theta_true.csv"), stringsAsFactors=FALSE)

results_all <- data.frame()

for (ode in ode_names) {
  if (target_ode != "all" && ode != target_ode) next
  model <- models[[ode]]

  for (tid in theta_ids) {
    # Get true theta
    row <- theta_df[theta_df$ode == toupper(ode) & theta_df$theta_id == tid, ]
    theta_true <- as.numeric(strsplit(row$theta_values, ";")[[1]])

    for (seed in seeds) {
      fname <- file.path(data_dir, sprintf("%s_N%d_t%d_seed%d.csv", ode, target_N, tid, seed))
      if (!file.exists(fname)) {
        cat(sprintf("SKIP: %s not found\n", fname))
        next
      }
      dat <- read.csv(fname)

      y_obs <- as.data.frame(dat[, -1])  # remove time column
      y_obs <- cbind(time = dat$time, y_obs)

      y_input <- setDiscretization(y_obs, level = 1)

      cat(sprintf("%s θ=%d N=%d seed=%d ...\n", toupper(ode), tid, target_N, seed))
      set.seed(seed)
      t_start <- proc.time()

      result <- tryCatch({
        MagiSolver(y = y_input, odeModel = model,
                   control = list(niterHmc = niter, nstepsHmc = nsteps, burninRatio = 0.5))
      }, error = function(e) {
        cat(sprintf("  ERROR: %s\n", e$message))
        return(NULL)
      })

      t_elapsed <- (proc.time() - t_start)["elapsed"]

      if (is.null(result)) {
        row_out <- data.frame(ode=toupper(ode), theta_id=tid, N=target_N, seed=seed,
                              rmsd=NA, ess_mean=NA, time_sec=as.numeric(t_elapsed), status="ERROR")
        results_all <- rbind(results_all, row_out)
        next
      }

      theta_samples <- result$theta
      theta_mean <- colMeans(theta_samples)
      rmsd <- sqrt(mean((theta_mean - theta_true)^2))
      ess_vals <- effectiveSize(theta_samples)
      ess_mean <- mean(ess_vals)

      cat(sprintf("  RMSD=%.4f, ESS=%.1f, time=%.1fs\n", rmsd, ess_mean, t_elapsed))

      row_out <- data.frame(ode=toupper(ode), theta_id=tid, N=target_N, seed=seed,
                            rmsd=rmsd, ess_mean=ess_mean, time_sec=as.numeric(t_elapsed), status="OK")
      for (i in seq_along(theta_true)) {
        row_out[[paste0("theta_mean_", i)]] <- theta_mean[i]
        row_out[[paste0("theta_true_", i)]] <- theta_true[i]
      }
      results_all <- rbind(results_all, row_out)
    }
  }
}

out_file <- file.path(results_dir, sprintf("rmagi_%s_N%d.csv", target_ode, target_N))
write.csv(results_all, out_file, row.names = FALSE)
cat(sprintf("Results saved to %s\n", out_file))
