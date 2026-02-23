#!/usr/bin/env Rscript
# MAGI baseline: Protein Signaling Transduction (PST)
#
# Uses MAGI's built-in ptransmodelODE which matches the Julia implementation.
# theta = (k1, k2, k3, k4, V, Km), true = (0.07, 0.6, 0.05, 0.3, 0.017, 0.3)
#
# Usage:
#   Rscript baselines/magi/run_pst.R
#   Rscript baselines/magi/run_pst.R --N 15 --seed 42

library(magi)
library(coda)

script_dir <- dirname(sys.frame(1)$ofile)

# ── Configuration ──────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(flag, default) {
  idx <- which(args == flag)
  if (length(idx) > 0 && idx < length(args)) return(args[idx + 1])
  return(default)
}

N_values   <- as.integer(strsplit(parse_arg("--N", "15"), ",")[[1]])
seeds      <- as.integer(strsplit(parse_arg("--seed", "42,123,456,789,1234"), ",")[[1]])
niter_hmc  <- as.integer(parse_arg("--niter", "20000"))
nsteps_hmc <- as.integer(parse_arg("--nsteps", "200"))
burnin     <- as.numeric(parse_arg("--burnin", "0.5"))

theta_true <- c(0.07, 0.6, 0.05, 0.3, 0.017, 0.3)
param_names <- c("k1", "k2", "k3", "k4", "V", "Km")
component_names <- c("S", "dS", "R", "Rs", "Rpp")

# Built-in MAGI model for PST
pstmodel <- list(
  fOde = ptransmodelODE,
  fOdeDx = ptransmodelDx,
  fOdeDtheta = ptransmodelDtheta,
  thetaLowerBound = rep(0, 6),
  thetaUpperBound = rep(Inf, 6)
)

data_dir    <- file.path(script_dir, "..", "data")
results_dir <- file.path(script_dir, "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

# ── Run ────────────────────────────────────────────────────────────────────

results_all <- data.frame()

for (N in N_values) {
  for (seed in seeds) {
    fname <- file.path(data_dir, sprintf("pst_N%d_seed%d.csv", N, seed))
    if (!file.exists(fname)) {
      cat(sprintf("SKIP: %s not found. Run generate_data.jl first.\n", fname))
      next
    }
    dat <- read.csv(fname)

    y_obs <- data.frame(
      time = dat$time,
      S = dat$S, dS = dat$dS, R = dat$R, Rs = dat$Rs, Rpp = dat$Rpp
    )
    y_input <- setDiscretization(y_obs, level = 1)

    cat(sprintf("PST: N=%d, seed=%d, niter=%d ...\n", N, seed, niter_hmc))
    set.seed(seed)
    t_start <- proc.time()
    result <- MagiSolver(
      y = y_input,
      odeModel = pstmodel,
      control = list(
        niterHmc = niter_hmc,
        nstepsHmc = nsteps_hmc,
        burninRatio = burnin
      )
    )
    t_elapsed <- (proc.time() - t_start)["elapsed"]

    theta_samples <- result$theta
    theta_mean <- colMeans(theta_samples)
    rmsd <- sqrt(mean((theta_mean - theta_true)^2))
    ess_vals <- effectiveSize(theta_samples)
    ess_mean <- mean(ess_vals)
    ess_per_sec <- ess_mean / as.numeric(t_elapsed)

    cat(sprintf("  RMSD=%.4f, ESS_mean=%.1f, time=%.1fs, ESS/sec=%.2f\n",
                rmsd, ess_mean, t_elapsed, ess_per_sec))

    row <- data.frame(
      ode = "PST", method = "MAGI", N = N, seed = seed,
      rmsd = rmsd, ess_mean = ess_mean, time_sec = as.numeric(t_elapsed),
      ess_per_sec = ess_per_sec
    )
    for (i in seq_along(param_names)) {
      row[[paste0("theta_", param_names[i], "_mean")]] <- theta_mean[i]
      row[[paste0("theta_", param_names[i], "_true")]] <- theta_true[i]
      row[[paste0("ess_", param_names[i])]] <- ess_vals[i]
    }
    results_all <- rbind(results_all, row)

    samples_df <- as.data.frame(theta_samples)
    colnames(samples_df) <- param_names
    write.csv(samples_df,
              file.path(results_dir, sprintf("pst_N%d_seed%d_samples.csv", N, seed)),
              row.names = FALSE)
  }
}

write.csv(results_all,
          file.path(results_dir, "pst_summary.csv"),
          row.names = FALSE)
cat("Results saved to", file.path(results_dir, "pst_summary.csv"), "\n")
