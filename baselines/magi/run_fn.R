#!/usr/bin/env Rscript
# MAGI baseline: FitzHugh-Nagumo
#
# NOTE: Uses custom FN model (fnmodel_jl) matching the Julia parameterization
# in src/ode.jl, NOT MAGI's built-in fnmodelODE (which has a different sign
# convention for dR/dt).
#
# Julia parameterization: theta = (c, a, b) with true values (3.0, 0.2, 0.2)
#   dV/dt = c*(V - V^3/3 + R)
#   dR/dt = (1/c)*(V - a + b*R)
#
# Usage:
#   Rscript baselines/magi/run_fn.R
#   Rscript baselines/magi/run_fn.R --N 40 --seed 42

library(magi)
library(coda)

script_dir <- dirname(sys.frame(1)$ofile)
source(file.path(script_dir, "models.R"))

# ── Configuration ──────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(flag, default) {
  idx <- which(args == flag)
  if (length(idx) > 0 && idx < length(args)) return(args[idx + 1])
  return(default)
}

N_values   <- as.integer(strsplit(parse_arg("--N", "10,25,40"), ",")[[1]])
seeds      <- as.integer(strsplit(parse_arg("--seed", "42,123,456,789,1234"), ",")[[1]])
niter_hmc  <- as.integer(parse_arg("--niter", "20000"))
nsteps_hmc <- as.integer(parse_arg("--nsteps", "200"))
burnin     <- as.numeric(parse_arg("--burnin", "0.5"))

theta_true <- c(3.0, 0.2, 0.2)  # (c, a, b) in Julia convention
param_names <- c("c", "a", "b")

data_dir    <- file.path(script_dir, "..", "data")
results_dir <- file.path(script_dir, "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

# ── Run ────────────────────────────────────────────────────────────────────

results_all <- data.frame()

for (N in N_values) {
  for (seed in seeds) {
    fname <- file.path(data_dir, sprintf("fn_N%d_seed%d.csv", N, seed))
    if (!file.exists(fname)) {
      cat(sprintf("SKIP: %s not found. Run generate_data.jl first.\n", fname))
      next
    }
    dat <- read.csv(fname)

    y_obs <- data.frame(time = dat$time, V = dat$V, R = dat$R)
    y_input <- setDiscretization(y_obs, level = 1)

    cat(sprintf("FN: N=%d, seed=%d, niter=%d ...\n", N, seed, niter_hmc))
    set.seed(seed)
    t_start <- proc.time()
    result <- MagiSolver(
      y = y_input,
      odeModel = fnmodel_jl,
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
      ode = "FN", method = "MAGI", N = N, seed = seed,
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
              file.path(results_dir, sprintf("fn_N%d_seed%d_samples.csv", N, seed)),
              row.names = FALSE)
  }
}

write.csv(results_all,
          file.path(results_dir, "fn_summary.csv"),
          row.names = FALSE)
cat("Results saved to", file.path(results_dir, "fn_summary.csv"), "\n")
