#!/usr/bin/env Rscript
# R MAGI baseline for comparison with Julia MAGI implementation
# Runs on LV (Lotka-Volterra) with N=10, seed=42
#
# Usage:
#   Rscript experiments/magi/run_r_magi.R

library(magi)
library(coda)

# ── Paths ─────────────────────────────────────────────────────────────────
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("--file=", "", file_arg))))
  }
  tryCatch(dirname(sys.frame(1)$ofile), error = function(e) ".")
}
script_dir <- get_script_dir()
project_root <- normalizePath(file.path(script_dir, "..", ".."))

source(file.path(project_root, "baselines", "magi", "models.R"))

data_file   <- file.path(project_root, "baselines", "data", "lv_N10_seed42.csv")
results_dir <- file.path(project_root, "experiments", "magi", "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

# ── Load data ─────────────────────────────────────────────────────────────
if (!file.exists(data_file)) {
  stop(sprintf("Data file not found: %s\nRun generate_data.jl first.", data_file))
}
dat <- read.csv(data_file)
y_obs <- data.frame(time = dat$time, prey = dat$prey, predator = dat$predator)

# Discretization: insert intermediate points (MAGI standard practice)
y_input <- setDiscretization(y_obs, level = 1)
cat(sprintf("Data: N=%d observations, %d points after discretization\n",
            nrow(dat), nrow(y_input)))

# ── Run MAGI ──────────────────────────────────────────────────────────────
theta_true <- c(2.0, 1.0, 4.0, 1.0)
param_names <- c("a", "b", "c", "d")

niter_hmc  <- 20000L
nsteps_hmc <- 200L
burnin_ratio <- 0.5

cat(sprintf("Running R MAGI: niter=%d, nsteps=%d, burnin=%.0f%%\n",
            niter_hmc, nsteps_hmc, burnin_ratio * 100))

set.seed(42)
t_start <- proc.time()
result <- MagiSolver(
  y = y_input,
  odeModel = lvmodel,
  control = list(
    niterHmc = niter_hmc,
    nstepsHmc = nsteps_hmc,
    burninRatio = burnin_ratio
  )
)
t_elapsed <- (proc.time() - t_start)["elapsed"]

# ── Extract & save results ────────────────────────────────────────────────

# θ posterior samples (post burn-in)
theta_samples <- result$theta
colnames(theta_samples) <- param_names
theta_mean <- colMeans(theta_samples)
theta_sd <- apply(theta_samples, 2, sd)
theta_q025 <- apply(theta_samples, 2, quantile, 0.025)
theta_q975 <- apply(theta_samples, 2, quantile, 0.975)

# ESS
ess_vals <- effectiveSize(theta_samples)

# Save θ samples
write.csv(as.data.frame(theta_samples),
          file.path(results_dir, "r_magi_theta_samples.csv"),
          row.names = FALSE)

# Save σ samples
sigma_samples <- result$sigma
colnames(sigma_samples) <- c("sigma_prey", "sigma_predator")
write.csv(as.data.frame(sigma_samples),
          file.path(results_dir, "r_magi_sigma_samples.csv"),
          row.names = FALSE)

# Save trajectory statistics
x_samples <- result$xsampled  # n_samples x N_grid x 2
x_mean  <- apply(x_samples, c(2, 3), mean)
x_lower <- apply(x_samples, c(2, 3), quantile, 0.025)
x_upper <- apply(x_samples, c(2, 3), quantile, 0.975)

traj_df <- data.frame(
  t = y_input$time,
  mean_prey      = x_mean[, 1],  lower_prey      = x_lower[, 1],  upper_prey      = x_upper[, 1],
  mean_predator  = x_mean[, 2],  lower_predator  = x_lower[, 2],  upper_predator  = x_upper[, 2]
)
write.csv(traj_df,
          file.path(results_dir, "r_magi_trajectory.csv"),
          row.names = FALSE)

# ── Print summary ─────────────────────────────────────────────────────────
cat("\n")
cat("═══════════════════════════════════════════════════════\n")
cat("R MAGI Results (LV, N=10, seed=42)\n")
cat("═══════════════════════════════════════════════════════\n")
cat(sprintf("Time: %.1f sec\n", t_elapsed))
cat(sprintf("Post burn-in samples: %d\n", nrow(theta_samples)))
cat("\n")
cat(sprintf("%-6s  %6s  %12s  %12s  %8s\n",
            "param", "true", "mean±sd", "95% CI", "ESS"))
cat(strrep("-", 55), "\n")
for (i in seq_along(param_names)) {
  cat(sprintf("%-6s  %6.2f  %5.3f±%.3f  [%5.3f,%5.3f]  %7.0f\n",
              param_names[i], theta_true[i],
              theta_mean[i], theta_sd[i],
              theta_q025[i], theta_q975[i],
              ess_vals[i]))
}
rmsd <- sqrt(mean((theta_mean - theta_true)^2))
cat(sprintf("\nRMSD(θ): %.4f\n", rmsd))
cat("═══════════════════════════════════════════════════════\n")
