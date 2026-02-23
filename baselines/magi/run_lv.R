#!/usr/bin/env Rscript
# MAGI baseline: Lotka-Volterra Predator-Prey
#
# Usage:
#   Rscript baselines/magi/run_lv.R                          # all N, all seeds
#   Rscript baselines/magi/run_lv.R --N 40 --seed 42         # single run
#   Rscript baselines/magi/run_lv.R --niter 20000 --nsteps 200

library(magi)
library(coda)

# Source custom model definitions
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("--file=", "", file_arg))))
  }
  # Fallback: try sys.frame (works in source() context)
  tryCatch(dirname(sys.frame(1)$ofile), error = function(e) ".")
}
script_dir <- get_script_dir()
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
do_trajectory <- "--trajectory" %in% args

theta_true <- c(2.0, 1.0, 4.0, 1.0)
param_names <- c("a", "b", "c", "d")

data_dir    <- file.path(script_dir, "..", "data")
results_dir <- file.path(script_dir, "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

# ── Run ────────────────────────────────────────────────────────────────────

results_all <- data.frame()

for (N in N_values) {
  for (seed in seeds) {
    # Load data
    fname <- file.path(data_dir, sprintf("lv_N%d_seed%d.csv", N, seed))
    if (!file.exists(fname)) {
      cat(sprintf("SKIP: %s not found. Run generate_data.jl first.\n", fname))
      next
    }
    dat <- read.csv(fname)

    # Format for MAGI: data.frame with "time" column
    y_obs <- data.frame(time = dat$time, prey = dat$prey, predator = dat$predator)

    # Discretization: insert intermediate points
    y_input <- setDiscretization(y_obs, level = 1)

    # Run MAGI
    cat(sprintf("LV: N=%d, seed=%d, niter=%d ...\n", N, seed, niter_hmc))
    set.seed(seed)
    t_start <- proc.time()
    result <- MagiSolver(
      y = y_input,
      odeModel = lvmodel,
      control = list(
        niterHmc = niter_hmc,
        nstepsHmc = nsteps_hmc,
        burninRatio = burnin
      )
    )
    t_elapsed <- (proc.time() - t_start)["elapsed"]

    # Extract posterior samples (post burn-in)
    theta_samples <- result$theta  # matrix: n_samples x n_params

    # Posterior mean
    theta_mean <- colMeans(theta_samples)

    # RMSD
    rmsd <- sqrt(mean((theta_mean - theta_true)^2))

    # ESS (using coda)
    ess_vals <- effectiveSize(theta_samples)
    ess_mean <- mean(ess_vals)
    ess_per_sec <- ess_mean / as.numeric(t_elapsed)

    cat(sprintf("  RMSD=%.4f, ESS_mean=%.1f, time=%.1fs, ESS/sec=%.2f\n",
                rmsd, ess_mean, t_elapsed, ess_per_sec))

    # Save summary
    row <- data.frame(
      ode = "LV", method = "MAGI", N = N, seed = seed,
      rmsd = rmsd, ess_mean = ess_mean, time_sec = as.numeric(t_elapsed),
      ess_per_sec = ess_per_sec
    )
    for (i in seq_along(param_names)) {
      row[[paste0("theta_", param_names[i], "_mean")]] <- theta_mean[i]
      row[[paste0("theta_", param_names[i], "_true")]] <- theta_true[i]
      row[[paste0("ess_", param_names[i])]] <- ess_vals[i]
    }
    results_all <- rbind(results_all, row)

    # Save full posterior samples
    samples_df <- as.data.frame(theta_samples)
    colnames(samples_df) <- param_names
    write.csv(samples_df,
              file.path(results_dir, sprintf("lv_N%d_seed%d_samples.csv", N, seed)),
              row.names = FALSE)

    # Trajectory output (Exp6)
    if (do_trajectory) {
      x_samples <- result$xsampled  # array: n_samples x N_grid x K
      x_mean  <- apply(x_samples, c(2, 3), mean)
      x_lower <- apply(x_samples, c(2, 3), quantile, 0.025)
      x_upper <- apply(x_samples, c(2, 3), quantile, 0.975)

      comp_names <- c("prey", "predator")
      traj_df <- data.frame(t = y_input$time)
      for (k in seq_along(comp_names)) {
        traj_df[[paste0("mean_", comp_names[k])]]  <- x_mean[, k]
        traj_df[[paste0("lower_", comp_names[k])]] <- x_lower[, k]
        traj_df[[paste0("upper_", comp_names[k])]] <- x_upper[, k]
      }

      traj_fname <- file.path(results_dir, "magi_lv_trajectory.csv")
      write.csv(traj_df, traj_fname, row.names = FALSE)
      cat(sprintf("  Trajectory saved to %s\n", traj_fname))
    }
  }
}

# Save aggregated results
write.csv(results_all,
          file.path(results_dir, "lv_summary.csv"),
          row.names = FALSE)
cat("Results saved to", file.path(results_dir, "lv_summary.csv"), "\n")
