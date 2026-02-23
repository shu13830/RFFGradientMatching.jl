#!/usr/bin/env Rscript
# Setup script for MAGI baseline experiments.
#
# Usage:
#   Rscript baselines/magi/setup.R

cat("Installing required R packages...\n")

# MAGI: Manifold-Constrained Gaussian Process Inference for ODEs
if (!requireNamespace("magi", quietly = TRUE)) {
  install.packages("magi", repos = "https://cloud.r-project.org")
}

# coda: for effectiveSize() (ESS computation)
if (!requireNamespace("coda", quietly = TRUE)) {
  install.packages("coda", repos = "https://cloud.r-project.org")
}

cat("Checking installations...\n")
library(magi)
library(coda)
cat(sprintf("  magi version: %s\n", packageVersion("magi")))
cat(sprintf("  coda version: %s\n", packageVersion("coda")))

# Verify LV model definitions
script_dir <- dirname(sys.frame(1)$ofile)
source(file.path(script_dir, "models.R"))

# Quick sanity check: evaluate LV ODE at a test point
x_test <- matrix(c(5.0, 3.0), nrow = 1)
theta_test <- c(2.0, 1.0, 4.0, 1.0)
dx <- lvmodelODE(theta_test, x_test, 0)
# Expected: du[1] = 2*5 - 1*5*3 = -5, du[2] = -4*3 + 1*5*3 = 3
stopifnot(abs(dx[1, 1] - (-5.0)) < 1e-10)
stopifnot(abs(dx[1, 2] - 3.0) < 1e-10)

# Verify gradient functions via finite differences
testDynamicalModel(lvmodelODE, lvmodelDx, lvmodelDtheta,
                   "Lotka-Volterra", x_test, theta_test, 0)
cat("  LV model: OK\n")

testDynamicalModel(fnmodelODE_jl, fnmodelDx_jl, fnmodelDtheta_jl,
                   "FitzHugh-Nagumo (Julia param.)",
                   matrix(c(-1.0, 1.0), nrow = 1), c(3.0, 0.2, 0.2), 0)
cat("  FN model: OK\n")

cat("\nSetup complete. Ready to run experiments.\n")
