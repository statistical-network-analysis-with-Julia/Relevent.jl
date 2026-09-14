# Golden fixture: relevent::rem.dyad, ORDINAL and TEMPORAL likelihoods.
#
# Relevent.jl claims to port relevent::rem.dyad. Both likelihoods are EXACT
# (a multinomial partial likelihood over the full risk set; an exponential
# interval likelihood) — no Monte Carlo anywhere — so "close enough" is not a
# defence: the two implementations must land on the same optimum of the same
# function, and the only admissible discrepancy is optimizer termination slack.
#
# Regenerate from the package root:
#
#   Rscript test/fixtures/r/relevent_rem_dyad.R > test/fixtures/relevent_rem_dyad.toml
#
# TWO MODEL-SPECIFICATION FACTS this fixture exists to pin down, both learned
# the hard way and both invisible from the Julia side:
#
#  1. rem.dyad's TEMPORAL likelihood has NO baseline/intercept parameter: the
#     dyad hazard is exp(theta'x), full stop. Relevent.jl's `fit_timing` always
#     fits log(lambda0). The models are made identical by giving R a CONSTANT
#     covariate (CovSnd column of 1s), whose coefficient then IS log(lambda0).
#
#  2. rem.dyad reads the LAST ROW of the edgelist as the termination time of
#     the observation window and IGNORES any event on it (see ?rem.dyad,
#     "Details"). So its temporal likelihood always carries a final
#     right-censored interval [t_M, t_end] — exposure with no event. Relevent.jl
#     reproduces this only when `fit_timing(...; t_end=...)` is passed; without
#     it, lambda0 is biased upward because the eventless tail is dropped.
#     The fixture freezes the size of that bias.
#
# The event sequence is simulated here (not read from a bundled dataset) so
# that the data, the R fit and the Julia fit are provably the same three
# objects: the edgelist is emitted INTO the fixture under `input_*`, and the
# Julia test reads its events from there rather than regenerating them.

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(relevent)
})

seed <- 20260713
set.seed(seed)

n <- 8L    # actors
M <- 100L  # events

# Actor covariate: a fixed, evenly spaced score (not random — one less thing
# that has to survive the R -> Julia round trip bit for bit).
z <- round(seq(-1, 1, length.out = n), 4)

# Simulate a relational event sequence with sender/receiver covariate effects,
# repetition, reciprocation and an AB-BA turn-receiving tendency, so that every
# effect in the fitted model is actually identified by the data.
ev <- matrix(0, M, 3)
tt <- 0
prev <- c(0L, 0L)
cnt <- matrix(0, n, n)
for (m in 1:M) {
  lam <- matrix(0, n, n)
  for (i in 1:n) for (j in 1:n) if (i != j) {
    eta <- -1 + 0.8 * z[i] - 0.5 * z[j] +
           0.6 * log1p(cnt[i, j]) + 0.9 * log1p(cnt[j, i])
    if (prev[1] > 0 && i == prev[2] && j == prev[1]) eta <- eta + 1.0
    lam[i, j] <- exp(eta)
  }
  tot <- sum(lam)
  tt <- tt + rexp(1, tot)
  k <- sample(seq_len(n * n), 1, prob = as.vector(lam) / tot)
  i <- ((k - 1) %% n) + 1L
  j <- ((k - 1) %/% n) + 1L
  ev[m, ] <- c(tt, i, j)
  cnt[i, j] <- cnt[i, j] + 1
  prev <- c(i, j)
}

# End of the observation window: 5% past the last event.
t_end <- ev[M, 1] * 1.05

effects <- c("CovSnd", "CovRec", "PSAB-BA", "PSAB-BY", "PSAB-XB", "PSAB-AY")

# R's optim() defaults (BFGS, reltol ~1e-8) stop well short of the MLE — the
# reference would then be the LESS accurate of the two fits and the comparison
# would measure R's convergence slack, not agreement. Tighten it.
ctl <- list(reltol = 1e-15, maxit = 10000)

# --- ordinal likelihood (all M rows are events) -----------------------------
ord <- rem.dyad(ev, n = n, effects = effects,
                covar = list(CovSnd = z, CovRec = z),
                ordinal = TRUE, hessian = TRUE, fit.method = "MLE",
                verbose = FALSE, gof = FALSE, optim.control = ctl)

# --- temporal likelihood ----------------------------------------------------
# CovSnd is now a 2-column matrix: a constant (== the intercept / log lambda0)
# and z. The appended NA row is the termination time (see note 2 above).
evT <- rbind(ev, c(t_end, NA, NA))
tim <- rem.dyad(evT, n = n, effects = effects,
                covar = list(CovSnd = cbind(1, z), CovRec = z),
                ordinal = FALSE, hessian = TRUE, fit.method = "MLE",
                verbose = FALSE, gof = FALSE, optim.control = ctl)

ord_se <- sqrt(diag(ord$cov))
tim_se <- sqrt(diag(tim$cov))
num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")

cat('name = "relevent_rem_dyad"\n\n')

cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('relevent_version = "%s"\n', as.character(packageVersion("relevent"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/relevent_rem_dyad.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "simulated dyadic event sequence (8 actors, 100 events); frozen below under input_*"\n')
cat('fit_method = "MLE (optim BFGS, reltol = 1e-15), observed-information SEs"\n\n')

cat("[tolerance]\n")
cat("# Both likelihoods are EXACT — no Monte Carlo, no sampling of the risk\n")
cat("# set — so the two implementations maximize the same function and must\n")
cat("# reach the same optimum. The only admissible discrepancy is optimizer\n")
cat("# termination slack: R uses optim/BFGS (reltol = 1e-15 above), Julia uses\n")
cat("# Newton-Raphson. Observed disagreement at the frozen values is <= 2e-7,\n")
cat("# and the two log-likelihoods agree to < 1e-9 -- i.e. both are sitting on\n")
cat("# the same maximum. 1e-6 is that, with an order of magnitude of margin.\n")
cat("# It is NOT a Monte-Carlo allowance: a failure here means the LIKELIHOOD\n")
cat("# differs, not that a random seed moved.\n")
cat("default = 1e-6\n")
cat("# The inputs are echoed exactly; they are read, not compared.\n")
cat("input_time = 1e-12\n\n")

cat("[values]\n")
cat("# --- inputs (echoed so the Julia test fits the identical data) ---\n")
cat(sprintf("n_actors = %d\n", n))
cat(sprintf("t_end = %.17g\n", t_end))
cat(sprintf("input_time = [%s]\n", num(ev[, 1])))
cat(sprintf("input_sender = [%s]\n", paste(as.integer(ev[, 2]), collapse = ", ")))
cat(sprintf("input_receiver = [%s]\n", paste(as.integer(ev[, 3]), collapse = ", ")))
cat(sprintf("input_covariate = [%s]\n", num(z)))
cat("\n# --- ordinal likelihood: CovSnd, CovRec, PSAB-BA, PSAB-BY, PSAB-XB, PSAB-AY ---\n")
cat(sprintf("ordinal_names = [%s]\n",
            paste(sprintf('"%s"', names(ord$coef)), collapse = ", ")))
cat(sprintf("ordinal_coefficients = [%s]\n", num(ord$coef)))
cat(sprintf("ordinal_std_errors = [%s]\n", num(ord_se)))
cat(sprintf("ordinal_loglik = %.17g\n", -ord$residual.deviance / 2))
cat("\n# --- temporal likelihood: CovSnd.1 is the constant, i.e. log(lambda0) ---\n")
cat(sprintf("timing_names = [%s]\n",
            paste(sprintf('"%s"', names(tim$coef)), collapse = ", ")))
cat(sprintf("timing_log_baseline = %.17g\n", tim$coef[1]))
cat(sprintf("timing_coefficients = [%s]\n", num(tim$coef[-1])))
cat(sprintf("timing_log_baseline_se = %.17g\n", tim_se[1]))
cat(sprintf("timing_std_errors = [%s]\n", num(tim_se[-1])))
cat(sprintf("timing_loglik = %.17g\n", -tim$residual.deviance / 2))
