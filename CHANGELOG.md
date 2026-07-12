# Changelog

All notable changes to Relevent.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

Release driven by the 2026-07 expert-panel review: the `relevent` feature
surface is completed (13 Gibson p-shifts, `CovSnd`/`CovRec`/`CovInt`,
`rem.dyad`-style entry point), the placeholder fitters become real maximum
likelihood, and statistics stream over the event history instead of
rescanning it.

### Breaking

- **`fit_obpm` and `fit_timing` are real MLE fitters now — results change
  entirely.** The 0.1.0 versions were stubs (zero gradients, `NaN` standard
  errors, `converged=false`); the ordinal model now maximizes the exact
  full-risk-set multinomial likelihood and the timing model an
  exponential-baseline proportional-hazards likelihood, both via Newton with
  step halving. *Migration:* expect finite fitted coefficients/SEs; remove
  any workarounds that treated the outputs as placeholders.
- **Non-exponential timing baselines are no longer "fitted":** `fit_timing`
  errors for `:weibull`/`:gompertz` (previously it silently returned NaN
  placeholders); `hazard_rate`/`survival_function` still support them.
  *Migration:* estimate with `baseline=:exponential`.
- **Statistics now interoperate with `REM.EventNetworkState`** (REM's
  renamed `NetworkState`); every advanced statistic gained a
  `compute(stat, state, sender, receiver)` method. *Migration:* refer to
  `REM.EventNetworkState` anywhere you named `REM.NetworkState`.
- **Constructor/input validation may throw where 0.1.0 accepted garbage:**
  `PriorInteraction` requires `halflife > 0`; `OrdinalBPM` requires at least
  one statistic and two actors; out-of-range actor IDs raise
  `ArgumentError`. *Migration:* fix the offending inputs.
- **Stateful accumulator caches** were added to `PriorInteraction`,
  `SendingCapacity`, `ReceivingCapacity`, and `Momentum`; public
  constructors are unchanged, but full positional construction and sharing
  one statistic instance across threads are no longer safe. *Migration:*
  use the documented constructors; give each concurrent stream its own
  statistic instances.
- **Minimum Julia raised to 1.12**; package UUID regenerated;
  `DataFrames`/`Optim`/`StatsBase` dependencies dropped. *Migration:*
  upgrade Julia and re-resolve environments pinning the old UUID.

### Added

- `fit_relevent` standardized entry point with the R-faithful alias
  `rem_dyad` (mirrors `relevent::rem.dyad`; `ordinal=true/false` dispatches
  to the ordinal/timing likelihood).
- The 13 Gibson (2003) participation shifts: `PShift(:AB_BA)` etc.,
  accepting symbols or R names (`"PSAB-BA"`), with group-directed
  (`receiver == 0`) semantics and `pshift_types()` listing.
- Covariate effects `CovSnd`, `CovRec`, `CovInt` (sender / receiver /
  sender+receiver covariates indexed by actor ID), as in R `relevent`.
- `t0` observation-onset keyword on `fit_timing`/`fit_relevent` for
  left-truncated observation windows (first waiting time is `t₁ − t0`;
  validated against the first event time).
- StatsAPI `coef`/`stderror` methods on both result types; both results
  print through the shared `Network.print_coeftable` with Wald z/p-values.
- All advanced statistics implement REM's 4-argument `compute` interface,
  so they plug directly into `REM.fit_rem`/`generate_observations`.

### Changed

- `TimingModelResult` reports the fitted baseline rate λ₀ explicitly;
  `rank_events` and risk-set evaluation warn once on tied timestamps.
- Baseline hazard/survival functions cover
  exponential/Weibull/Gompertz with an explicit error on unknown baselines.

### Fixed

- The core correctness fix of this release: ordinal-BPM and timing fits
  return proper MLEs with observed-information standard errors (see
  Breaking).
- Time arithmetic wrapped in `float(...)` so non-`Float64` time types work
  in decay and waiting-time computations.

### Performance

- Statistics no longer rescan the full event history per evaluation:
  streaming decayed accumulators (absorbed once via a cursor, lazily
  decayed on read) back `PriorInteraction`, `SendingCapacity`,
  `ReceivingCapacity`, and `Momentum`, turning full-risk-set estimation
  from O(E²·statistics) toward linear absorption.
- Ordinal-likelihood derivatives use preallocated buffers, `mul!`, and
  `BLAS.ger!` instead of per-event matrix allocations; risk-set statistics
  are tuple-backed for static dispatch in the inner dyad loop.

## [0.1.0] - 2026-02-09

Initial release: ordinal and interval relational event model scaffolding
extending REM.jl.
