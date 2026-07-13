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

- **`cache=:auto|:all|:chunked|:none` on `fit_obpm`, `fit_timing` and
  `fit_relevent`** (issue #2), with `chunk` and `cache_bytes`, bounding the
  memory the full risk sets take. See **Performance** below for the numbers
  and for why the fits are bit-identical under every policy.
- **Tied event times are now a policy, not a warning — and the two likelihoods
  take DIFFERENT policies** (issue #1, review finding 12). One vocabulary
  (`Networks.TIE_POLICIES`, defined once in Networks.jl and shared with REM.jl),
  one keyword (`ties=`), both fitters defaulting to **`:error`** — but
  `fit_obpm` and `fit_timing` claim different things about time, so each accepts
  a different subset and **refuses** the rest with an explanation instead of
  silently accepting it:
  - `fit_obpm` (a likelihood over the *order* of events) takes
    `:error | :ordered | :breslow | :efron`. `:breslow` freezes the interaction
    history across a tie block, so the simultaneous events cannot enter one
    another's statistics, and gives each the same denominator; `:efron` adds the
    `1 − (j−1)/d` denominator weights on the tied cases (and requires them to be
    distinct dyads). `:batch` is **refused**: with the history frozen, a
    simultaneous batch in an ordinal likelihood *is* the Breslow correction.
  - `fit_timing` (an **exact-time** likelihood) takes `:error | :ordered | :batch`.
    `:breslow`/`:efron` are **refused** and not as a matter of taste: they correct
    a *partial* likelihood, in which the baseline hazard is profiled out and only
    the order survives; there is no such denominator in an exponential interval
    likelihood. Under a continuous-time model a tie has probability **zero** — it
    is a violated assumption, not an ambiguous ordering — so the policies say what
    the tie *means*: `:batch` (a coarsened simultaneous batch: statistics frozen,
    one exposure interval) or `:ordered` (the legacy behaviour: each tied event
    after the first enters with a zero-length waiting interval while still
    updating the next one's statistics, i.e. one event caused another in no time
    at all).
- **`tie_method(fit)` now reports what ACTUALLY happened** on both result types:
  `:none` when the data had no ties (it no longer names a correction that
  corrected nothing), otherwise the policy that bit; `:error` can never appear.
  `approximations(fit)` carries the matching caveat, `show` prints the policy only
  when it bit, and **`is_exact(fit)` is now `false` whenever ties were present** —
  under every policy, because no policy can make a likelihood over an unobserved
  order (or an exact-time likelihood on data the exact-time model says cannot
  occur) exact. On tie-free data every policy gives the identical fit.

- **Real R golden fixtures against `relevent::rem.dyad`** (issue #8): the
  ordinal and temporal likelihoods are frozen from an actual R run
  (relevent 1.2.1, R 4.6.1) in `test/fixtures/relevent_rem_dyad.toml`, with
  the generating script `test/fixtures/r/relevent_rem_dyad.R` beside it and
  the full provenance the Networks.jl `load_golden` harness demands. Both
  likelihoods are exact — no Monte Carlo — so the tolerance is **1e-6**, and
  that is not a fudge factor but the reference optimizer's termination slack:
  the two fits' log-likelihoods agree to <1e-9, i.e. they sit on the same
  maximum. Agreement observed: ≤ 2e-7 on every coefficient and standard error
  of both models.
- **`fit_timing(...; t_end=)` — the observation-window end.** `rem.dyad`'s
  temporal likelihood *always* carries a final right-censored interval (its
  last edgelist row is the termination time, and any event on it is ignored);
  `fit_timing` had no way to express one, so it implicitly ended the
  observation window at the last event and biased the baseline rate λ₀ upward.
  On the golden sequence that bias is **0.047 in log λ₀** — 350× the tolerance
  above, and enough to move a reported event rate by 5%. `t_end` adds the
  censored interval `[t_M, t_end]` (exposure, no event term) and closes the
  gap; the default (`nothing`) preserves the old behavior, and the golden
  testset pins both branches so the gap cannot come back silently.
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
  print through the shared `Networks.print_coeftable` with Wald z/p-values.
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

- **The full-risk-set likelihood streams (issue #2).** The risk sets used
  to be materialized in full — an `n(n−1) × p` design matrix for *every*
  event, `O(E · n² · p)` doubles, 906 MB at `(n, E, p) = (100, 2000, 6)` —
  which put a ceiling on exact estimation well below the point where the
  arithmetic gets hard. `fit_obpm`, `fit_timing` and `fit_relevent` now
  take `cache=`:

  | `cache` | design matrices alive | (100, 2000, 6) | time |
  |---|---|---|---|
  | `:all` | all `E` | 906.4 MB | 11.0 s |
  | `:chunked` (`chunk=32`) | `chunk` | 14.5 MB | 34.1 s |
  | `:none` | 1 | 0.5 MB | 34.9 s |

  with `chunk` (default: as many as fit in `cache_bytes`, 256 MiB) exposing
  the bound. The default is `cache=:auto`: `:all` while the projected
  footprint fits in `cache_bytes`, `:chunked` above it — `:all` is right
  where it is affordable, and the case where it is *not* affordable is
  exactly the one that used to fall over. A `chunk` covering every interval
  collapses to `:all` rather than recomputing for nothing.

  **The fits are bit-identical across every policy** — `==`, not `≈`, and
  asserted against the `relevent_rem_dyad.toml` golden fixture under each
  one, including under the `:breslow`/`:efron` tie corrections (where
  `:all` shares one design matrix across a frozen tie block and the
  streamed policies recompute it from the same frozen history). Only memory
  and time move.
- Statistics no longer rescan the full event history per evaluation:
  streaming decayed accumulators (absorbed once via a cursor, lazily
  decayed on read) back `PriorInteraction`, `SendingCapacity`,
  `ReceivingCapacity`, and `Momentum`, turning full-risk-set estimation
  from O(E²·statistics) toward linear absorption.
- **`fit_timing`'s derivative loop no longer allocates (review finding
  15).** The per-interval `Xm' * (w .* Xm)` allocated a whole weighted copy
  of the risk-set design matrix plus a fresh `p×p` outer product on every
  event of every Newton evaluation; it is now `WX .= w .* Xm;
  mul!(Sxx, Xm', WX)` on preallocated workspaces — the same products in the
  same order, in place. Both likelihoods' derivative closures
  (`_obpm_derivatives`, `_timing_derivatives`) now allocate only the `p`
  gradient and `p×p` Hessian they hand to `_newton`: 192 bytes per
  evaluation, independent of `E` and of the risk-set size, pinned by an
  `@allocated` regression test.
- Ordinal-likelihood derivatives use preallocated buffers, `mul!`, and
  `BLAS.ger!` instead of per-event matrix allocations; risk-set statistics
  are tuple-backed for static dispatch in the inner dyad loop.

## [0.1.0] - 2026-02-09

Initial release: ordinal and interval relational event model scaffolding
extending REM.jl.
