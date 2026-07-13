# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Relevent.jl is a Julia port of the R `relevent` package from the StatNet collection. It extends REM.jl with advanced relational event model features including interaction history tracking, ordinal Butts-Park Models (BPM), parametric timing models, and decay-weighted network statistics.

## Development Commands

- **Run tests:** `julia --project -e 'using Pkg; Pkg.test()'`
- **Build docs:** `julia --project=docs docs/make.jl`
- **Load in REPL:** `julia --project` then `using Relevent`
- **Install local deps:** `julia --project -e 'using Pkg; Pkg.instantiate()'`

Note: This package depends on the local (non-registered) package `REM` via a relative path (`../REM.jl`). That sibling directory must be present.

## Architecture

The entire package lives in a single file: `src/Relevent.jl`. It is organized into these sections:

1. **Interaction History Tracking** -- `InteractionHistory{T}` struct that records per-dyad event times, sender/receiver histories, and event counts. Mutated via `update_history!`.
2. **Advanced REM Statistics** -- `PriorInteraction`, `SendingCapacity`, `ReceivingCapacity`, `LocalInertia`, `Momentum` subtype `AbstractStatistic` and extend the shared statistic generics via `import REM: compute, name` (these are Networks.jl's `compute`/`name`, which REM re-exports — one generic for the whole ecosystem, so ERGM and REM co-load without the verbs colliding; see Networks.jl `src/statistics.jl`). Each implements BOTH interfaces: 5-arg `compute(stat, history::InteractionHistory, s, r, t)` (used by this package's full-risk-set estimators) and 4-arg `compute(stat, state::REM.EventNetworkState, s, r)` (so they work inside `REM.fit_rem`). The two agree exactly (tested). Momentum's `normalize` divides by the sender's own event count. Decay-weighted statistics are backed by streaming `(value, last_time)` accumulators that absorb each event once and decay lazily on read — never rescan the event history per evaluation.
2b. **Participation shifts and covariates** -- `PShift` implements the 13 Gibson (2003) participation-shift indicators (`pshift_types()`; R names like `"PSAB-BA"` accepted), including the null-actor (`receiver == 0`, group-directed) variants. `CovSnd`/`CovRec`/`CovInt` are the R relevent actor-covariate effects, indexed by actor ID and validated on access.
3. **Ordinal Butts-Park Model** -- `fit_obpm()`: exact multinomial partial likelihood over the FULL risk set (P(event) = exp(θ'x_case)/Σ_dyads exp(θ'x)), Newton-Raphson with step-halving, observed-information SEs. Complements REM.fit_rem (which samples the risk set). Tied times: `ties=:error` (default) | `:ordered` | `:breslow` | `:efron` — see below.
4. **Timing Models** -- `TimingModel` with three PH baselines (exponential, Weibull, Gompertz) for `hazard_rate()`/`survival_function()` (S = exp(−∫h) consistency tested). `fit_timing()` implements the exponential-baseline interval likelihood ℓ = Σ_m[log λ₀ + θ'x_case − λ₀Δt_m Σ exp(θ'x)] with joint Newton estimation of (log λ₀, θ); Weibull/Gompertz fitting raises an informative error instead of returning placeholders. The `t0` keyword is the observation onset (first waiting time Δt₁ = t₁ − t0; default 0 as in R relevent) and `t_end` is the observation **offset**: it appends the right-censored final interval [t_M, t_end] (exposure, no event term; `case_idx == 0` inside `_risk_set_stats`). `rem.dyad` ALWAYS has that interval — its last edgelist row is the termination time and any event on it is ignored — so `t_end` is required to reproduce it; without it λ₀ is biased upward because the eventless tail is dropped (0.047 in log λ₀ on the golden sequence).
4b. **Standardized entry point** -- `fit_relevent(events, statistics, n_actors; ordinal=true, kwargs...)` dispatches to `fit_obpm` (ordinal) or `fit_timing` (interval, forwarding `t0`); `rem_dyad` is the R-faithful alias.
5. **Cumulative Network State** -- `CumulativeState{T}` tracks a decaying adjacency matrix and degree vectors, updated via `update_state!`.

## The risk sets stream: `cache=` (Relevent#2)

The full risk set means an `n(n−1) × p` design matrix per event, and the package used to materialize every one of them: **`O(E · n² · p)`** doubles, 906 MB at `(n, E, p) = (100, 2000, 6)`. That is what capped exact full-risk-set estimation — not the arithmetic. The machinery is split in two:

- **`_RiskSetPlan`** — the `O(E + n²)` skeleton, always materialized: `case_idx` (which dyad acted; `0` on the right-censored tail), `waiting`, `read_time` (the time interval `m`'s statistics are read at — the *block's* time under a freeze policy), and `absorb[m]` (which events the history absorbs *after* interval `m` is emitted; empty for the non-last members of a frozen tie block, the whole block for its last). The Efron denominator weights live here too, **sparsely**: every tied case in a block takes the same weight `1 − (j−1)/d`, so a list of dyad indices plus one scalar per interval replaces the dense `n(n−1)`-vector per event the old code stored.
- **`_RiskSets`** — the design matrices, under one of three policies, iterated by `each_interval(f, rs)` calling `f(m, Xm, twm)`:
  - **`:all`** (`_CachedRiskSets`) — every matrix materialized once, sharing one matrix across a frozen tie block. Fastest, `O(E · n² · p)`.
  - **`:chunked`** (`_StreamedRiskSets`) — a bounded cache of `chunk` matrices, refilled by replaying the interaction history from the start on every pass. `O(chunk · n² · p)`.
  - **`:none`** — `:chunked` with `chunk = 1`.

  `_resolve_cache(plan; cache, chunk, cache_bytes)` returns `(mode, k)` — the policy and the number of matrices alive — so the footprint a fit will pay is computable **without allocating any of it**. `cache=:auto` (the default) is `:all` under `cache_bytes` (256 MiB) and `:chunked` above it; a `chunk` covering every interval collapses to `:all` rather than recomputing for nothing.

**The fits are bit-identical across all three — `==`, not `≈`, and the tests assert `==`.** A design matrix is a deterministic function of (pre-interval history, read time); every policy visits the intervals in the same order and replays the same events into the same history in the same order, so every accumulation into `ll`/`grad`/`hess` is summed identically. This holds under the tie corrections too, where `:all` *shares* one matrix across a frozen block and the streamed policies *recompute* it — same frozen history, same read time, same numbers. The golden fixture is re-asserted under each policy; if a cache mode moves a coefficient, the streaming is wrong.

Measured (`fit_obpm`, p = 6, an event stream with real inertia and reciprocity):

| (n, E) | `:all` | `:chunked` (32) | `:none` |
|---|---|---|---|
| (40, 800) | 57.1 MB / 0.56 s | 2.3 MB / 1.78 s | 0.1 MB / 1.53 s |
| (60, 1500) | 243.1 MB / 2.31 s | 5.2 MB / 6.83 s | 0.2 MB / 6.28 s |
| (100, 2000) | 906.4 MB / 11.0 s | 14.5 MB / 32.1 s | **0.5 MB** / 34.9 s |

1800x less design memory for ~3x the time. Note `:none` is not slower than `:chunked` — the recomputation cost is the same (both replay the history once per pass) and the single hot buffer has better cache locality, so `chunk` buys **memory headroom, not speed**. The streamed policies are not allocation-free: replaying the history allocates `O(E)`. That is the history, not the risk sets, and it is an order of magnitude below the design cache it avoids (asserted).

## Derivative loops allocate O(p²) (review finding 15)

`_obpm_derivatives(rs)` and `_timing_derivatives(rs)` are **named** functions returning the `θ -> (ll, grad, hess)` closures for `_newton`, not closures buried inside the fitters — so the `@allocated` regression test measures the code that runs, not a copy of it. They allocate the `p` gradient and `p×p` Hessian they return and **nothing else**: 192 bytes per evaluation, independent of `E` and of the risk-set size. `fit_timing` previously computed `Sxx = Xm' * (w .* Xm)` per interval, allocating a full weighted copy of the risk-set design matrix *and* a fresh `p×p` outer product on every event of every Newton evaluation. Do not reintroduce a temporary in these loops; the test will catch it, and it is there to.

## Tied event times: the two likelihoods take DIFFERENT policies (Relevent#1, finding 12)

One vocabulary (**`Networks.TIE_POLICIES`**, defined once in Networks.jl and shared with REM.jl), one keyword (`ties=`), one meaning per symbol — but `fit_obpm` and `fit_timing` claim different things about time, so they accept different subsets of it, and each **refuses** the rest with an explanation rather than silently accepting it (`Networks.check_tie_policy`). Both default to **`:error`**.

| | `:error` | `:ordered` | `:breslow` | `:efron` | `:batch` |
|---|---|---|---|---|---|
| `fit_obpm` (order) | default | ✓ | ✓ | ✓ | **refused** → use `:breslow` |
| `fit_timing` (exact time) | default | ✓ | **refused** | **refused** | ✓ |

- **`fit_obpm`** is a likelihood over the *order* of events, i.e. a multinomial partial likelihood over the full risk set. A tie says the order is unobserved, so the classical Cox corrections apply verbatim: `:breslow` freezes the interaction history across the tie block (the tied events cannot enter one another's statistics; the block is absorbed as a whole afterwards) and gives each event the same denominator; `:efron` adds the `1 − (j−1)/d` denominator weights on the tied cases (`_risk_set_stats` returns them as `tw`, `nothing` when every weight is 1, which keeps the unweighted inner loop bit-for-bit as it was). `:efron` requires the tied cases to be **distinct dyads** — a dyad competing with itself has no fractional weight. `:batch` is refused *because with the history frozen it IS Breslow*.
- **`fit_timing`** is an **exact-time** likelihood, and this is the sharp case in the ecosystem: under the continuous-time process it fits, a tie has probability **zero**. It is a violated assumption, not an ambiguous order. Breslow and Efron are corrections to a *partial* likelihood — the baseline hazard profiled out, only the order surviving — and there is no such denominator here, so they are **not "a bit wrong", they are undefined**: refused, with a pointer to `fit_obpm(...; ties=:efron)` if the order is what matters. `:batch` reads the tie as one simultaneous batch: statistics frozen across the block, one exposure interval consumed. `:ordered` (the legacy behaviour) instead lets each tied event after the first enter with a **zero-length waiting interval** — an event term with no exposure — while still updating the next event's statistics, i.e. it claims one event caused another in no time at all. Because the tied intervals have Δt = 0, `:ordered` and `:batch` can only differ through the tied events' *own* statistics — which is exactly where the invented order does its damage (a `PShift(:AB_BA)` fires on an event that did not precede it).

`tie_method(fit)` reports what actually happened: `:none` when the data had no ties, never `:error` (which throws instead of returning). `is_exact(fit)` is `false` as soon as ties were present, under every policy. On tie-free data every policy gives the identical fit — pinned by the tests, and the sharpest check available that a correction is a correction.

## Key Dependencies

- **REM.jl** (local, `../REM.jl`) -- Core relational event modeling; provides `Event{T}`, `AbstractStatistic`, `EventSequence`, `fit_rem`
- **LinearAlgebra**, **Statistics** (stdlib) -- Newton solver, dot products, summaries

## Golden fixtures (R)

`test/fixtures/relevent_rem_dyad.toml` freezes a real `relevent::rem.dyad` fit (relevent 1.2.1, R 4.6.1) of BOTH likelihoods on one simulated 8-actor / 100-event sequence; `test/fixtures/r/relevent_rem_dyad.R` regenerates it (`Rscript test/fixtures/r/relevent_rem_dyad.R > test/fixtures/relevent_rem_dyad.toml`). Loaded with Networks.jl's `load_golden`, which refuses a fixture without provenance.

Two model-specification facts the fixture exists to pin, both invisible from the Julia side:
- **`rem.dyad`'s temporal likelihood has no intercept** — the dyad hazard is `exp(θ'x)`, full stop. `fit_timing` always fits `log λ₀`. The models are made identical by giving R a constant `CovSnd` column of 1s, whose coefficient then *is* `log λ₀`.
- **`rem.dyad` reads the last edgelist row as the observation end** and ignores any event on it, so its temporal likelihood always carries a right-censored final interval. That is what `t_end` reproduces.

Tolerance **1e-6**, and it is not a Monte-Carlo allowance: both likelihoods are exact, so the only admissible discrepancy is the reference's BFGS termination slack (R's `optim` is tightened to `reltol=1e-15` in the script; the two log-likelihoods then agree to <1e-9 and the coefficients to ≤2e-7). A failure here means the LIKELIHOOD differs — do not widen it.

## Conventions

- All statistics use half-life parameterization for exponential decay (`decay = log(2) / halflife`).
- Statistic types are immutable structs subtyping `AbstractStatistic` with BOTH compute signatures (history 5-arg and REM.EventNetworkState 4-arg) plus `name(stat)`.
- Mutable state types (`CumulativeState`) use `update_!` naming convention (bang suffix).
- The package uses parametric types (e.g., `InteractionHistory{T}`, `CumulativeState{T}`) where `T` is the time type, defaulting to `Float64`.
- Actor IDs are 1-based `Int` values.
- Julia 1.12+ is required.
