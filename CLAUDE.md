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

The entire package lives in a single file: `src/Relevent.jl`. It is organized into four sections:

1. **Interaction History Tracking** -- `InteractionHistory{T}` struct that records per-dyad event times, sender/receiver histories, and event counts. Mutated via `update_history!`.
2. **Advanced REM Statistics** -- `PriorInteraction`, `SendingCapacity`, `ReceivingCapacity`, `LocalInertia`, `Momentum` subtype `AbstractStatistic` and extend the REM generics via `import REM: compute, name`. Each implements BOTH interfaces: 5-arg `compute(stat, history::InteractionHistory, s, r, t)` (used by this package's full-risk-set estimators) and 4-arg `compute(stat, state::REM.EventNetworkState, s, r)` (so they work inside `REM.fit_rem`). The two agree exactly (tested). Momentum's `normalize` divides by the sender's own event count.
3. **Ordinal Butts-Park Model** -- `fit_obpm()`: exact multinomial partial likelihood over the FULL risk set (P(event) = exp(θ'x_case)/Σ_dyads exp(θ'x)), Newton-Raphson with step-halving, observed-information SEs. Complements REM.fit_rem (which samples the risk set). Ties warned, not corrected.
4. **Timing Models** -- `TimingModel` with three PH baselines (exponential, Weibull, Gompertz) for `hazard_rate()`/`survival_function()` (S = exp(−∫h) consistency tested). `fit_timing()` implements the exponential-baseline interval likelihood ℓ = Σ_m[log λ₀ + θ'x_case − λ₀Δt_m Σ exp(θ'x)] with joint Newton estimation of (log λ₀, θ); Weibull/Gompertz fitting raises an informative error instead of returning placeholders.
5. **Cumulative Network State** -- `CumulativeState{T}` tracks a decaying adjacency matrix and degree vectors, updated via `update_state!`.

## Key Dependencies

- **REM.jl** (local, `../REM.jl`) -- Core relational event modeling; provides `Event{T}`, `AbstractStatistic`, `EventSequence`, `fit_rem`
- **LinearAlgebra**, **Statistics** (stdlib) -- Newton solver, dot products, summaries

## Conventions

- All statistics use half-life parameterization for exponential decay (`decay = log(2) / halflife`).
- Statistic types are immutable structs subtyping `AbstractStatistic` with BOTH compute signatures (history 5-arg and REM.EventNetworkState 4-arg) plus `name(stat)`.
- Mutable state types (`CumulativeState`) use `update_!` naming convention (bang suffix).
- The package uses parametric types (e.g., `InteractionHistory{T}`, `CumulativeState{T}`) where `T` is the time type, defaulting to `Float64`.
- Actor IDs are 1-based `Int` values.
- Julia 1.12+ is required.
