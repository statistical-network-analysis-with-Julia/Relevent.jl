# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Relevent.jl is a Julia port of the R `relevent` package from the StatNet collection. It extends REM.jl with advanced relational event model features including interaction history tracking, ordinal Butts-Park Models (BPM), parametric timing models, and decay-weighted network statistics.

## Development Commands

- **Run tests:** `julia --project -e 'using Pkg; Pkg.test()'`
- **Build docs:** `julia --project=docs docs/make.jl`
- **Load in REPL:** `julia --project` then `using Relevent`
- **Install local deps:** `julia --project -e 'using Pkg; Pkg.instantiate()'`

Note: This package depends on local (non-registered) packages `Network` and `REM` via relative paths (`../Network`, `../REM`). These sibling directories must be present.

## Architecture

The entire package lives in a single file: `src/Relevent.jl`. It is organized into four sections:

1. **Interaction History Tracking** -- `InteractionHistory{T}` struct that records per-dyad event times, sender/receiver histories, and event counts. Mutated via `update_history!`.
2. **Advanced REM Statistics** -- Statistic types (`PriorInteraction`, `SendingCapacity`, `ReceivingCapacity`, `LocalInertia`, `Momentum`) that subtype `AbstractStatistic` (from REM.jl) and implement `compute()` with half-life exponential decay.
3. **Ordinal Butts-Park Model** -- `OrdinalBPM` and `fit_obpm()` for modeling event ordering without exact timestamps. Estimation is currently a placeholder (gradient descent stub).
4. **Timing Models** -- `TimingModel` with three baseline hazards (exponential, Weibull, Gompertz). Provides `hazard_rate()`, `survival_function()`, and `fit_timing()`. Fitting is currently a placeholder returning initial parameter estimates.
5. **Cumulative Network State** -- `CumulativeState{T}` tracks a decaying adjacency matrix and degree vectors, updated via `update_state!`.

## Key Dependencies

- **REM.jl** (local, `../REM`) -- Core relational event modeling; provides `Event{T}`, `AbstractStatistic`, `EventSequence`, `fit_rem`
- **Network** (local, `../Network`) -- Network data structures
- **Optim.jl** -- Numerical optimization
- **Distributions.jl**, **StatsBase.jl**, **DataFrames.jl** -- Statistical computing
- **LinearAlgebra** (stdlib) -- `dot` product in hazard/survival computations

## Conventions

- All statistics use half-life parameterization for exponential decay (`decay = log(2) / halflife`).
- Statistic types are immutable structs subtyping `AbstractStatistic` with a `compute()` method signature: `compute(stat, history, sender, receiver, current_time)`.
- Mutable state types (`CumulativeState`) use `update_!` naming convention (bang suffix).
- The package uses parametric types (e.g., `InteractionHistory{T}`, `CumulativeState{T}`) where `T` is the time type, defaulting to `Float64`.
- Actor IDs are 1-based `Int` values.
- Julia 1.9+ is required.
