# Relevent.jl

*Additional Relational Event Model Features for Julia*

A Julia package providing advanced relational event model features including interaction history tracking, ordinal timing models, and specialized statistics.

## Overview

Relevent.jl extends REM.jl with additional capabilities for analyzing relational event sequences. It provides detailed interaction history tracking, advanced statistics with half-life decay, ordinal event models (Butts-Park Model), and parametric timing models for inter-event durations.

Relevent.jl is a port of the R [relevent](https://github.com/statnet/relevent) package from the StatNet collection.

### What Does Relevent.jl Add?

While REM.jl provides the core relational event modeling framework, Relevent.jl adds:

```text
REM.jl:      Events -> Statistics -> Estimation
Relevent.jl: Events -> History Tracking -> Advanced Statistics -> Timing Models
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **InteractionHistory** | Detailed tracking of all past interactions per dyad |
| **PriorInteraction** | Decayed count of prior interactions between actors |
| **LocalInertia** | Tendency for repeat interactions on the same dyad |
| **OrdinalBPM** | Model event ordering without exact timing information |
| **TimingModel** | Parametric model for inter-event time distributions |

### Applications

Relevent.jl is designed for:

- **Communication analysis**: Tracking email, message, and call patterns over time
- **Organizational studies**: Modeling collaboration dynamics with detailed history
- **Survey data**: Analyzing event sequences where only ordering is known
- **Duration analysis**: Modeling the timing of interactions, not just their occurrence
- **Longitudinal studies**: Understanding how interaction patterns evolve

## Features

- **Interaction history**: Track complete interaction history with per-dyad detail
- **Advanced statistics**: Prior interaction, sending/receiving capacity, momentum, local inertia
- **Half-life decay**: All statistics support configurable exponential decay
- **Ordinal BPM**: Butts-Park Model for ordinal (rank-order) event data
- **Timing models**: Exponential, Weibull, and Gompertz baseline hazards
- **Cumulative state**: Decaying network state for efficient computation

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Statistical-network-analysis-with-Julia/Relevent.jl")
```

Or for development:

```julia
using Pkg
Pkg.develop(path="/path/to/Relevent.jl")
```

## Quick Start

```julia
using REM
using Relevent

# Create events
events = [
    Event(1, 2, 1.0),
    Event(2, 1, 2.0),
    Event(1, 3, 3.0),
    Event(3, 2, 4.0),
    Event(1, 2, 5.0),
    Event(2, 3, 6.0),
]

# Track interaction history
history = InteractionHistory{Float64}()
for event in events
    update_history!(history, event)
end

# Query history
count = get_interaction_count(history, 1, 2)        # 2
last = get_last_interaction(history, 1, 2)           # 5.0

# Advanced statistics with decay
stats = [
    LocalInertia(10.0),           # Repeat interaction tendency
    PriorInteraction(10.0),       # Prior contact with decay
    SendingCapacity(10.0),        # Sender activity
    ReceivingCapacity(10.0),      # Receiver popularity
]
```

## Choosing Statistics

| Use Case | Recommended Statistics |
|----------|----------------------|
| Repeat interaction patterns | [`LocalInertia`](@ref) |
| Prior contact effects | [`PriorInteraction`](@ref) |
| Sender activity levels | [`SendingCapacity`](@ref) |
| Receiver popularity | [`ReceivingCapacity`](@ref) |
| Overall activity momentum | [`Momentum`](@ref) |
| Ordinal event data | [`OrdinalBPM`](@ref) with `fit_obpm` |
| Inter-event timing | [`TimingModel`](@ref) with `fit_timing` |

## Documentation

```@contents
Pages = [
    "getting_started.md",
    "guide/history.md",
    "guide/statistics.md",
    "guide/timing.md",
    "api/types.md",
    "api/statistics.md",
    "api/estimation.md",
]
Depth = 2
```

## Theoretical Background

### The Butts-Park Model

The ordinal Butts-Park Model (BPM) models the relative ordering of events rather than their exact timing. For a sequence of events $e_1, e_2, \ldots, e_m$, the likelihood is:

$$L(\boldsymbol{\theta}) = \prod_{k=1}^{m} \frac{\exp(\boldsymbol{\theta}^\top \mathbf{x}_k)}{\sum_{l \in \mathcal{R}_k} \exp(\boldsymbol{\theta}^\top \mathbf{x}_l)}$$

Where $\mathcal{R}_k$ is the set of potential events at step $k$ and $\mathbf{x}_k$ are statistics for the observed event.

### Parametric Timing Models

Timing models extend REMs by explicitly modeling inter-event durations using parametric baseline hazards:

- **Exponential**: Constant hazard $h(t) = \lambda$
- **Weibull**: $h(t) = (k/\lambda)(t/\lambda)^{k-1}$ -- increasing or decreasing hazard
- **Gompertz**: $h(t) = a \cdot \exp(b \cdot t)$ -- exponentially increasing hazard

## References

1. Butts, C.T. (2008). A relational event framework for social action. *Sociological Methodology*, 38(1), 155-200.

2. Butts, C.T., Marcum, C.S. (2017). A relational event approach to modeling behavioral dynamics. In *Group Processes*, Springer.

3. Lerner, J., Lomi, A. (2020). Reliability of relational event model estimates under sampling. *Network Science*, 8(1), 97-135.

4. Stadtfeld, C., Block, P. (2017). Interactions, actors, and time: Dynamic network actor models for relational events. *Sociological Science*, 4, 318-352.
