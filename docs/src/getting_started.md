# Getting Started

This tutorial walks through common use cases for Relevent.jl, from tracking interaction histories to fitting ordinal and timing models.

## Installation

Install Relevent.jl from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Statistical-network-analysis-with-Julia/Relevent.jl")
```

## Basic Workflow

The typical Relevent.jl workflow consists of four steps:

1. **Create events** - Prepare your relational event data
2. **Track history** - Build interaction history for advanced statistics
3. **Define statistics** - Choose advanced statistics with decay parameters
4. **Fit models** - Use ordinal BPM or timing models as appropriate

## Step 1: Create Event Data

Relevent.jl works with the `Event` type from REM.jl:

```julia
using REM
using Relevent

# Create events: sender, receiver, time
events = [
    Event(1, 2, 1.0),   # Alice -> Bob at t=1
    Event(2, 1, 2.0),   # Bob -> Alice at t=2
    Event(1, 3, 3.0),   # Alice -> Carol at t=3
    Event(3, 2, 4.0),   # Carol -> Bob at t=4
    Event(2, 3, 5.0),   # Bob -> Carol at t=5
    Event(1, 2, 6.0),   # Alice -> Bob at t=6 (repeat)
    Event(3, 1, 7.0),   # Carol -> Alice at t=7
    Event(2, 1, 8.0),   # Bob -> Alice at t=8
]
```

## Step 2: Track Interaction History

The `InteractionHistory` type provides detailed tracking of past interactions:

```julia
# Create a history tracker
history = InteractionHistory{Float64}()

# Process events
for event in events
    update_history!(history, event)
end

# Query the history
println("1->2 count: ", get_interaction_count(history, 1, 2))    # 2
println("2->1 count: ", get_interaction_count(history, 2, 1))    # 2
println("1->3 count: ", get_interaction_count(history, 1, 3))    # 1

# Last interaction time
println("Last 1->2: ", get_last_interaction(history, 1, 2))      # 6.0
println("Last 3->2: ", get_last_interaction(history, 3, 2))      # 4.0
println("Last 2->3: ", get_last_interaction(history, 2, 3))      # 5.0
```

### Accessing Detailed History

```julia
# All receivers of actor 1 (ordered by time)
history.sender_history[1]    # [2, 3, 2]

# All senders to actor 2 (ordered by time)
history.receiver_history[2]  # [1, 3, 1]

# Event times for specific dyad
history.pair_history[(1, 2)]  # [1.0, 6.0]

# Event counts per dyad
history.event_counts[(1, 2)]  # 2
```

## Step 3: Define Advanced Statistics

Relevent.jl provides statistics that use the full interaction history with half-life decay:

```julia
# Prior interaction with 10-unit halflife
prior = PriorInteraction(10.0; direction=:both)

# Local inertia: tendency for repeat interactions
inertia = LocalInertia(10.0)

# Sender activity momentum
momentum = Momentum(10.0)

# Sending and receiving capacity
send_cap = SendingCapacity(10.0)
recv_cap = ReceivingCapacity(10.0)
```

### Exploring Available Statistics

| Category | Statistics | Description |
|----------|-----------|-------------|
| **Prior Contact** | `PriorInteraction` | Decayed count of prior interactions |
| **Inertia** | `LocalInertia` | Recency of last same-dyad interaction |
| **Capacity** | `SendingCapacity`, `ReceivingCapacity` | Actor-level activity/popularity |
| **Momentum** | `Momentum` | Overall sender activity momentum |

### Computing Statistics Manually

```julia
# Compute a statistic for a potential event
# PriorInteraction computes using the history
value = compute(
    PriorInteraction(10.0; direction=:outgoing),
    history,
    1,        # sender
    2,        # receiver
    9.0       # current time
)
println("Prior interaction value: ", value)
```

### Choosing Halflife

The halflife parameter controls how quickly past events lose influence:

| Halflife | Use Case |
|----------|----------|
| 1-5 | Very fast decay, only recent events matter |
| 10-50 | Moderate decay, reasonable for most applications |
| 100+ | Slow decay, long memory |
| Inf | No decay, all events equally weighted |

## Step 4a: Fit a Standard REM

Use Relevent.jl statistics with REM.jl's fitting framework:

```julia
# Create event sequence
seq = EventSequence(events)

# Define model with Relevent statistics
stats = [
    LocalInertia(10.0),
    PriorInteraction(10.0; direction=:outgoing),
    SendingCapacity(10.0),
    ReceivingCapacity(10.0),
]

# Fit REM
result = fit_rem(seq, stats; n_controls=50, seed=42)
println(result)
```

## Step 4b: Fit an Ordinal BPM

When only event ordering is known (not exact times):

```julia
# Define statistics
stats = [
    LocalInertia(10.0),
    SendingCapacity(10.0),
]

n_actors = 3

# Fit ordinal Butts-Park model
result = fit_obpm(events, stats, n_actors)
println(result)

# Access results
println("Coefficients: ", result.coefficients)
println("Log-likelihood: ", result.loglik)
println("Converged: ", result.converged)
```

### When to Use Ordinal vs. Timed Models

| Data Type | Model | Use Case |
|-----------|-------|----------|
| Exact timestamps | `fit_rem` | Email logs, chat messages |
| Only ordering known | `fit_obpm` | Survey responses, narrative data |
| Inter-event durations | `fit_timing` | When timing itself is of interest |

## Step 4c: Fit a Timing Model

Model inter-event time distributions:

```julia
# Define statistics affecting the hazard rate
stats = [
    LocalInertia(10.0),
    SendingCapacity(10.0),
]

# Fit with exponential baseline hazard
result_exp = fit_timing(events, stats, n_actors; baseline=:exponential)
println(result_exp)

# Fit with Weibull baseline hazard
result_weib = fit_timing(events, stats, n_actors; baseline=:weibull)
println(result_weib)

# Fit with Gompertz baseline hazard
result_gomp = fit_timing(events, stats, n_actors; baseline=:gompertz)
println(result_gomp)
```

### Baseline Hazard Functions

| Baseline | Hazard Shape | Parameters | Best For |
|----------|-------------|------------|----------|
| Exponential | Constant | rate $\lambda$ | Memoryless processes |
| Weibull | Monotone increasing/decreasing | scale $\lambda$, shape $k$ | Aging effects |
| Gompertz | Exponentially increasing | $a$, $b$ | Accelerating processes |

## Step 5: Interpret Results

### Ordinal BPM Results

```julia
println(result)

# Ordinal Butts-Park Model Results
# ================================
# N actors: 3
# N events: 8
# Log-likelihood: -12.3456
# Converged: true
#
# Coefficients:
#   LocalInertia               0.4523 (SE: 0.1234)
#   SendingCapacity             0.0987 (SE: 0.0567)
```

### Timing Model Results

```julia
println(result_exp)

# Timing Model Results
# ====================
# Baseline: exponential
# Baseline params: [0.125]
# Log-likelihood: -45.6789
# Converged: true
```

### Computing Hazard and Survival

```julia
model = TimingModel(stats; baseline=:weibull)
coef = result_weib.coefficients
baseline_params = result_weib.baseline_params
x = [0.5, 0.3]  # Statistic values

# Hazard rate at time t
h = hazard_rate(model, coef, baseline_params, 5.0, x)

# Survival probability at time t
S = survival_function(model, coef, baseline_params, 5.0, x)
```

## Complete Example: Email Communication

```julia
using REM
using Relevent

# Email events with timestamps (hours)
events = [
    Event(1, 2, 0.0),   Event(2, 1, 0.5),   Event(1, 3, 1.0),
    Event(3, 2, 1.5),   Event(2, 3, 2.0),   Event(1, 2, 3.0),
    Event(3, 1, 4.0),   Event(2, 1, 5.0),   Event(1, 3, 6.0),
    Event(2, 3, 7.0),   Event(3, 2, 8.0),   Event(1, 2, 9.0),
]

# Track history
history = InteractionHistory{Float64}()
for e in events
    update_history!(history, e)
end

# Statistics with 4-hour halflife
stats = [
    LocalInertia(4.0),
    PriorInteraction(4.0; direction=:both),
    SendingCapacity(4.0),
    ReceivingCapacity(4.0),
]

# Standard REM
seq = EventSequence(events)
result = fit_rem(seq, stats; n_controls=20, seed=42)
println(result)

# Positive LocalInertia -> tendency to reply to same person
# Positive PriorInteraction -> prior contact increases future contact
```

## Cumulative Network State

For tracking decaying network state:

```julia
# Create state tracker with 10-unit halflife
state = CumulativeState{Float64}(3; halflife=10.0)

# Update with events
for event in events
    update_state!(state, event)
end

# Query state
println("Actor 1 out-degree: ", get_outdegree_history(state, 1))
println("Actor 2 in-degree: ", get_indegree_history(state, 2))

# View decayed adjacency matrix
println("Adjacency matrix:\n", state.adj_matrix)
```

## Best Practices

1. **Choose appropriate halflife**: Match the temporal scale of your data
2. **Start with basic statistics**: LocalInertia and PriorInteraction are usually informative
3. **Check for sufficient events**: Need at least 10 events per parameter
4. **Verify convergence**: Check `result.converged == true`
5. **Compare models**: Try different baseline hazards for timing models
6. **Use ordinal BPM when appropriate**: When exact timing is unreliable

## Next Steps

- Learn about [Interaction History](guide/history.md) tracking in detail
- Explore [Advanced Statistics](guide/statistics.md) available
- Understand [Timing Models](guide/timing.md) for duration analysis
