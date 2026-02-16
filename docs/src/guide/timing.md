# Timing Models

Relevent.jl provides parametric timing models for analyzing inter-event durations in relational event sequences. While standard REMs model which events occur, timing models additionally model when they occur.

## Overview

Timing models extend the relational event framework by explicitly modeling the time between events. They combine:

- A **baseline hazard** function describing the intrinsic rate of events
- **Covariates** (statistics) that modify the hazard rate

## The Timing Model Framework

The hazard rate for an event at time $t$ given covariates $\mathbf{x}$ is:

$$h(t | \mathbf{x}) = h_0(t) \cdot \exp(\boldsymbol{\beta}^\top \mathbf{x})$$

Where:

- $h_0(t)$ is the baseline hazard function
- $\boldsymbol{\beta}$ are coefficients to be estimated
- $\mathbf{x}$ are statistic values (from REM or Relevent statistics)

## Baseline Hazard Functions

### Exponential

Constant hazard rate -- events occur at a fixed rate:

$$h_0(t) = \lambda$$

```julia
model = TimingModel(stats; baseline=:exponential)
```

**Parameters**: rate $\lambda > 0$

**Use case**: Processes with no memory (Markov property). Inter-event times are exponentially distributed.

### Weibull

Monotonically increasing or decreasing hazard:

$$h_0(t) = \frac{k}{\lambda}\left(\frac{t}{\lambda}\right)^{k-1}$$

```julia
model = TimingModel(stats; baseline=:weibull)
```

**Parameters**: scale $\lambda > 0$, shape $k > 0$

- $k > 1$: Increasing hazard (events accelerate)
- $k = 1$: Constant hazard (reduces to exponential)
- $k < 1$: Decreasing hazard (events slow down)

**Use case**: Processes where the rate of events changes monotonically over time.

### Gompertz

Exponentially increasing hazard:

$$h_0(t) = a \cdot \exp(b \cdot t)$$

```julia
model = TimingModel(stats; baseline=:gompertz)
```

**Parameters**: $a > 0$, $b > 0$

**Use case**: Processes with exponentially accelerating event rates, such as contagion or cascade effects.

## Creating a Timing Model

```julia
using REM
using Relevent

# Define statistics
stats = [
    LocalInertia(10.0),
    SendingCapacity(10.0),
]

# Create model with specified baseline
model = TimingModel(stats; baseline=:exponential)
```

## Fitting a Timing Model

```julia
# Create event data
events = [
    Event(1, 2, 1.0),
    Event(2, 1, 3.0),
    Event(1, 3, 4.5),
    Event(3, 2, 7.0),
    Event(2, 3, 8.5),
    Event(1, 2, 10.0),
]

n_actors = 3

# Fit with exponential baseline
result = fit_timing(events, stats, n_actors; baseline=:exponential)
println(result)
```

### Fitting Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `baseline` | Hazard function type | `:exponential` |
| `maxiter` | Maximum iterations | `100` |

## Understanding Results

The `TimingModelResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `model` | `TimingModel` | The model specification |
| `coefficients` | `Vector{Float64}` | Estimated coefficients |
| `baseline_params` | `Vector{Float64}` | Baseline hazard parameters |
| `std_errors` | `Vector{Float64}` | Standard errors |
| `loglik` | `Float64` | Log-likelihood |
| `converged` | `Bool` | Convergence status |

### Displaying Results

```julia
println(result)

# Timing Model Results
# ====================
# Baseline: exponential
# Baseline params: [0.125]
# Log-likelihood: -45.6789
# Converged: true
```

## Computing Hazard and Survival

### Hazard Rate

The instantaneous event rate at time $t$:

```julia
h = hazard_rate(model, coef, baseline_params, t, x)
```

Where `x` is a vector of statistic values for the potential event.

### Survival Function

The probability that no event occurs before time $t$:

```julia
S = survival_function(model, coef, baseline_params, t, x)
```

### Example: Hazard Curves

```julia
model = TimingModel(stats; baseline=:weibull)
coef = [0.5, 0.3]
baseline_params = [2.0, 1.5]  # lambda=2, k=1.5
x = [0.5, 0.2]

# Compute hazard at multiple time points
times = 0.1:0.1:10.0
hazards = [hazard_rate(model, coef, baseline_params, t, x) for t in times]
survivals = [survival_function(model, coef, baseline_params, t, x) for t in times]

# Hazard increases over time (k=1.5 > 1)
# Survival decreases over time
```

## Ordinal Butts-Park Model

When exact event times are unknown but ordering is known, use the ordinal BPM:

```julia
# Define statistics
stats = [
    LocalInertia(10.0),
    SendingCapacity(10.0),
]

# Fit ordinal model
result = fit_obpm(events, stats, n_actors)
```

### When to Use Ordinal vs. Timing Models

| Situation | Model |
|-----------|-------|
| Exact timestamps available, want to model "who interacts" | Standard REM (`fit_rem`) |
| Exact timestamps available, want to model "when" as well | `fit_timing` |
| Only ordering known | `fit_obpm` |
| Timestamps unreliable | `fit_obpm` |

### Rank Events

Convert events to ordinal ranks:

```julia
ranks = rank_events(events)
# Returns [1, 2, 3, 4, 5, 6] if events are in time order
```

### OrdinalBPM Results

```julia
println(result)

# Ordinal Butts-Park Model Results
# ================================
# N actors: 3
# N events: 6
# Log-likelihood: -12.3456
# Converged: true
#
# Coefficients:
#   LocalInertia               0.4523 (SE: 0.1234)
#   SendingCapacity             0.0987 (SE: 0.0567)
```

## Comparing Models

### Different Baselines

```julia
for baseline in [:exponential, :weibull, :gompertz]
    result = fit_timing(events, stats, n_actors; baseline=baseline)
    println("$baseline: LL=$(round(result.loglik, digits=2))")
end
```

### Different Statistics

```julia
# Model 1: Basic effects
stats1 = [LocalInertia(10.0)]

# Model 2: Add capacity effects
stats2 = [LocalInertia(10.0), SendingCapacity(10.0), ReceivingCapacity(10.0)]

result1 = fit_timing(events, stats1, n_actors)
result2 = fit_timing(events, stats2, n_actors)

println("Model 1 LL: ", result1.loglik)
println("Model 2 LL: ", result2.loglik)
```

## Best Practices

1. **Choose baseline by domain knowledge**: Exponential for memoryless, Weibull for monotone trends
2. **Compare baselines**: Fit multiple and compare log-likelihoods
3. **Check convergence**: Verify `result.converged == true`
4. **Sufficient inter-event times**: Need enough events for reliable duration estimates
5. **Scale covariates**: Large covariate values can cause numerical issues
6. **Start simple**: Begin with exponential baseline before trying more complex forms
7. **Use ordinal BPM when timing is unreliable**: Survey data, reconstructed sequences
