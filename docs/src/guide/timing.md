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
- $\mathbf{x}$ are statistic values that stay constant between events

## Which statistics can be fitted?

The exponential timing likelihood integrates a constant hazard over each waiting
interval. `fit_timing` therefore requires
[`Relevent.is_interval_constant`](@ref): participation shifts, cumulative R
catalogue statistics, fixed effects and static `CovSnd`/`CovRec`/`CovInt`/`CovEvent`
are supported. These statistics can change after an event; they do not change
while waiting for the next event.

`PriorInteraction`, `SendingCapacity`, `ReceivingCapacity`, `LocalInertia` and
`Momentum` with a finite half-life change continuously between events. The fitter
rejects them because it does not integrate their varying hazard. Evaluating a
decayed statistic at the next event and multiplying that hazard by the elapsed
time is not exact exposure. These statistics remain usable with `fit_obpm` and
REM conditional estimation when their time scale is observed.

Positive `halflife=Inf` explicitly specifies zero decay and is supported for all
five types. This changes the model: for example, `PriorInteraction(Inf)` counts
all prior dyad events, and `LocalInertia(Inf)` indicates any previous dyad event.
Choose these semantics deliberately. Custom statistics default to unsupported;
author an `is_interval_constant` method only if the contract holds for every
admissible history. Hazard/survival helpers evaluate user-supplied fixed `x`;
they do not integrate an arbitrary time-varying statistic either.

## Baseline Hazard Functions

### Exponential

Constant hazard rate -- events occur at a fixed rate:

$$h_0(t) = \lambda$$

```julia
using REM, Relevent

stats = [PShift(:AB_BA), CovSnd([-1., 0., 1.])]
model = TimingModel(stats; baseline=:exponential)
```

**Parameters**: rate $\lambda > 0$

**Use case**: Conditional on the current event history, the next waiting time is exponential. Its rate can change after each event.

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

# Statistics change only when a new event enters the history
stats = [PShift(:AB_BA), CovSnd([-1., 0., 1.])]

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

### Separation and finite estimates

A small Newton objective change can also occur while coefficients run toward
infinity. Both `fit_obpm` and `fit_timing` now throw `ArgumentError` when an
exactly verified direction proves that no finite maximum-likelihood estimate
exists. For example, if every event is sent by the actor with the largest
sender covariate, increasing its coefficient can keep improving the ordinal
likelihood; the timing baseline can decrease simultaneously to preserve that
actor's rate. More iterations cannot create a finite optimum.

The guard tests directions suggested by the final Newton step and coefficients.
For ordinal fits, every positive-weight competitor must score no higher than
its case along the direction, with a strict contrast somewhere. For timing
fits, every event predictor must remain fixed, every positive-exposure predictor
must decrease or stay fixed, and some exposure must strictly decrease. Censored
tails count as exposure; events at zero waiting times still impose constraints.

Promising directions are checked with exact rational arithmetic on the stored
Float64 design, so tiny real overlap cannot be rounded into a certificate.
This is a sufficient-certificate search, **not an exhaustive separation test**.
A fit that returns still needs convergence, uncertainty and model checks;
`converged=true` alone does not establish that every boundary direction was
excluded. Revise separating effects rather than treating a very large estimate
as a finite maximum or simply increasing `maxiter`.

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

<!-- skip-check -->
```julia
h = hazard_rate(model, coef, baseline_params, t, x)
```

Where `x` is a vector of statistic values for the potential event.

### Survival Function

The probability that no event occurs before time $t$:

<!-- skip-check -->
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

Use the ordinal BPM for event choice conditional on the observed history.
Finite half-life statistics still require a meaningful elapsed-time scale:

```julia
# Elapsed-time decay is valid for conditional event choice when times are known.
ordinal_stats = [LocalInertia(10.0), SendingCapacity(10.0)]
result = fit_obpm(events, ordinal_stats, n_actors)
```

### When to Use Ordinal vs. Timing Models

| Situation | Model |
|-----------|-------|
| Exact timestamps available, want to model "who interacts" | Standard REM (`fit_rem`) |
| Exact timestamps available, want to model "when" as well | `fit_timing` |
| Only ordering known | `fit_obpm` |
| Timestamps unreliable | `fit_obpm` |

### Tied event times

Both fitters take `ties=` and both **default to `:error`**: tied timestamps are named
and refused, because both likelihoods claim something the tied data does not contain.
They claim *different* things, so they accept different policies (the shared
`Networks.TIE_POLICIES` vocabulary), and each refuses the rest with an explanation:

| `ties=` | `fit_obpm` (order) | `fit_timing` (exact time) |
|---|---|---|
| `:error` | default | default |
| `:ordered` | sequence order, no correction | tied events after the first get a zero-length waiting interval |
| `:breslow` | Breslow correction | **refused** — it corrects a *partial* likelihood |
| `:efron` | Efron correction (prefer it) | **refused** — same reason |
| `:batch` | **refused** — with the history frozen it *is* Breslow | one simultaneous batch: history frozen, one exposure interval |

```julia
fit_obpm(events, stats, n_actors; ties=:efron)     # order unknown within a tie
fit_timing(events, stats, n_actors; ties=:batch)   # a coarse clock, read as batches
```

For `fit_timing` this is the sharp case: under the continuous-time process it fits, two
events at one instant have probability **zero**. A tie is not an ambiguous ordering but
the model's own assumption failing, and `is_exact(fit)` turns `false` as soon as one is
present — under every policy. `tie_method(fit)` reports what actually happened (`:none`
when the data had no ties), and on tie-free data every policy gives the identical fit.

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
#                   Estimate  Std.Error  z value  Pr(>|z|)
# local_inertia       0.4523     0.1234   3.6653    0.0002 ***
# sending_capacity    0.0987     0.0567   1.7407    0.0817 .
# ---
# Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

## Comparing Models

### Different Baselines

Only the exponential baseline is fittable; requesting `:weibull` or
`:gompertz` from `fit_timing` raises an informative error (those baselines
are available for `hazard_rate`/`survival_function` with user-supplied
parameters):

```julia
result = fit_timing(events, stats, n_actors; baseline=:exponential)
println("exponential: LL=$(round(result.loglik, digits=2))")
```

### Different Statistics

```julia
# Model 1: Basic effects
stats1 = [PShift(:AB_BA)]

# Model 2: Add static actor covariates
stats2 = [PShift(:AB_BA), CovSnd([-1.,0.,1.]), CovRec([-1.,0.,1.])]

result1 = fit_timing(events, stats1, n_actors)
result2 = fit_timing(events, stats2, n_actors)

println("Model 1 LL: ", result1.loglik)
println("Model 2 LL: ", result2.loglik)
```

## Best Practices

1. **Use supported hazards**: Fitting requires an exponential baseline and interval-constant statistics
2. **Compare like likelihoods**: Use the same events, observation window and tie policy
3. **Check convergence**: Verify `result.converged == true`
4. **Sufficient inter-event times**: Need enough events for reliable duration estimates
5. **Scale covariates**: Large covariate values can cause numerical issues
6. **Start simple**: Add supported effects only when the data can identify them
7. **Use ordinal BPM when timing is unreliable**: Survey data, reconstructed sequences
