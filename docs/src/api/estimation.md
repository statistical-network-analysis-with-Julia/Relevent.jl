# Estimation API Reference

This page documents the model fitting functions in Relevent.jl.

## Standardized Entry Point

`fit_relevent` dispatches between the ordinal and interval-timing
likelihoods. `rem_dyad === fit_relevent` is a Julia alias; R arguments and
Bayesian fitting defaults are not emulated.

```@docs
fit_relevent
rem_dyad
```

## Ordinal Models

### fit_obpm

```@docs
fit_obpm
```

## Timing Models

### fit_timing

```@docs
fit_timing
Relevent.is_interval_constant
```

### hazard_rate

```@docs
hazard_rate
```

### survival_function

```@docs
survival_function
```

## Result Accessors

Relevent.jl extends the StatsAPI generics for both result types, so a fitted
model behaves like any other Julia statistical model.

```@docs
coef(::OrdinalBPMResult)
coef(::TimingModelResult)
stderror(::OrdinalBPMResult)
stderror(::TimingModelResult)
```

Timing accessors consistently include the estimated log baseline as the first
coefficient: `coef(fit) == [fit.log_baseline; fit.coefficients]`,
`stderror(fit) == [fit.log_baseline_se; fit.std_errors]`. `vcov`, `confint`,
`coefnames`, `coeftable`, and parameter counts use this same order. The
`.coefficients` and `.std_errors` fields remain effect-only; pass those effect
coefficients and `.baseline_params` to the hazard/survival helpers.

```@docs
coefnames
vcov
loglikelihood
nobs
dof
aic
aicc
bic
confint
coeftable
```

Both fitters throw `ArgumentError` when an exactly verified separating direction
proves that no finite MLE exists. Their candidate search is not exhaustive;
a returned result is not a general certificate of existence.

Both result types retain `iterations`, `converged`, and the full covariance.
Non-convergence emits a warning; singular information gives NaN uncertainty.
AIC/BIC comparisons require the same data, observation window, and likelihood.
`nobs` counts observed events, excluding a right-censored tail.
