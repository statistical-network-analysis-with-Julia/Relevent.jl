# Estimation API Reference

This page documents the model fitting functions in Relevent.jl.

## Standardized Entry Point

`fit_relevent` dispatches between the ordinal and interval-timing
likelihoods, exactly like `relevent::rem.dyad`; `rem_dyad` is the
R-faithful alias.

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
