# Relevent.jl

```@raw html
<p>Analyze event choice and, for supported models, inter-event durations over a <strong>full directed risk set</strong>. Relevent.jl adds participation shifts, interaction history and a subset of R relevent&#x27;s effects to the event types in <a href="/REM.jl/dev/">REM.jl</a>.</p>
```

**Start here:** [Getting started](getting_started.md) ·
[History statistics](guide/statistics.md) · [Timing models](guide/timing.md) ·
[R effect coverage](guide/r_catalogue.md) · [Estimation API](api/estimation.md)

## Fit an ordinal model to radio calls

The bundled World Trade Center data record call order among 37 eligible officers.
`CovInt` adds the sender and receiver covariate values; here the covariate marks
institutionalized coordinator roles.

```@raw html
<p>Use Julia <strong>1.12+</strong> and the <a href="/getting-started/">workspace installation guide</a> for the current <strong>0.2.0 development version, unreleased</strong>. The examples assume that environment is already prepared.</p>
```

```julia
using Networks, REM, Relevent

wtc = load_dataset(:wtc_police_calls)
calls = [Event(row[2], row[3], Float64(row[1])) for row in eachrow(wtc.events)]
stats = [CovInt(Float64.(wtc.is_icr))]

# Retain all eligible officers, including those absent from the event log.
fit = fit_obpm(calls, stats, wtc.n_actors)
@assert fit.converged
coeftable(fit)
Networks.fit_metadata(fit)
```

This fits event choice conditional on history. The dataset's ordinal clock cannot
supply elapsed durations for a timing analysis.

## What can be fitted?

| Task | Supported path and boundary |
|---|---|
| Next-event choice | `fit_obpm`, or `fit_relevent(...; ordinal=true)`, evaluates every eligible directed dyad. Elapsed-time decay is available when the clock has meaningful units. |
| Inter-event durations | `fit_timing`, or `fit_relevent(...; ordinal=false)`, fits an exponential baseline with statistics constant between events. Specify the observation onset and any censored tail. |
| Other baseline shapes | `TimingModel` can evaluate Weibull/Gompertz hazards and survival for supplied parameters. **Their parameters cannot be fitted.** |
| History and R effects | Participation shifts, static covariates, cumulative catalogue effects and five half-life statistics; [coverage and conventions](guide/r_catalogue.md) are explicit. |

Finite-half-life history statistics are **refused by `fit_timing`** because the
changing hazard requires exposure integration. Positive infinite half-life is
zero decay and changes the model. Ordinal fitting still supports finite decay.

Both fitters reject an exactly verified separating direction proving that no
finite maximum-likelihood estimate exists. The search checks the final Newton
increment and coefficient vector; **it is not an exhaustive separation test**.
Check convergence, uncertainty and the scientific specification for every returned
fit. Tied times default to refusal; supported corrections are explained in the
[timing guide](guide/timing.md).

```@raw html
<p>Fitting currently assumes a fixed <code>1:n</code> actor universe with directed non-self dyads. For sampled controls or restricted/changing risk sets, use <a href="/REM.jl/dev/">REM.jl</a>. The R-compatible names do not imply full R feature or argument compatibility; Bayesian estimation and R&#x27;s simulation/GOF suite are not provided.</p>
```

Package citation: [CITATION.bib](https://github.com/statistical-network-analysis-with-Julia/Relevent.jl/blob/main/CITATION.bib).

## Module

```@docs
Relevent
```
