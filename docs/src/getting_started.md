# Getting Started

Use `fit_obpm` for event choice and `fit_timing` for the supported duration model.
This tutorial first fits bundled radio calls on their ordinal clock, then uses
a small duration example to make the timing model's observation window explicit.

## Prepare the environment

```@raw html
<p>Use Julia <strong>1.12+</strong> and the <a href="/getting-started/">workspace installation guide</a> for this <strong>unreleased 0.2.0 development version</strong>. From the prepared workspace:</p>
```

```bash
julia --project=Relevent.jl
```

```@raw html
<p>Relevent uses <a href="/REM.jl/dev/">REM.jl</a> event types and the bundled data in <a href="/Networks.jl/dev/">Networks.jl</a>. It does not require NetworkDynamic.jl for these examples.</p>
```

## Fit event choice on the bundled radio calls

```julia
using Networks, REM, Relevent

wtc = load_dataset(:wtc_police_calls)
calls = [Event(row[2], row[3], Float64(row[1])) for row in eachrow(wtc.events)]
n = wtc.n_actors
stats = [CovInt(Float64.(wtc.is_icr))]
ordinal = fit_obpm(calls, stats, n)
@assert ordinal.converged
coeftable(ordinal)
Networks.fit_metadata(ordinal)
```

The 481 calls are ordered events among 37 eligible officers, including two who
never appear in the log. The fitter compares each call with all `n*(n-1)`
directed dyads. `CovInt` adds sender and receiver values of the coordinator-role
indicator; its coefficient is about 2.10 here. The dataset's event numbers are
not elapsed times and cannot support a duration analysis.

`fit_relevent(calls, stats, n)` invokes the same ordinal path. `rem_dyad` is its
alias, with Julia arguments and MLE behavior; it does not reproduce all R
`rem.dyad` arguments or Bayesian defaults.

## Add history or participation shifts

```julia
history = InteractionHistory{Float64}()
for event in calls
    update_history!(history, event)
end
get_interaction_count(history, calls[1].sender, calls[1].receiver)

# Does the next event return the previous sender's call?
shift_model = fit_obpm(calls, [PShift(:AB_BA), CovInt(Float64.(wtc.is_icr))], n)
@assert shift_model.converged
coeftable(shift_model)
```

Fitters build the pre-event history internally. The manual history above is for
inspection; do not feed a history containing future events into a pre-event
statistic calculation. [Participation shifts and cumulative effects](guide/r_catalogue.md)
are suitable for an order-only clock. Finite-half-life statistics such as
`PriorInteraction(10.0)` are also available in ordinal fits when decay in the
chosen clock units is intended. There is no universally appropriate half-life.

## Fit durations on an elapsed-time clock

Here times are minutes after observation starts at zero. All six directed dyads
among three actors occur once; observation continues to minute 7 with no further
event. The balanced design makes the static sender effect zero and the per-dyad
baseline rate `1/7` per minute, providing a simple check of the interface.

```julia
events = [Event(1,2,1.), Event(2,3,2.), Event(3,1,3.),
          Event(1,3,4.), Event(2,1,5.), Event(3,2,6.)]
time_stats = [CovSnd([0., 1., 2.])]
timing = fit_timing(events, time_stats, 3; t0=0., t_end=7.)
@assert timing.converged
coeftable(timing)
timing.baseline_params
```

`t_end` contributes the event-free final interval to exposure; omitting it changes
the observation window. This example explains the parameterization, rather than
providing enough data for substantive inference.

Only the **exponential baseline can be fitted**. `TimingModel` also evaluates
Weibull/Gompertz hazard and survival functions for parameters supplied by the user;
`fit_timing` rejects those baselines.

Every timing statistic must be constant as time advances at fixed history.
Participation shifts, static covariates and cumulative catalogue effects meet
this condition. The five half-life statistics are admissible only at positive
`Inf`, meaning zero decay. Finite half-lives are refused because their hazard
requires integration over the interval. Changing to zero decay changes the model.
Custom effects must explicitly satisfy `Relevent.is_interval_constant`; see
[timing models](guide/timing.md) for the contract.

## Inspect results and model boundaries

```julia
coef(timing)
stderror(timing)
vcov(timing)
confint(timing)
Networks.fit_metadata(timing)
```

Timing StatsAPI vectors put **log baseline first**, followed by effect coefficients;
its covariance and coefficient table use that same order. The legacy
`timing.coefficients` and `timing.std_errors` fields contain only effects.
`timing.baseline_params` contains the positive baseline rate.

Both fitters throw `ArgumentError` when an exactly checked direction proves no
finite MLE exists. Ordinal verification compares each event with every
positive-weight competitor; timing verification holds event predictors fixed and
strictly decreases some positive exposure without increasing any. Censored tails
and zero-duration event constraints are included. The final Newton increment and
coefficient vector supply candidates: **this is not an exhaustive separation
search**, and a returned fit does not prove existence or uniqueness of its MLE.
Check convergence and uncertainty before inference; singular information produces
undefined standard errors.

Tied times default to refusal. Ordinal fits accept `:ordered`, `:breslow` and
`:efron`; timing fits accept `:ordered` and `:batch`. Each makes a different claim
about the unresolved ordering. The [timing guide](guide/timing.md) explains those
claims and their metadata.

## Choose the next step

- Use [history tracking](guide/history.md) and [statistics](guide/statistics.md)
  to inspect the quantities entering a model.
- Consult [R effect coverage](guide/r_catalogue.md) for the validated subset;
  catalogue design fixtures do not establish fitted parity for every combination.
- Use `cache=:all`, `:chunked` or `:none` to trade design storage for replay work;
  all enumerate the same full risk set. See the [estimation API](api/estimation.md).

```@raw html
<p>Use <a href="/REM.jl/dev/">REM.jl</a> for sampled controls or restricted/changing risk sets. Relevent fitting uses a fixed <code>1:n</code> actor universe with directed non-self dyads.</p>
```
