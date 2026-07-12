"""
    Relevent.jl - Relational Event Models (Additional Features)

Extends REM.jl with features from R `relevent`'s `rem.dyad`:

- decay-weighted interaction-history statistics that plug directly into
  `REM.fit_rem` (they implement REM's 4-argument `compute` interface) and
  into this package's own full-risk-set estimators — computed via
  streaming accumulators (each event is absorbed once), not by rescanning
  the event history per evaluation;
- the 13 Gibson (2003) participation-shift effects (`PShift`, named as in
  relevent: `PSAB-BA`, `PSAB-BY`, ...) and the `CovSnd`/`CovRec`/`CovInt`
  covariate effects;
- **ordinal** estimation (`fit_obpm`): the exact multinomial partial
  likelihood over the full risk set (no case-control sampling);
- **interval-timing** estimation (`fit_timing`): the exponential-baseline
  proportional-hazards likelihood using the observed waiting times;
- the standardized entry point `fit_relevent` (with the R-faithful alias
  `rem_dyad`), dispatching between the two likelihoods via `ordinal=true/false`
  exactly like `relevent::rem.dyad`;
- baseline hazard/survival functions (exponential, Weibull, Gompertz) and
  cumulative decayed network state.

Port of the R relevent package from the StatNet collection.
"""
module Relevent

using LinearAlgebra
using REM
using Statistics
using StatsAPI

# ccdf keeps precision in the tail of the Wald p-values: 2*(1 - cdf(...))
# underflows to exactly 0 for |z| ≳ 8
using Distributions: Normal, ccdf
# Shared result-presentation infrastructure (Network.jl): the R-style
# coefficient table used by every model package in the ecosystem
using Network: print_coeftable

import REM: compute, name
# StatsAPI generics extended for this package's result types (shared with
# REM, StatsBase, GLM, ...)
import StatsAPI: coef, stderror

# StatsAPI accessors (methods for OrdinalBPMResult and TimingModelResult)
export coef, stderror

# Additional REM statistics
export InteractionHistory, PriorInteraction
export SendingCapacity, ReceivingCapacity
export LocalInertia, Momentum

# Participation shifts (Gibson 2003) and covariate effects, as in R relevent
export PShift, pshift_types
export CovSnd, CovRec, CovInt

# Standardized entry point (fit_relevent) and its R-faithful alias (rem.dyad)
export fit_relevent, rem_dyad

# Ordinal models
export OrdinalBPM, fit_obpm, OrdinalBPMResult
export rank_events

# Timing models
export TimingModel, fit_timing, TimingModelResult
export hazard_rate, survival_function

# History tracking
export update_history!
export get_interaction_count, get_last_interaction

# Network state
export CumulativeState, update_state!
export get_outdegree_history, get_indegree_history

# =============================================================================
# Interaction History Tracking
# =============================================================================

"""
    InteractionHistory{T}

Track detailed interaction history for advanced REM statistics.

# Fields
- `events::Vector{Event{T}}`: All observed events
- `sender_history::Dict{Int, Vector{Int}}`: Actor -> list of receivers (ordered by time)
- `receiver_history::Dict{Int, Vector{Int}}`: Actor -> list of senders (ordered by time)
- `pair_history::Dict{Tuple{Int,Int}, Vector{T}}`: (sender, receiver) -> list of event times
- `event_counts::Dict{Tuple{Int,Int}, Int}`: Count of events per dyad
"""
struct InteractionHistory{T}
    events::Vector{Event{T}}
    sender_history::Dict{Int, Vector{Int}}
    receiver_history::Dict{Int, Vector{Int}}
    pair_history::Dict{Tuple{Int,Int}, Vector{T}}
    event_counts::Dict{Tuple{Int,Int}, Int}

    function InteractionHistory{T}() where T
        new{T}(
            Event{T}[],
            Dict{Int, Vector{Int}}(),
            Dict{Int, Vector{Int}}(),
            Dict{Tuple{Int,Int}, Vector{T}}(),
            Dict{Tuple{Int,Int}, Int}()
        )
    end
end

InteractionHistory() = InteractionHistory{Float64}()

"""
    update_history!(history::InteractionHistory, event::Event)

Add an event to the interaction history.
"""
function update_history!(history::InteractionHistory{T}, event::Event{T}) where T
    push!(history.events, event)

    senders = get!(history.sender_history, event.sender, Int[])
    push!(senders, event.receiver)

    receivers = get!(history.receiver_history, event.receiver, Int[])
    push!(receivers, event.sender)

    pair = (event.sender, event.receiver)
    times = get!(history.pair_history, pair, T[])
    push!(times, event.time)

    history.event_counts[pair] = get(history.event_counts, pair, 0) + 1

    return history
end

"""
    get_interaction_count(history::InteractionHistory, sender, receiver) -> Int

Number of past events from sender to receiver.
"""
get_interaction_count(history::InteractionHistory, sender::Int, receiver::Int) =
    get(history.event_counts, (sender, receiver), 0)

"""
    get_last_interaction(history::InteractionHistory, sender, receiver)

Time of the most recent event from sender to receiver, or `nothing`.
"""
function get_last_interaction(history::InteractionHistory{T}, sender::Int, receiver::Int) where T
    times = get(history.pair_history, (sender, receiver), T[])
    return isempty(times) ? nothing : times[end]
end

# =============================================================================
# Streaming decayed accumulators
# =============================================================================
#
# The decay-weighted statistics below used to rescan their full event
# history on every evaluation, which makes full-risk-set estimation
# O(E² · dyads). Instead we maintain, per (event vector, decay rate),
# lazily decayed accumulators: each entry stores (value, last event time)
# and is decayed on read, and every event is absorbed exactly once via a
# cursor. The values equal the direct sums Σₖ exp(-decay · (t_now - tₖ))
# up to floating-point rounding — the decay factors telescope exactly.

mutable struct _DecayAccum
    cursor::Int                                        # events absorbed so far
    last_event::Tuple{Int,Int,Float64}                 # last absorbed (s, r, t)
    dyad::Dict{Tuple{Int,Int}, Tuple{Float64,Float64}} # (value, last time) per dyad
    out_val::Dict{Int, Tuple{Float64,Float64}}         # per sender
    in_val::Dict{Int, Tuple{Float64,Float64}}          # per receiver
    out_n::Dict{Int, Int}                              # undecayed sender event counts

    _DecayAccum() = new(0, (0, 0, 0.0),
                        Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}(),
                        Dict{Int, Tuple{Float64,Float64}}(),
                        Dict{Int, Tuple{Float64,Float64}}(),
                        Dict{Int, Int}())
end

function _reset!(acc::_DecayAccum)
    acc.cursor = 0
    acc.last_event = (0, 0, 0.0)
    empty!(acc.dyad)
    empty!(acc.out_val)
    empty!(acc.in_val)
    empty!(acc.out_n)
    return acc
end

# Each decay-weighted statistic owns an _AccumCache: one accumulator per
# event vector it has been evaluated against (identified by `===`, held
# through a WeakRef so a statistic does not keep dead states alive). The
# list is typically length 1, so lookup is a couple of pointer compares —
# this sits on the innermost estimation loop.
mutable struct _AccumCache
    entries::Vector{Tuple{WeakRef, _DecayAccum}}

    _AccumCache() = new(Tuple{WeakRef, _DecayAccum}[])
end

_event_srt(e::Event) = (e.sender, e.receiver, float(e.time))
_event_srt(e::Tuple) = (e[1], e[2], float(e[3]))

function _bump!(d::Dict{K, Tuple{Float64,Float64}}, k::K, t::Float64,
                decay::Float64) where K
    entry = get(d, k, nothing)
    if entry === nothing
        d[k] = (1.0, t)
    else
        v, t0 = entry
        d[k] = (v * exp(-decay * (t - t0)) + 1.0, t)
    end
    return nothing
end

function _read(d::Dict{K, Tuple{Float64,Float64}}, k::K, t_now::Float64,
               decay::Float64) where K
    entry = get(d, k, nothing)
    entry === nothing && return 0.0
    v, t0 = entry
    return v * exp(-decay * (t_now - t0))
end

# The synced accumulator for `events` at rate `decay`: absorb any events
# appended since the last call (rebuilding from scratch if the vector was
# reset or rewritten in place).
function _accum(cache::_AccumCache, events::AbstractVector, decay::Float64)
    entries = cache.entries
    acc = nothing
    i = 1
    while i <= length(entries)
        source = entries[i][1].value
        if source === nothing
            deleteat!(entries, i)          # its state was garbage-collected
        elseif source === events
            acc = entries[i][2]
            break
        else
            i += 1
        end
    end
    if acc === nothing
        acc = _DecayAccum()
        push!(entries, (WeakRef(events), acc))
    end

    n = length(events)
    if acc.cursor > n ||
       (acc.cursor > 0 && _event_srt(events[acc.cursor]) != acc.last_event)
        _reset!(acc)
    end

    for i in (acc.cursor + 1):n
        s, r, t = _event_srt(events[i])
        _bump!(acc.dyad, (s, r), t, decay)
        _bump!(acc.out_val, s, t, decay)
        _bump!(acc.in_val, r, t, decay)
        acc.out_n[s] = get(acc.out_n, s, 0) + 1
    end
    if n > 0
        acc.cursor = n
        acc.last_event = _event_srt(events[n])
    end

    return acc
end

# =============================================================================
# Advanced REM Statistics
# =============================================================================
#
# Each statistic implements BOTH interfaces:
#
#   compute(stat, history::InteractionHistory, sender, receiver, current_time)
#       — used by this package's full-risk-set estimators;
#   compute(stat, state::REM.EventNetworkState, sender, receiver)
#       — REM.jl's interface, so these statistics work inside REM.fit_rem
#         and generate_observations.
#
# Decay is half-life parameterized: decay = log(2)/halflife.

"""
    PriorInteraction(halflife; direction=:outgoing) <: AbstractStatistic

Decayed count of prior interactions on the dyad (`:outgoing` = sender→
receiver events, `:incoming` = receiver→sender, `:both` = their sum).
"""
struct PriorInteraction <: AbstractStatistic
    halflife::Float64
    direction::Symbol
    cache::_AccumCache

    function PriorInteraction(halflife::Float64; direction::Symbol=:outgoing)
        direction in (:outgoing, :incoming, :both) ||
            throw(ArgumentError("direction must be :outgoing, :incoming, or :both"))
        halflife > 0 || throw(ArgumentError("halflife must be positive"))
        new(halflife, direction, _AccumCache())
    end
end

name(stat::PriorInteraction) = "prior_interaction_$(stat.direction)"

function _prior_interaction(stat::PriorInteraction, events::AbstractVector,
                            sender::Int, receiver::Int, t_now::Float64)
    decay = log(2) / stat.halflife
    acc = _accum(stat.cache, events, decay)
    value = 0.0
    if stat.direction in (:outgoing, :both)
        value += _read(acc.dyad, (sender, receiver), t_now, decay)
    end
    if stat.direction in (:incoming, :both)
        value += _read(acc.dyad, (receiver, sender), t_now, decay)
    end
    return value
end

compute(stat::PriorInteraction, history::InteractionHistory{T},
        sender::Int, receiver::Int, current_time::T) where T =
    _prior_interaction(stat, history.events, sender, receiver, float(current_time))

compute(stat::PriorInteraction, state::REM.EventNetworkState,
        sender::Int, receiver::Int) =
    _prior_interaction(stat, state.event_history, sender, receiver,
                       float(state.current_time))

"""
    SendingCapacity(halflife) <: AbstractStatistic

Decayed count of ALL of the sender's past outgoing events (sender
activity), regardless of receiver.
"""
struct SendingCapacity <: AbstractStatistic
    halflife::Float64
    cache::_AccumCache

    SendingCapacity(halflife::Float64) = new(halflife, _AccumCache())
end

name(::SendingCapacity) = "sending_capacity"

function compute(stat::SendingCapacity, history::InteractionHistory{T},
                 sender::Int, receiver::Int, current_time::T) where T
    decay = log(2) / stat.halflife
    return _read(_accum(stat.cache, history.events, decay).out_val, sender,
                 float(current_time), decay)
end

function compute(stat::SendingCapacity, state::REM.EventNetworkState,
                 sender::Int, receiver::Int)
    decay = log(2) / stat.halflife
    return _read(_accum(stat.cache, state.event_history, decay).out_val, sender,
                 float(state.current_time), decay)
end

"""
    ReceivingCapacity(halflife) <: AbstractStatistic

Decayed count of ALL of the receiver's past incoming events (receiver
popularity).
"""
struct ReceivingCapacity <: AbstractStatistic
    halflife::Float64
    cache::_AccumCache

    ReceivingCapacity(halflife::Float64) = new(halflife, _AccumCache())
end

name(::ReceivingCapacity) = "receiving_capacity"

function compute(stat::ReceivingCapacity, history::InteractionHistory{T},
                 sender::Int, receiver::Int, current_time::T) where T
    decay = log(2) / stat.halflife
    return _read(_accum(stat.cache, history.events, decay).in_val, receiver,
                 float(current_time), decay)
end

function compute(stat::ReceivingCapacity, state::REM.EventNetworkState,
                 sender::Int, receiver::Int)
    decay = log(2) / stat.halflife
    return _read(_accum(stat.cache, state.event_history, decay).in_val, receiver,
                 float(state.current_time), decay)
end

"""
    LocalInertia(halflife) <: AbstractStatistic

Decayed recency of the *most recent* event on the dyad (0 when the dyad
has no history).
"""
struct LocalInertia <: AbstractStatistic
    halflife::Float64
end

name(::LocalInertia) = "local_inertia"

function compute(stat::LocalInertia, history::InteractionHistory{T},
                 sender::Int, receiver::Int, current_time::T) where T
    decay = log(2) / stat.halflife
    times = get(history.pair_history, (sender, receiver), T[])
    isempty(times) && return 0.0
    return exp(-decay * float(current_time - times[end]))
end

function compute(stat::LocalInertia, state::REM.EventNetworkState,
                 sender::Int, receiver::Int)
    decay = log(2) / stat.halflife
    t_last = get(state.last_event_time, (sender, receiver), nothing)
    isnothing(t_last) && return 0.0
    return exp(-decay * float(state.current_time - t_last))
end

"""
    Momentum(halflife; normalize=false) <: AbstractStatistic

The sender's decayed activity. With `normalize=true` the value is divided
by the sender's own (undecayed) event count, giving the average decay
weight of the sender's past events.
"""
struct Momentum <: AbstractStatistic
    halflife::Float64
    normalize::Bool
    cache::_AccumCache

    Momentum(halflife::Float64; normalize::Bool=false) =
        new(halflife, normalize, _AccumCache())
end

name(::Momentum) = "momentum"

function _momentum(stat::Momentum, events::AbstractVector, sender::Int,
                   t_now::Float64)
    decay = log(2) / stat.halflife
    acc = _accum(stat.cache, events, decay)
    value = _read(acc.out_val, sender, t_now, decay)
    if stat.normalize
        n_sender = get(acc.out_n, sender, 0)
        n_sender > 0 && (value /= n_sender)
    end
    return value
end

compute(stat::Momentum, history::InteractionHistory{T},
        sender::Int, receiver::Int, current_time::T) where T =
    _momentum(stat, history.events, sender, float(current_time))

compute(stat::Momentum, state::REM.EventNetworkState,
        sender::Int, receiver::Int) =
    _momentum(stat, state.event_history, sender, float(state.current_time))

# =============================================================================
# Participation shifts (Gibson 2003; Butts 2008)
# =============================================================================

# The 13 participation shifts, named as in R relevent's rem.dyad
# ("PSAB-BA" ↦ :AB_BA). Grouped as in Gibson (2003):
#   turn receiving:  AB-BA, AB-B0, AB-BY
#   turn claiming:   A0-X0, A0-XA, A0-XY
#   turn usurping:   AB-X0, AB-XA, AB-XB, AB-XY
#   turn continuing: A0-AY, AB-A0, AB-AY
const _PSHIFT_TYPES = (:AB_BA, :AB_B0, :AB_BY,
                       :A0_X0, :A0_XA, :A0_XY,
                       :AB_X0, :AB_XA, :AB_XB, :AB_XY,
                       :A0_AY, :AB_A0, :AB_AY)

"""
    pshift_types() -> NTuple{13, Symbol}

The 13 Gibson (2003) participation-shift types accepted by [`PShift`](@ref),
in relevent's grouping (turn receiving, claiming, usurping, continuing).
"""
pshift_types() = _PSHIFT_TYPES

"""
    PShift(shift::Symbol) <: AbstractStatistic
    PShift(shift::AbstractString) <: AbstractStatistic

Participation-shift indicator (Gibson 2003), matching R relevent's
`PSAB-BA`-family effects: with previous event A→B, the statistic is 1 for
a candidate event that realizes the shift, 0 otherwise (and 0 for the
first event, which has no previous event).

`shift` is one of [`pshift_types`](@ref) (e.g. `:AB_BA`) or the R name
(e.g. `"PSAB-BA"`). In shift names, `A`/`B` are the previous event's
sender/receiver, `X`/`Y` are any *other* actors, and `0` is the null
actor: an event "to the group" is encoded with `receiver == 0`, so shifts
involving `0` (e.g. `:AB_B0`, `:A0_X0`) can only be nonzero when such
group-directed events occur in the data — with strictly dyadic events
they are structurally zero, exactly as in `relevent::rem.dyad`.

# Examples
```julia
PShift(:AB_BA)      # turn receiving: B answers A
PShift("PSAB-XY")   # turn usurping: an outsider addresses another outsider
```
"""
struct PShift <: AbstractStatistic
    shift::Symbol
    stat_name::String

    function PShift(shift::Symbol; name::String="")
        shift in _PSHIFT_TYPES ||
            throw(ArgumentError("unknown participation shift :$shift; " *
                                "valid shifts: $(join(_PSHIFT_TYPES, ", "))"))
        new(shift, isempty(name) ? "PS" * replace(String(shift), "_" => "-") : name)
    end
end

PShift(shift::AbstractString; kwargs...) =
    PShift(Symbol(replace(replace(String(shift), r"^PS" => ""), "-" => "_")); kwargs...)

name(stat::PShift) = stat.stat_name

# Indicator that the candidate event i→j realizes `shift` after the
# previous event a→b (b == 0 means the previous event was group-directed).
function _pshift_value(shift::Symbol, a::Int, b::Int, i::Int, j::Int)
    i == j && return 0.0
    if b == 0
        # Previous event was A→0 (to the group)
        shift === :A0_X0 && return (i != a && j == 0) ? 1.0 : 0.0
        shift === :A0_XA && return (i != a && j == a) ? 1.0 : 0.0
        shift === :A0_XY && return (i != a && j != a && j != 0) ? 1.0 : 0.0
        shift === :A0_AY && return (i == a && j != a && j != 0) ? 1.0 : 0.0
        return 0.0
    else
        # Previous event was dyadic A→B
        new_i = i != a && i != b
        new_j = j != a && j != b && j != 0
        shift === :AB_BA && return (i == b && j == a) ? 1.0 : 0.0
        shift === :AB_B0 && return (i == b && j == 0) ? 1.0 : 0.0
        shift === :AB_BY && return (i == b && new_j) ? 1.0 : 0.0
        shift === :AB_A0 && return (i == a && j == 0) ? 1.0 : 0.0
        shift === :AB_AY && return (i == a && new_j) ? 1.0 : 0.0
        shift === :AB_X0 && return (new_i && j == 0) ? 1.0 : 0.0
        shift === :AB_XA && return (new_i && j == a) ? 1.0 : 0.0
        shift === :AB_XB && return (new_i && j == b) ? 1.0 : 0.0
        shift === :AB_XY && return (new_i && new_j) ? 1.0 : 0.0
        return 0.0
    end
end

function compute(stat::PShift, history::InteractionHistory{T},
                 sender::Int, receiver::Int, current_time::T) where T
    isempty(history.events) && return 0.0
    prev = history.events[end]
    return _pshift_value(stat.shift, prev.sender, prev.receiver, sender, receiver)
end

function compute(stat::PShift, state::REM.EventNetworkState,
                 sender::Int, receiver::Int)
    isempty(state.event_history) && return 0.0
    a, b, _, _ = state.event_history[end]
    return _pshift_value(stat.shift, a, b, sender, receiver)
end

# =============================================================================
# Covariate effects (CovSnd / CovRec / CovInt, as in R relevent)
# =============================================================================

"""
    CovSnd(values::AbstractVector{<:Real}; name="CovSnd") <: AbstractStatistic

Covariate effect for outgoing actions, as in R relevent: the statistic
for a candidate event i→j is `values[i]` (the sender's covariate value).
`values` is indexed by actor ID.
"""
struct CovSnd <: AbstractStatistic
    values::Vector{Float64}
    stat_name::String

    CovSnd(values::AbstractVector{<:Real}; name::String="CovSnd") =
        new(collect(Float64, values), name)
end

name(stat::CovSnd) = stat.stat_name

"""
    CovRec(values::AbstractVector{<:Real}; name="CovRec") <: AbstractStatistic

Covariate effect for incoming actions, as in R relevent: the statistic
for a candidate event i→j is `values[j]` (the receiver's covariate
value). `values` is indexed by actor ID.
"""
struct CovRec <: AbstractStatistic
    values::Vector{Float64}
    stat_name::String

    CovRec(values::AbstractVector{<:Real}; name::String="CovRec") =
        new(collect(Float64, values), name)
end

name(stat::CovRec) = stat.stat_name

"""
    CovInt(values::AbstractVector{<:Real}; name="CovInt") <: AbstractStatistic

Covariate effect for both outgoing and incoming actions, as in R
relevent: the statistic for a candidate event i→j is
`values[i] + values[j]`. `values` is indexed by actor ID.
"""
struct CovInt <: AbstractStatistic
    values::Vector{Float64}
    stat_name::String

    CovInt(values::AbstractVector{<:Real}; name::String="CovInt") =
        new(collect(Float64, values), name)
end

name(stat::CovInt) = stat.stat_name

function _cov_value(values::Vector{Float64}, actor::Int)
    1 <= actor <= length(values) ||
        throw(ArgumentError("actor $actor has no covariate value " *
                            "(covariate has length $(length(values)))"))
    return values[actor]
end

compute(stat::CovSnd, ::InteractionHistory{T}, sender::Int, receiver::Int,
        ::T) where T = _cov_value(stat.values, sender)
compute(stat::CovSnd, ::REM.EventNetworkState, sender::Int, receiver::Int) =
    _cov_value(stat.values, sender)

compute(stat::CovRec, ::InteractionHistory{T}, sender::Int, receiver::Int,
        ::T) where T = _cov_value(stat.values, receiver)
compute(stat::CovRec, ::REM.EventNetworkState, sender::Int, receiver::Int) =
    _cov_value(stat.values, receiver)

compute(stat::CovInt, ::InteractionHistory{T}, sender::Int, receiver::Int,
        ::T) where T =
    _cov_value(stat.values, sender) + _cov_value(stat.values, receiver)
compute(stat::CovInt, ::REM.EventNetworkState, sender::Int, receiver::Int) =
    _cov_value(stat.values, sender) + _cov_value(stat.values, receiver)

# =============================================================================
# Full-risk-set machinery shared by the estimators
# =============================================================================

# For each event (in time order): the case's statistic vector and the
# matrix of statistics for every dyad in the full risk set. History is
# strictly pre-event (no look-ahead). `t0` is the observation onset: the
# first event's waiting time is measured from it.
#
# Statistics are converted to a tuple (mirroring REM's tuple-backed
# StatisticSet) so the inner loop over dyads compiles to statically
# dispatched compute calls instead of dynamic dispatch through an
# abstractly-typed vector.
_risk_set_stats(events::Vector{Event{T}}, statistics::AbstractVector, n_actors::Int;
                kwargs...) where T =
    _risk_set_stats(events, Tuple(statistics), n_actors; kwargs...)

function _risk_set_stats(events::Vector{Event{T}}, statistics::Tuple, n_actors::Int;
                         t0::T=zero(T)) where T
    sorted = sort(events, by=e -> e.time)
    if length(sorted) > 1 && !allunique(e.time for e in sorted)
        @warn "Event sequence contains tied timestamps; ties are ordered " *
              "arbitrarily and no tie correction is applied" maxlog = 1
    end

    p = length(statistics)
    dyads = [(s, r) for s in 1:n_actors for r in 1:n_actors if s != r]
    history = InteractionHistory{T}()

    if !isempty(sorted) && t0 > sorted[1].time
        throw(ArgumentError("t0 = $t0 is after the first event time " *
                            "$(sorted[1].time); the observation onset must " *
                            "precede all events"))
    end

    case_idx = Vector{Int}(undef, length(sorted))
    X = Vector{Matrix{Float64}}(undef, length(sorted))
    waiting = Vector{Float64}(undef, length(sorted))
    t_prev = t0

    for (m, ev) in enumerate(sorted)
        Xm = Matrix{Float64}(undef, length(dyads), p)
        ci = 0
        for (d, (s, r)) in enumerate(dyads)
            vals = map(stat -> compute(stat, history, s, r, ev.time), statistics)
            for k in 1:p
                Xm[d, k] = vals[k]
            end
            (s == ev.sender && r == ev.receiver) && (ci = d)
        end
        ci > 0 || throw(ArgumentError("event $(m) references actors outside 1:$n_actors"))
        case_idx[m] = ci
        X[m] = Xm
        waiting[m] = float(ev.time - t_prev)
        t_prev = ev.time

        update_history!(history, ev)
    end

    return dyads, case_idx, X, waiting
end

# Newton-Raphson with step-halving on a concave objective returning
# (value, gradient, Hessian)
function _newton(derivatives, p::Int; maxiter::Int=100, tol::Float64=1e-8)
    θ = zeros(p)
    ll, grad, hess = derivatives(θ)
    converged = false

    for _ in 1:maxiter
        step = try
            -hess \ grad
        catch
            break
        end

        stepsize = 1.0
        ll_new, grad_new, hess_new = ll, grad, hess
        for _ in 1:10
            ll_new, grad_new, hess_new = derivatives(θ .+ stepsize .* step)
            ll_new >= ll && break
            stepsize /= 2
        end

        θ .+= stepsize .* step
        ll_change = abs(ll_new - ll)
        ll, grad, hess = ll_new, grad_new, hess_new

        if ll_change < tol && norm(grad) < sqrt(tol)
            converged = true
            break
        end
    end

    se = try
        sqrt.(abs.(diag(pinv(-hess))))
    catch
        fill(NaN, p)
    end

    return θ, se, ll, converged
end

# =============================================================================
# Ordinal Butts-Park Model
# =============================================================================

"""
    OrdinalBPM

Ordinal relational event model: only the *order* of events is used. Each
event contributes a multinomial-logit term over the **full risk set**,

    P(event m is (s, r)) = exp(θ'x_sr) / Σ_{(i,j)} exp(θ'x_ij),

the ordinal likelihood of `relevent::rem.dyad` (no case-control
sampling — compare `REM.fit_rem`, which samples the risk set).
"""
struct OrdinalBPM
    statistics::Vector{AbstractStatistic}
    n_actors::Int

    function OrdinalBPM(statistics::Vector{<:AbstractStatistic}, n::Int)
        isempty(statistics) && throw(ArgumentError("need at least one statistic"))
        n >= 2 || throw(ArgumentError("need at least two actors"))
        new(collect(AbstractStatistic, statistics), n)
    end
end

"""
    OrdinalBPMResult

Results from fitting an ordinal BPM: exact maximum-likelihood
coefficients over the full risk set, with observed-information standard
errors.
"""
struct OrdinalBPMResult
    model::OrdinalBPM
    coefficients::Vector{Float64}
    std_errors::Vector{Float64}
    loglik::Float64
    converged::Bool
    n_events::Int
end

# z statistics and two-sided normal (Wald) p-values for a coefficient table;
# NaN where the standard error is unavailable.
function _wald_zp(coefs::Vector{Float64}, ses::Vector{Float64})
    z = [se > 0 ? c / se : NaN for (c, se) in zip(coefs, ses)]
    p = [isnan(zk) ? NaN : 2 * ccdf(Normal(), abs(zk)) for zk in z]
    return z, p
end

function Base.show(io::IO, result::OrdinalBPMResult)
    println(io, "Ordinal Butts-Park Model Results")
    println(io, "================================")
    println(io, "N actors: $(result.model.n_actors)")
    println(io, "N events: $(result.n_events)")
    println(io, "Log-likelihood: $(round(result.loglik, digits=4))")
    println(io, "Converged: $(result.converged)")
    println(io)
    z, p = _wald_zp(result.coefficients, result.std_errors)
    print_coeftable(io, [name(stat) for stat in result.model.statistics],
                    result.coefficients, result.std_errors, p; z_values=z)
end

"""
    coef(result::OrdinalBPMResult) -> Vector{Float64}

Extract coefficients from a fitted ordinal BPM (StatsAPI method).
"""
coef(result::OrdinalBPMResult) = result.coefficients

"""
    stderror(result::OrdinalBPMResult) -> Vector{Float64}

Extract standard errors from a fitted ordinal BPM (StatsAPI method).
"""
stderror(result::OrdinalBPMResult) = result.std_errors

"""
    rank_events(events::Vector{Event}) -> Vector{Int}

Ordinal ranks of the events by time (1 = earliest). Ties are broken
arbitrarily by `sortperm` and produce a one-time warning: the ordinal
likelihood applies no tie correction.
"""
function rank_events(events::Vector{Event{T}}) where T
    times = [e.time for e in events]
    if length(times) > 1 && !allunique(times)
        @warn "rank_events: tied timestamps are ranked arbitrarily" maxlog = 1
    end
    sorted_indices = sortperm(times)
    ranks = zeros(Int, length(events))
    for (rank, idx) in enumerate(sorted_indices)
        ranks[idx] = rank
    end
    return ranks
end

"""
    fit_obpm(events, statistics, n_actors; maxiter=100, tol=1e-8) -> OrdinalBPMResult

Fit the ordinal relational event model by exact maximum likelihood over
the full risk set (Newton-Raphson with step-halving).
"""
function fit_obpm(events::Vector{Event{T}}, statistics::Vector{<:AbstractStatistic},
                  n_actors::Int; maxiter::Int=100, tol::Float64=1e-8) where T
    model = OrdinalBPM(collect(AbstractStatistic, statistics), n_actors)
    p = length(statistics)
    isempty(events) && throw(ArgumentError("no events to fit"))

    _, case_idx, X, _ = _risk_set_stats(events, model.statistics, n_actors)
    M = length(X)

    # Preallocated buffers shared by every derivative evaluation (all risk
    # sets have the same size); the Hessian is accumulated in place via
    # BLAS (gemm on sqrt(prob)-weighted rows for -E[XX'], ger! for the
    # +E[X]E[X]' rank-1 update) instead of allocating per-event matrices.
    D = size(X[1], 1)
    η = Vector{Float64}(undef, D)
    probs = Vector{Float64}(undef, D)
    W = Matrix{Float64}(undef, D, p)
    x_exp = Vector{Float64}(undef, p)

    function derivatives(θ)
        ll = 0.0
        grad = zeros(p)
        hess = zeros(p, p)
        for m in 1:M
            Xm = X[m]
            mul!(η, Xm, θ)
            ηmax = maximum(η)
            Z = 0.0
            @inbounds for d in 1:D
                probs[d] = exp(η[d] - ηmax)
                Z += probs[d]
            end
            probs ./= Z

            ll += η[case_idx[m]] - ηmax - log(Z)

            mul!(x_exp, transpose(Xm), probs)
            @inbounds for k in 1:p
                grad[k] += Xm[case_idx[m], k] - x_exp[k]
            end
            W .= sqrt.(probs) .* Xm
            mul!(hess, transpose(W), W, -1.0, 1.0)
            BLAS.ger!(1.0, x_exp, x_exp, hess)
        end
        return ll, grad, hess
    end

    θ, se, ll, converged = _newton(derivatives, p; maxiter=maxiter, tol=tol)
    return OrdinalBPMResult(model, θ, se, ll, converged, length(events))
end

# =============================================================================
# Timing Models
# =============================================================================

"""
    TimingModel

Proportional-hazards model for inter-event times: dyad (i,j) has hazard
`λ_ij(t) = h₀(t)·exp(θ'x_ij)` with baseline `h₀` one of `:exponential`,
`:weibull`, or `:gompertz`. Only the exponential baseline has a fitted
likelihood (see [`fit_timing`](@ref)); the others are available for
hazard/survival evaluation.
"""
struct TimingModel
    statistics::Vector{AbstractStatistic}
    baseline::Symbol

    function TimingModel(statistics::Vector{<:AbstractStatistic};
                         baseline::Symbol=:exponential)
        baseline in (:exponential, :weibull, :gompertz) ||
            throw(ArgumentError("baseline must be :exponential, :weibull, or :gompertz"))
        new(collect(AbstractStatistic, statistics), baseline)
    end
end

"""
    TimingModelResult

Results from fitting a timing model. `baseline_params[1]` is the fitted
baseline rate λ₀; coefficients are proportional-hazard log-rate effects.
"""
struct TimingModelResult
    model::TimingModel
    coefficients::Vector{Float64}
    baseline_params::Vector{Float64}
    std_errors::Vector{Float64}
    loglik::Float64
    converged::Bool
end

function Base.show(io::IO, result::TimingModelResult)
    println(io, "Timing Model Results")
    println(io, "====================")
    println(io, "Baseline: $(result.model.baseline)")
    println(io, "Baseline rate λ₀: $(round(result.baseline_params[1], digits=6))")
    println(io, "Log-likelihood: $(round(result.loglik, digits=4))")
    println(io, "Converged: $(result.converged)")
    println(io)
    z, p = _wald_zp(result.coefficients, result.std_errors)
    print_coeftable(io, [name(stat) for stat in result.model.statistics],
                    result.coefficients, result.std_errors, p; z_values=z)
end

"""
    coef(result::TimingModelResult) -> Vector{Float64}

Extract coefficients from a fitted timing model (StatsAPI method; the
baseline rate is in `result.baseline_params`).
"""
coef(result::TimingModelResult) = result.coefficients

"""
    stderror(result::TimingModelResult) -> Vector{Float64}

Extract standard errors from a fitted timing model (StatsAPI method).
"""
stderror(result::TimingModelResult) = result.std_errors

"""
    hazard_rate(model::TimingModel, coef, baseline_params, t, x) -> Float64

The hazard `h₀(t)·exp(θ'x)` at time `t` for a dyad with statistics `x`.
"""
function hazard_rate(model::TimingModel, coef::Vector{Float64},
                     baseline_params::Vector{Float64}, t::Float64, x::Vector{Float64})
    eta = dot(coef, x)

    if model.baseline == :exponential
        lambda = baseline_params[1]
        return lambda * exp(eta)
    elseif model.baseline == :weibull
        lambda, k = baseline_params[1], baseline_params[2]
        return (k / lambda) * (t / lambda)^(k - 1) * exp(eta)
    elseif model.baseline == :gompertz
        a, b = baseline_params[1], baseline_params[2]
        return a * exp(b * t) * exp(eta)
    end
    error("unreachable: unknown baseline $(model.baseline)")
end

"""
    survival_function(model::TimingModel, coef, baseline_params, t, x) -> Float64

The survival probability `exp(-∫₀ᵗ h(u) du)` (proportional-hazards
parameterization for every baseline).
"""
function survival_function(model::TimingModel, coef::Vector{Float64},
                           baseline_params::Vector{Float64}, t::Float64, x::Vector{Float64})
    eta = dot(coef, x)

    if model.baseline == :exponential
        lambda = baseline_params[1]
        return exp(-lambda * exp(eta) * t)
    elseif model.baseline == :weibull
        lambda, k = baseline_params[1], baseline_params[2]
        return exp(-(t / lambda)^k * exp(eta))
    elseif model.baseline == :gompertz
        a, b = baseline_params[1], baseline_params[2]
        return exp(-(a / b) * (exp(b * t) - 1) * exp(eta))
    end
    error("unreachable: unknown baseline $(model.baseline)")
end

"""
    fit_timing(events, statistics, n_actors; baseline=:exponential,
               t0=zero(T), maxiter=100, tol=1e-8) -> TimingModelResult

Fit the interval-timing relational event model with an exponential
baseline by exact maximum likelihood: with waiting time `Δt_m` before
event m and per-dyad hazards `λ₀·exp(θ'x_ij)` (statistics frozen between
events),

    ℓ(λ₀, θ) = Σ_m [ log λ₀ + θ'x_case − λ₀ Δt_m Σ_{ij} exp(θ'x_ij) ].

`(log λ₀, θ)` are estimated jointly by Newton-Raphson; standard errors
come from the observed information. Weibull/Gompertz baselines are not
fitted (an informative error is raised); use them with
[`hazard_rate`](@ref)/[`survival_function`](@ref).

`t0` is the observation onset: the first event's waiting time is
`Δt_1 = t_1 − t0`. The default `t0 = zero(T)` reproduces the previous
behavior and matches R relevent, where "event times should be relative
to onset of observation" (i.e. the clock starts at 0). For a
left-truncated observation window — a process already running when
recording started — pass the window start as `t0` so the first interval
is not overstated; `t0` must not exceed the first event time.
"""
function fit_timing(events::Vector{Event{T}}, statistics::Vector{<:AbstractStatistic},
                    n_actors::Int; baseline::Symbol=:exponential,
                    t0::T=zero(T),
                    maxiter::Int=100, tol::Float64=1e-8) where T
    model = TimingModel(collect(AbstractStatistic, statistics); baseline=baseline)
    baseline == :exponential ||
        error("fit_timing implements the exponential-baseline likelihood; " *
              ":$baseline is available only for hazard_rate/survival_function")
    isempty(events) && throw(ArgumentError("no events to fit"))

    p = length(statistics)
    _, case_idx, X, waiting = _risk_set_stats(events, model.statistics, n_actors; t0=t0)
    M = length(X)

    # Parameters: β = (log λ₀, θ)
    function derivatives(β)
        logλ = β[1]
        θ = β[2:end]
        λ = exp(logλ)

        ll = 0.0
        grad = zeros(p + 1)
        hess = zeros(p + 1, p + 1)

        for m in 1:M
            Xm = X[m]
            η = Xm * θ
            w = exp.(η)
            S = sum(w)                      # Σ exp(θ'x)
            Sx = Xm' * w                    # Σ x exp(θ'x)
            Sxx = Xm' * (w .* Xm)           # Σ xx' exp(θ'x)
            Δt = waiting[m]

            ll += logλ + η[case_idx[m]] - λ * Δt * S

            grad[1] += 1.0 - λ * Δt * S
            grad[2:end] .+= Xm[case_idx[m], :] .- λ .* Δt .* Sx

            hess[1, 1] += -λ * Δt * S
            hess[1, 2:end] .+= -λ .* Δt .* Sx
            hess[2:end, 1] .+= -λ .* Δt .* Sx
            hess[2:end, 2:end] .-= λ .* Δt .* Sxx
        end

        return ll, grad, hess
    end

    β, se, ll, converged = _newton(derivatives, p + 1; maxiter=maxiter, tol=tol)

    λ0 = exp(β[1])
    return TimingModelResult(model, β[2:end], [λ0], se[2:end], ll, converged)
end

# =============================================================================
# Standardized entry point
# =============================================================================

"""
    fit_relevent(events, statistics, n_actors; ordinal=true, kwargs...)

Fit a dyadic relational event model over the full risk set — the standardized
entry point of this package (`fit_<model>` naming, alongside `REM.fit_rem`,
`ERGM.fit_ergm`, `Siena.fit_siena`, ...). [`rem_dyad`](@ref) is the R-faithful
alias, mirroring `relevent::rem.dyad`.

With `ordinal=true` (the default, as in `rem.dyad`) only the order of events is
used and the model is fitted by [`fit_obpm`](@ref), returning an
[`OrdinalBPMResult`](@ref). With `ordinal=false` the observed waiting times
enter the exponential-baseline interval likelihood of [`fit_timing`](@ref),
returning a [`TimingModelResult`](@ref). Keyword arguments (`maxiter`, `tol`,
and for the interval likelihood `t0`) are forwarded to the underlying fitter.

# Example
```julia
fit_relevent(events, [PShift(:AB_BA), CovSnd(z)], n_actors)                # ordinal
fit_relevent(events, [PShift(:AB_BA)], n_actors; ordinal=false, t0=0.0)   # timing
```
"""
function fit_relevent(events::Vector{Event{T}},
                      statistics::Vector{<:AbstractStatistic}, n_actors::Int;
                      ordinal::Bool=true, kwargs...) where T
    return ordinal ? fit_obpm(events, statistics, n_actors; kwargs...) :
                     fit_timing(events, statistics, n_actors; kwargs...)
end

"""
    rem_dyad(events, statistics, n_actors; ordinal=true, kwargs...)

Alias for [`fit_relevent`](@ref), keeping the R `relevent::rem.dyad` name.
"""
rem_dyad(events::Vector{Event{T}}, statistics::Vector{<:AbstractStatistic},
         n_actors::Int; kwargs...) where T =
    fit_relevent(events, statistics, n_actors; kwargs...)

# =============================================================================
# Cumulative Network State
# =============================================================================

"""
    CumulativeState{T}

Track a decaying adjacency matrix and degree vectors over an event
stream.
"""
mutable struct CumulativeState{T}
    n_actors::Int
    adj_matrix::Matrix{Float64}
    outdegree::Vector{Float64}
    indegree::Vector{Float64}
    last_update::T
    decay::Float64

    function CumulativeState{T}(n::Int; halflife::Float64=Inf) where T
        n >= 1 || throw(ArgumentError("need at least one actor"))
        decay = halflife == Inf ? 0.0 : log(2) / halflife
        new{T}(n, zeros(n, n), zeros(n), zeros(n), zero(T), decay)
    end
end

CumulativeState(n::Int; kwargs...) = CumulativeState{Float64}(n; kwargs...)

"""
    update_state!(state::CumulativeState, event::Event)

Decay the state to the event's time and record the event.
"""
function update_state!(state::CumulativeState{T}, event::Event{T}) where T
    (1 <= event.sender <= state.n_actors && 1 <= event.receiver <= state.n_actors) ||
        throw(ArgumentError("event actors ($(event.sender), $(event.receiver)) " *
                            "outside 1:$(state.n_actors)"))

    if state.decay > 0 && event.time > state.last_update
        dt = float(event.time - state.last_update)
        decay_factor = exp(-state.decay * dt)
        state.adj_matrix .*= decay_factor
        state.outdegree .*= decay_factor
        state.indegree .*= decay_factor
    end

    state.adj_matrix[event.sender, event.receiver] += 1.0
    state.outdegree[event.sender] += 1.0
    state.indegree[event.receiver] += 1.0
    state.last_update = event.time

    return state
end

"""
    get_outdegree_history(state::CumulativeState, actor) -> Float64

The (possibly decayed) out-degree of `actor`.
"""
get_outdegree_history(state::CumulativeState, actor::Int) = state.outdegree[actor]

"""
    get_indegree_history(state::CumulativeState, actor) -> Float64

The (possibly decayed) in-degree of `actor`.
"""
get_indegree_history(state::CumulativeState, actor::Int) = state.indegree[actor]

end # module
