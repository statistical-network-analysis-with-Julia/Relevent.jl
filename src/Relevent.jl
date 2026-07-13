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
# Shared result-presentation infrastructure (Networks.jl): the R-style
# coefficient table used by every model package in the ecosystem
using Networks: print_coeftable

# The shared result-metadata protocol (Networks.jl `src/results.jl`): the seven
# generic accessors that say what a fit actually did. Imported by name because
# Relevent adds methods for `OrdinalBPMResult` and `TimingModelResult`;
# `Networks.fit_metadata(fit)` collects them.
import Networks: estimand, objective, is_exact, se_method, missing_method,
                 tie_method, approximations

# The shared TIED-EVENT vocabulary (Networks.jl `src/results.jl`): one `ties=`
# keyword and one meaning per symbol across REM.jl and Relevent.jl.
# `check_tie_policy` refuses a policy a model cannot honour — `:batch` for the
# ordinal likelihood, `:breslow`/`:efron` for the exact-time one — instead of
# letting it silently no-op.
using Networks: TIE_POLICIES, check_tie_policy

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

# =============================================================================
# Tied event times (issue Relevent#1, review finding 12)
# =============================================================================
#
# Both estimators here claim more than an event list gives them when two events
# share a timestamp, and they claim DIFFERENT things, so they take different
# subsets of the shared `Networks.TIE_POLICIES` vocabulary:
#
#   fit_obpm  — a likelihood over the ORDER of events. A tie means the order is
#               genuinely unknown; sorting it invents information (and, because
#               the statistics are read off the pre-event history, lets the event
#               placed first enter the statistics of the one placed second). The
#               likelihood is a multinomial partial likelihood, so the classical
#               Cox tie corrections apply verbatim:
#                 :error (default) | :ordered | :breslow | :efron
#               `:batch` is refused — with the history frozen across the tied
#               events, a "simultaneous batch" IS the Breslow correction.
#
#   fit_timing — an exact-time (exponential) likelihood. Under a continuous-time
#               model P(tie) = 0: a tie is not a broken ordering but a violated
#               assumption — a coarsened clock, or a genuinely simultaneous
#               batch. There is no partial likelihood here to correct, so
#               :breslow and :efron are refused (they are not "a bit wrong"
#               here; they are undefined):
#                 :error (default) | :ordered | :batch
#               `:batch` reads the tie as one simultaneous batch: the events
#               cannot have influenced one another (history frozen across the
#               block) and the block consumes ONE exposure interval. `:ordered`
#               instead lets each tied event after the first enter with a
#               ZERO-LENGTH waiting interval — exposure the model then never
#               sees — while still updating the history, i.e. it claims that one
#               event caused the next in no time at all.

const _OBPM_TIES_SUPPORTED = (:error, :ordered, :breslow, :efron)
const _OBPM_TIES_MODEL = "`fit_obpm` (ordinal BPM: a likelihood over the ORDER of events)"
const _OBPM_TIES_REASONS = Dict(
    :batch => "an ordinal likelihood has no exposure interval for a batch to " *
              "consume; holding the risk set fixed across the tied events and " *
              "giving each its own multinomial term IS the Breslow correction, " *
              "so pass `ties=:breslow` (or `:efron`) instead")

const _TIMING_TIES_SUPPORTED = (:error, :ordered, :batch)
const _TIMING_TIES_MODEL =
    "`fit_timing` (exponential-baseline interval likelihood: an EXACT-TIME model)"
const _TIMING_TIES_REASONS = Dict(
    :breslow => "Breslow and Efron are corrections to a PARTIAL likelihood, in " *
                "which the baseline hazard is profiled out and only the order of " *
                "events is used. `fit_timing` maximizes the exact exponential " *
                "likelihood of the waiting TIMES; there is no partial-likelihood " *
                "denominator here for them to re-weight. Ties under a " *
                "continuous-time model are a violated assumption, not a broken " *
                "ordering: read them as a coarsened, simultaneous batch " *
                "(`ties=:batch`), or fit the ordinal model with `fit_obpm(...; " *
                "ties=:breslow)` if the order is what you care about",
    :efron   => "Breslow and Efron are corrections to a PARTIAL likelihood, in " *
                "which the baseline hazard is profiled out and only the order of " *
                "events is used. `fit_timing` maximizes the exact exponential " *
                "likelihood of the waiting TIMES; there is no partial-likelihood " *
                "denominator here for them to re-weight. Ties under a " *
                "continuous-time model are a violated assumption, not a broken " *
                "ordering: read them as a coarsened, simultaneous batch " *
                "(`ties=:batch`), or fit the ordinal model with `fit_obpm(...; " *
                "ties=:efron)` if the order is what you care about")

# Maximal runs of equal event time in a time-sorted vector: a tie is a run of
# length > 1.
function _tie_blocks(sorted::Vector{Event{T}}) where T
    blocks = UnitRange{Int}[]
    i = 1
    n = length(sorted)
    while i <= n
        j = i
        while j < n && sorted[j + 1].time == sorted[i].time
            j += 1
        end
        push!(blocks, i:j)
        i = j + 1
    end
    return blocks
end

# `ties=:error`: name the tie, do not fit. `claim` says what the model is
# claiming that the tied data cannot support.
function _reject_ties(sorted::Vector{Event{T}}, blocks::Vector{UnitRange{Int}},
                      claim::AbstractString, advice::AbstractString) where T
    tied = filter(b -> length(b) > 1, blocks)
    isempty(tied) && return nothing
    b = first(tied)
    n_tied_events = sum(length, tied)
    throw(ArgumentError(
        "Event sequence contains tied timestamps: events $(first(b))–$(last(b)) " *
        "($(length(b)) of them, in time order) all occur at t = " *
        "$(sorted[first(b)].time)" *
        (length(tied) > 1 ?
         "; $(length(tied)) timestamps carry ties in all ($n_tied_events events)" :
         "") * ". $claim $advice"))
end

# =============================================================================
# The risk-set PLAN, and the three cache policies (issue Relevent#2)
# =============================================================================
#
# Both estimators need, for every interval m (in time order): the case's index
# into the full risk set, the waiting time, and the `n(n−1) × p` matrix of
# statistics for EVERY dyad in the risk set, read off the strictly pre-event
# interaction history (no look-ahead).
#
# Materializing all of those matrices costs `O(E · n(n−1) · p)` doubles — 8 GB
# at (n, E, p) = (100, 2000, 6) — which put a ceiling on exact full-risk-set
# estimation long before the arithmetic became infeasible. Everything else about
# an interval is O(1) or O(n²) *once*, so the split is:
#
#   * `_RiskSetPlan` — the O(E + n²) skeleton, always materialized: which dyad
#     is the case, the waiting time, the time at which the statistics are read,
#     and which events the history absorbs after the interval is emitted (the
#     tie-freeze policies absorb a whole block at once). Also the sparse Efron
#     denominator weights: within a tie block every tied case takes the SAME
#     weight 1 − (j−1)/d, so a list of dyad indices plus one scalar per interval
#     replaces the dense `n(n−1)`-vector the old code stored per event.
#
#   * `_RiskSets` — the design matrices, under one of three policies:
#       `:all`     every matrix materialized once (what the package always did):
#                  fastest, and `O(E · n² · p)` memory.
#       `:chunked` a BOUNDED cache of `chunk` matrices, refilled by replaying the
#                  history from the start on each pass: `O(chunk · n² · p)`.
#       `:none`    `:chunked` with `chunk = 1` — one matrix alive at a time.
#     The passes visit the intervals in the SAME order under every policy and the
#     matrices hold the same values (a design matrix is a deterministic function
#     of the pre-interval history and the read time), so the likelihood, gradient
#     and Hessian are accumulated in an identical order and the fits are
#     bit-identical — asserted in the tests, not hoped for.
#
# `cache=:auto` (the default) is `:all` while the projected footprint fits in
# `cache_bytes` (256 MiB) and `:chunked` above it — `:all` is the right default
# where it is affordable, and it is exactly the case where it is NOT affordable
# that the old code fell over.
#
# Statistics are converted to a tuple (mirroring REM's tuple-backed
# StatisticSet) so the inner loop over dyads compiles to statically
# dispatched compute calls instead of dynamic dispatch through an
# abstractly-typed vector.

const CACHE_MODES = (:auto, :all, :chunked, :none)
const _DEFAULT_CACHE_BYTES = 1 << 28      # 256 MiB

struct _RiskSetPlan{T, S<:Tuple}
    sorted::Vector{Event{T}}
    statistics::S
    dyads::Vector{Tuple{Int,Int}}
    p::Int
    n_int::Int
    case_idx::Vector{Int}          # 0 on the right-censored tail
    waiting::Vector{Float64}
    read_time::Vector{Float64}     # time at which interval m's statistics are read
    absorb::Vector{UnitRange{Int}} # events absorbed AFTER interval m is emitted
    # Efron denominator weights, sparsely: the tied dyads of interval m's block
    # (shared, hence a per-interval reference) and the weight they take. `nothing`
    # unless `ties=:efron` actually bit, which keeps the unweighted inner loop
    # bit-for-bit as it was.
    tw_dyads::Union{Nothing, Vector{Vector{Int}}}
    tw_val::Union{Nothing, Vector{Float64}}
end

n_dyads(plan::_RiskSetPlan) = length(plan.dyads)

# Bytes one design matrix costs, and the projected cost of caching all of them.
_design_bytes(plan::_RiskSetPlan) = n_dyads(plan) * plan.p * sizeof(Float64)
_full_cache_bytes(plan::_RiskSetPlan) = _design_bytes(plan) * plan.n_int

_risk_set_plan(events::Vector{Event{T}}, statistics::AbstractVector, n_actors::Int;
               kwargs...) where T =
    _risk_set_plan(events, Tuple(statistics), n_actors; kwargs...)

function _risk_set_plan(events::Vector{Event{T}}, statistics::Tuple, n_actors::Int;
                        t0::T=zero(T), t_end::Union{Nothing,T}=nothing,
                        ties::Symbol=:ordered) where T
    sorted = sort(events, by=e -> e.time)
    blocks = _tie_blocks(sorted)
    freeze = ties in (:breslow, :efron, :batch)

    p = length(statistics)
    dyads = [(s, r) for s in 1:n_actors for r in 1:n_actors if s != r]
    dyad_index = Dict(dy => d for (d, dy) in enumerate(dyads))

    if !isempty(sorted) && t0 > sorted[1].time
        throw(ArgumentError("t0 = $t0 is after the first event time " *
                            "$(sorted[1].time); the observation onset must " *
                            "precede all events"))
    end

    if t_end !== nothing && !isempty(sorted) && t_end < sorted[end].time
        throw(ArgumentError("t_end = $t_end is before the last event time " *
                            "$(sorted[end].time); the observation window must " *
                            "contain every event"))
    end
    n_int = length(sorted) + (t_end === nothing ? 0 : 1)

    case_idx = Vector{Int}(undef, n_int)
    waiting = Vector{Float64}(undef, n_int)
    read_time = Vector{Float64}(undef, n_int)
    absorb = fill(1:0, n_int)
    weighted = ties === :efron && any(b -> length(b) > 1, blocks)
    tw_dyads = weighted ? Vector{Vector{Int}}(undef, n_int) : nothing
    tw_val = weighted ? Vector{Float64}(undef, n_int) : nothing
    t_prev = float(t0)

    for block in blocks
        d = length(block)
        # Efron re-weights each tied CASE's contribution to ONE risk-set
        # denominator; a dyad that is its own competitor has no such weight
        # (and the naive one can go negative). Refuse rather than invent.
        tied_dyads = [(sorted[k].sender, sorted[k].receiver) for k in block]
        if ties === :efron && d > 1 && !allunique(tied_dyads)
            throw(ArgumentError(
                "ties=:efron requires the events tied at one timestamp to be " *
                "distinct dyads, but a dyad acts twice at t = " *
                "$(sorted[first(block)].time). Efron's correction re-weights each " *
                "tied case's contribution to ONE risk-set denominator, and a " *
                "risk-set member that is its own competitor has no such weight. " *
                "Use `ties=:breslow` (whose denominator is the plain risk-set " *
                "sum) or `ties=:ordered`."))
        end

        # Under a correction the whole block is read off the history as it stands
        # BEFORE any of the tied events, at the block's (single) timestamp; the
        # block is then absorbed as a whole, so no simultaneous event can enter
        # another's statistics.
        tb = float(sorted[first(block)].time)
        shared = freeze && d > 1
        tied_idx = weighted && d > 1 ?
            [dyad_index[dy] for dy in tied_dyads] : Int[]

        for (j, m) in enumerate(block)
            ev = sorted[m]
            ci = get(dyad_index, (ev.sender, ev.receiver), 0)
            ci > 0 || throw(ArgumentError("event $(m) references actors outside 1:$n_actors"))
            case_idx[m] = ci
            waiting[m] = float(ev.time - t_prev)
            t_prev = float(ev.time)
            read_time[m] = shared ? tb : float(ev.time)
            # Absorb the event now unless a tie correction is in force, in which
            # case the whole block is absorbed after its last interval.
            absorb[m] = freeze ? (m == last(block) ? block : (1:0)) : (m:m)
            if weighted
                tw_dyads[m] = tied_idx
                tw_val[m] = d > 1 ? 1.0 - (j - 1) / d : 1.0
            end
        end
    end

    # The right-censored tail: exposure from the last event to the end of
    # observation, with no event term (case_idx == 0).
    if t_end !== nothing
        m = n_int
        case_idx[m] = 0
        waiting[m] = float(t_end - t_prev)
        read_time[m] = float(t_end)
        if weighted
            tw_dyads[m] = Int[]
            tw_val[m] = 1.0
        end
    end

    return _RiskSetPlan(sorted, statistics, dyads, p, n_int, case_idx, waiting,
                        read_time, absorb, tw_dyads, tw_val)
end

# Interval m's design matrix, into `dest`, against `history` as it stands before
# the interval. A deterministic function of (history, read time): this is why the
# cached and streamed policies agree bit-for-bit.
function _fill_design!(dest::AbstractMatrix{Float64}, plan::_RiskSetPlan, history,
                       m::Int)
    t = plan.read_time[m]
    stats = plan.statistics
    @inbounds for (dd, (s, r)) in enumerate(plan.dyads)
        vals = map(stat -> compute(stat, history, s, r, t), stats)
        for k in 1:plan.p
            dest[dd, k] = vals[k]
        end
    end
    return dest
end

_absorb!(history, plan::_RiskSetPlan, m::Int) =
    (for k in plan.absorb[m]; update_history!(history, plan.sorted[k]); end; history)

# The Efron denominator weights of interval m, materialized into the reusable
# buffer `buf` — or `nothing` when no weight bites, which is every policy but a
# biting `:efron` and keeps the unweighted inner loop exactly as it was.
function _tie_weights!(buf::Vector{Float64}, plan::_RiskSetPlan, m::Int)
    plan.tw_dyads === nothing && return nothing
    fill!(buf, 1.0)
    w = plan.tw_val[m]
    @inbounds for d in plan.tw_dyads[m]
        buf[d] = w
    end
    return buf
end

abstract type _RiskSets end

# `cache=:all` — every design matrix materialized once. Fastest; O(E · n² · p).
struct _CachedRiskSets{T,S} <: _RiskSets
    plan::_RiskSetPlan{T,S}
    X::Vector{Matrix{Float64}}
    tw_buf::Vector{Float64}
end

# `cache=:chunked` / `:none` — a bounded cache of `length(buffers)` design
# matrices, refilled by replaying the interaction history from the start on every
# pass. `:none` is `chunk = 1`. Memory O(chunk · n² · p); the price is that the
# statistics are recomputed once per derivative evaluation instead of once.
struct _StreamedRiskSets{T,S} <: _RiskSets
    plan::_RiskSetPlan{T,S}
    buffers::Vector{Matrix{Float64}}
    history::InteractionHistory{T}
    tw_buf::Vector{Float64}
end

peak_design_bytes(rs::_CachedRiskSets) = _full_cache_bytes(rs.plan)
peak_design_bytes(rs::_StreamedRiskSets) =
    length(rs.buffers) * _design_bytes(rs.plan)

# Visit the intervals in time order, calling `f(m, Xm, twm)` on each: `Xm` is the
# risk-set design matrix and `twm` the Efron denominator weights (or `nothing`).
# The order is the same under every policy, so any accumulation over `f` is
# summed in the same order and lands on the same floating-point number.
function each_interval(f::F, rs::_CachedRiskSets) where F
    plan = rs.plan
    for m in 1:plan.n_int
        f(m, rs.X[m], _tie_weights!(rs.tw_buf, plan, m))
    end
    return nothing
end

function each_interval(f::F, rs::_StreamedRiskSets) where F
    plan = rs.plan
    history = _reset_history!(rs.history)
    k = length(rs.buffers)
    m = 1
    while m <= plan.n_int
        hi = min(m + k - 1, plan.n_int)
        # Fill the chunk, walking the history forward as we go...
        for j in m:hi
            _fill_design!(rs.buffers[j - m + 1], plan, history, j)
            _absorb!(history, plan, j)
        end
        # ...then consume it. (Tied intervals under a freeze policy share one
        # matrix in `:all`; here they are recomputed, which costs a little and
        # gives the identical values — the history is frozen and the read time is
        # the block's.)
        for j in m:hi
            f(j, rs.buffers[j - m + 1], _tie_weights!(rs.tw_buf, plan, j))
        end
        m = hi + 1
    end
    return nothing
end

function _reset_history!(history::InteractionHistory)
    empty!(history.events)
    empty!(history.sender_history)
    empty!(history.receiver_history)
    empty!(history.pair_history)
    empty!(history.event_counts)
    return history
end

# Resolve `cache=:auto` to a concrete policy and a concrete cache size, in design
# matrices. `chunk` defaults to as many as fit in `cache_bytes`; a chunk that
# covers every interval IS `:all` (same memory, but the streamed path would
# recompute every statistic on every pass for nothing), so it collapses to it.
# Returns `(mode, k)` with `k` the number of matrices alive at once, so the
# footprint a fit will pay is `k * _design_bytes(plan)` — computable without
# allocating any of it.
function _resolve_cache(plan::_RiskSetPlan; cache::Symbol=:auto,
                        chunk::Union{Nothing,Int}=nothing,
                        cache_bytes::Int=_DEFAULT_CACHE_BYTES)
    cache in CACHE_MODES || throw(ArgumentError(
        "cache must be one of $(CACHE_MODES), got :$cache. `:all` materializes " *
        "every risk-set design matrix (fastest, O(E·n²·p) memory), `:chunked` " *
        "keeps a bounded cache of `chunk` of them and recomputes the rest on " *
        "each pass, `:none` keeps exactly one, and `:auto` picks `:all` when " *
        "the projected footprint fits in `cache_bytes`."))
    chunk === nothing || chunk >= 1 ||
        throw(ArgumentError("chunk must be at least 1, got $chunk"))

    mode = cache
    mode === :auto &&
        (mode = _full_cache_bytes(plan) <= cache_bytes ? :all : :chunked)
    mode === :none && return (:none, 1)
    if mode === :chunked
        k = something(chunk, cache_bytes ÷ max(1, _design_bytes(plan)))
        k >= plan.n_int && return (:all, plan.n_int)
        return (:chunked, max(1, k))
    end
    return (:all, plan.n_int)
end

function _risk_sets(plan::_RiskSetPlan{T,S}; cache::Symbol=:auto,
                    chunk::Union{Nothing,Int}=nothing,
                    cache_bytes::Int=_DEFAULT_CACHE_BYTES) where {T,S}
    mode, k = _resolve_cache(plan; cache=cache, chunk=chunk,
                             cache_bytes=cache_bytes)
    D = n_dyads(plan)
    tw_buf = plan.tw_dyads === nothing ? Float64[] : Vector{Float64}(undef, D)

    if mode === :all
        history = InteractionHistory{T}()
        X = Vector{Matrix{Float64}}(undef, plan.n_int)
        m = 1
        while m <= plan.n_int
            Xm = Matrix{Float64}(undef, D, plan.p)
            _fill_design!(Xm, plan, history, m)
            X[m] = Xm
            # A frozen tie block reads ONE matrix off ONE history at ONE time:
            # share it rather than recomputing (and re-storing) it per event.
            hi = m
            while hi < plan.n_int && isempty(plan.absorb[hi])
                hi += 1
                X[hi] = Xm
            end
            for j in m:hi
                _absorb!(history, plan, j)
            end
            m = hi + 1
        end
        return _CachedRiskSets(plan, X, tw_buf)
    end

    buffers = [Matrix{Float64}(undef, D, plan.p) for _ in 1:k]
    return _StreamedRiskSets(plan, buffers, InteractionHistory{T}(), tw_buf)
end

# The eager 5-tuple the package was built on: `(dyads, case_idx, X, waiting, W)`
# with every design matrix and every dense Efron weight vector materialized.
# `cache=:all` in one call — kept because it is the clearest statement of what
# the risk sets ARE, and the tests read it directly.
_risk_set_stats(events::Vector{Event{T}}, statistics::AbstractVector, n_actors::Int;
                kwargs...) where T =
    _risk_set_stats(events, Tuple(statistics), n_actors; kwargs...)

function _risk_set_stats(events::Vector{Event{T}}, statistics::Tuple, n_actors::Int;
                         kwargs...) where T
    plan = _risk_set_plan(events, statistics, n_actors; kwargs...)
    rs = _risk_sets(plan; cache=:all)
    W = plan.tw_dyads === nothing ? nothing :
        [copy(_tie_weights!(Vector{Float64}(undef, n_dyads(plan)), plan, m))
         for m in 1:plan.n_int]
    return plan.dyads, plan.case_idx, rs.X, plan.waiting, W
end

# =============================================================================
# The two derivative closures (review finding 15)
# =============================================================================
#
# Both are `θ -> (ll, grad, hess)` for `_newton`, and both run on workspaces
# allocated ONCE here rather than per interval per Newton evaluation. What a
# single evaluation allocates is the O(p) gradient and O(p²) Hessian handed back
# to the optimizer — and nothing else: no `Xm * θ`, no `exp.(η)`, no
# `Xm' * (w .* Xm)` weighted copy of the risk-set design matrix, no per-event
# outer product. Pinned by an `@allocated` regression test, which is why they are
# named functions and not closures buried inside the fitters: the test measures
# the code that runs, not a copy of it.

# Ordinal (multinomial partial) likelihood over the full risk set.
function _obpm_derivatives(rs::_RiskSets)
    plan = rs.plan
    p = plan.p
    D = n_dyads(plan)
    case_idx = plan.case_idx
    η = Vector{Float64}(undef, D)
    probs = Vector{Float64}(undef, D)
    W = Matrix{Float64}(undef, D, p)
    x_exp = Vector{Float64}(undef, p)

    return function (θ)
        ll = Ref(0.0)
        grad = zeros(p)
        hess = zeros(p, p)
        # `twm === nothing` is the unweighted likelihood (every policy but a
        # biting `:efron`), left bit-for-bit as it was. Under Efron the
        # DENOMINATOR carries the tied cases' weights 1 − (j−1)/d; the
        # numerator is always the case's own exp(η).
        each_interval(rs) do m, Xm, twm
            mul!(η, Xm, θ)
            ηmax = maximum(η)
            Z = 0.0
            if twm === nothing
                @inbounds for d in 1:D
                    probs[d] = exp(η[d] - ηmax)
                    Z += probs[d]
                end
            else
                @inbounds for d in 1:D
                    probs[d] = twm[d] * exp(η[d] - ηmax)
                    Z += probs[d]
                end
            end
            probs ./= Z

            ll[] += η[case_idx[m]] - ηmax - log(Z)

            mul!(x_exp, transpose(Xm), probs)
            @inbounds for k in 1:p
                grad[k] += Xm[case_idx[m], k] - x_exp[k]
            end
            # -E[XX'] by gemm on the sqrt(prob)-weighted rows, +E[X]E[X]' by a
            # rank-1 update — both in place into `hess`.
            W .= sqrt.(probs) .* Xm
            mul!(hess, transpose(W), W, -1.0, 1.0)
            BLAS.ger!(1.0, x_exp, x_exp, hess)
        end
        return ll[], grad, hess
    end
end

# Exponential-baseline interval (exact-time) likelihood; β = (log λ₀, θ).
function _timing_derivatives(rs::_RiskSets)
    plan = rs.plan
    p = plan.p
    D = n_dyads(plan)
    case_idx, waiting = plan.case_idx, plan.waiting
    θbuf = Vector{Float64}(undef, p)
    η = Vector{Float64}(undef, D)
    w = Vector{Float64}(undef, D)
    WX = Matrix{Float64}(undef, D, p)
    Sx = Vector{Float64}(undef, p)
    Sxx = Matrix{Float64}(undef, p, p)

    return function (β)
        logλ = β[1]
        copyto!(θbuf, 1, β, 2, p)
        λ = exp(logλ)

        ll = Ref(0.0)
        grad = zeros(p + 1)
        hess = zeros(p + 1, p + 1)

        each_interval(rs) do m, Xm, _
            mul!(η, Xm, θbuf)
            w .= exp.(η)
            S = sum(w)                      # Σ exp(θ'x)
            mul!(Sx, transpose(Xm), w)      # Σ x exp(θ'x)
            WX .= w .* Xm
            mul!(Sxx, transpose(Xm), WX)    # Σ xx' exp(θ'x)
            Δt = waiting[m]
            ci = case_idx[m]                # 0 for the right-censored tail
            observed = ci > 0

            # Every interval contributes exposure; only an interval that
            # ends in an event contributes an event term.
            ll[] += (observed ? logλ + η[ci] : 0.0) - λ * Δt * S

            λΔt = λ * Δt
            grad[1] += (observed ? 1.0 : 0.0) - λΔt * S
            hess[1, 1] += -λΔt * S
            @inbounds for k in 1:p
                observed && (grad[k + 1] += Xm[ci, k])
                grad[k + 1] -= λΔt * Sx[k]
                hess[1, k + 1] += -λΔt * Sx[k]
                hess[k + 1, 1] += -λΔt * Sx[k]
                for l in 1:p
                    hess[l + 1, k + 1] -= λΔt * Sxx[l, k]
                end
            end
        end

        return ll[], grad, hess
    end
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
    # What was ACTUALLY done with tied event times: `:none` (the data had no
    # ties), or the policy that bit — `:ordered`, `:breslow`, `:efron`. `:error`
    # can never appear: under it a tie throws instead of fitting.
    tie_type::Symbol
end

OrdinalBPMResult(model, coefficients, std_errors, loglik, converged, n_events) =
    OrdinalBPMResult(model, coefficients, std_errors, loglik, converged, n_events,
                     :none)

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
    result.tie_type === :none ||
        println(io, "Tied event times: $(result.tie_type)")
    println(io)
    z, p = _wald_zp(result.coefficients, result.std_errors)
    print_coeftable(io, [name(stat) for stat in result.model.statistics],
                    result.coefficients, result.std_errors, p; z_values=z)
end

# ============================================================================
# The shared result-metadata protocol (Networks.jl `src/results.jl`)
# ============================================================================
#
# Both Relevent models enumerate the FULL risk set — this is the package's
# distinguishing property against `REM.fit_rem`, which samples controls — so
# their objectives ARE the exact likelihoods. `fit_metadata(fit)` makes that
# claim inspectable rather than a sentence in a docstring, and it names the one
# approximation that remains: tied timestamps are ordered arbitrarily.

estimand(::OrdinalBPMResult) = :relational_event

"""
    objective(::OrdinalBPMResult) -> Symbol

`:likelihood` — the ordinal relational-event likelihood: each event contributes a
multinomial-logit term over the **full risk set** of all `n(n−1)` dyads. Nothing
is sampled and nothing is approximated away.
"""
objective(::OrdinalBPMResult) = :likelihood

"""
    is_exact(result::OrdinalBPMResult) -> Bool

`true` on strictly ordered data. The objective IS the exact ordinal likelihood of
the model (full risk set, no case-control sampling — compare `REM.fit_rem`, whose
default sampled risk set makes the same kind of objective an approximation),
maximized by Newton-Raphson to a stationary point.

`false` when the data carried **tied timestamps** (`tie_type != :none`): the
ordinal likelihood is a likelihood over an order the tied data does not
determine, and every policy that lets the fit proceed — `:ordered`, `:breslow`,
`:efron` — is an approximation to it.
"""
is_exact(result::OrdinalBPMResult) = result.tie_type === :none

se_method(::OrdinalBPMResult) = :hessian

missing_method(::OrdinalBPMResult) = :none

"""
    tie_method(result::OrdinalBPMResult) -> Symbol

What was ACTUALLY done with tied event times: `:none` (the data had none),
`:ordered`, `:breslow` or `:efron`. `:error` — the default policy — never
appears, because under it a tie throws instead of fitting. See `ties=` in
[`fit_obpm`](@ref).
"""
tie_method(result::OrdinalBPMResult) = result.tie_type

# Prose for the tie policy that actually bit; `nothing` when the data had no
# ties, since a correction on tie-free data corrected nothing.
function _tie_approximation(tie_type::Symbol)
    if tie_type === :ordered
        return "tied event times were ordered arbitrarily with NO tie correction " *
               "(`ties=:ordered`): the ordinal likelihood is a likelihood over the " *
               "ORDER of events, and the event placed first also enters the " *
               "statistics of the events placed after it, so the estimate depends " *
               "on a sort the data does not determine"
    elseif tie_type === :breslow
        return "tied event times were handled by the BRESLOW correction " *
               "(`ties=:breslow`): the tied events share one risk set (statistics " *
               "frozen across the tie) and each contributes the same denominator. " *
               "An approximation to the average over the d! orderings, and the " *
               "cruder of the two — it biases coefficients toward zero as ties get " *
               "heavier (`ties=:efron` is the better approximation)"
    elseif tie_type === :efron
        return "tied event times were handled by the EFRON correction " *
               "(`ties=:efron`): the tied cases enter the j-th denominator with " *
               "weight 1 − (j−1)/d. A close approximation to the average over the " *
               "d! orderings — what `survival::coxph` defaults to — but the order " *
               "of simultaneous events remains unobserved"
    elseif tie_type === :batch
        return "tied event times were read as a simultaneous BATCH (`ties=:batch`): " *
               "the tied events could not influence one another (statistics frozen " *
               "across the block) and the block consumes ONE exposure interval. " *
               "This is the coarsened-observation model, NOT the continuous-time " *
               "model the likelihood otherwise assumes — under which a tie has " *
               "probability zero"
    end
    return nothing
end

function approximations(result::OrdinalBPMResult)
    out = String[]
    result.converged ||
        push!(out, "the Newton-Raphson maximization did NOT converge: the reported " *
                   "estimates are not a stationary point of the likelihood")
    tie_note = _tie_approximation(result.tie_type)
    isnothing(tie_note) || push!(out, tie_note)
    return out
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
    fit_obpm(events, statistics, n_actors; ties=:error, cache=:auto, chunk=nothing,
             maxiter=100, tol=1e-8) -> OrdinalBPMResult

Fit the ordinal relational event model by exact maximum likelihood over
the full risk set (Newton-Raphson with step-halving).

# Risk-set caching (issue Relevent#2)

The full risk set means an `n(n−1) × p` design matrix per event. Materializing
them all costs `O(E · n² · p)` doubles — 8 GB at `(n, E, p) = (100, 2000, 6)` —
so `cache=` decides how many are alive at once:

- `:auto` (default) — `:all` while the projected footprint fits in `cache_bytes`
  (256 MiB), `:chunked` above it.
- `:all` — every design matrix materialized once. Fastest; `O(E · n² · p)`.
- `:chunked` — a bounded cache of `chunk` matrices (default: as many as fit in
  `cache_bytes`), refilled by replaying the event history on each pass.
- `:none` — `:chunked` with `chunk = 1`: one matrix alive at a time, least
  memory, most recomputation.

Every policy visits the intervals in the same order and reads the same
statistics off the same histories, so **the fits are bit-identical** — asserted
in the tests. Only memory and time differ.

# Tied event times (issue Relevent#1, review finding 12)

This likelihood is a likelihood **over the order of the events** — it uses
nothing else. A tied timestamp therefore means the very thing being modelled is
unobserved, and sorting the tie invents it (worse: the statistics are read off
the pre-event history, so the event sorted first enters the *statistics* of the
one sorted second). `ties=` (the shared `Networks.TIE_POLICIES` vocabulary) says
what to do about it:

- `:error` (default) — name the tie and refuse. A user with tied data is told,
  not handed a number that depends on an arbitrary sort.
- `:ordered` — sequence order, no correction (what the package did before).
- `:breslow` — the Breslow correction: the tied events share one risk set (the
  history is frozen across the tie block, so they cannot enter each other's
  statistics) and each contributes the same denominator.
- `:efron` — the Efron correction: as `:breslow`, plus the `1 − (j−1)/d`
  denominator weights on the tied cases; the better approximation, and what
  `survival::coxph` defaults to. Requires the tied events to be distinct dyads.
- `:batch` — refused: with the history frozen, a simultaneous batch in an ordinal
  likelihood IS the Breslow correction. (It is `fit_timing`'s policy, where there
  is an exposure interval for a batch to consume.)

On tie-free data all four give the identical fit. `tie_method(fit)` reports what
actually happened and `approximations(fit)` carries the caveat.
"""
function fit_obpm(events::Vector{Event{T}}, statistics::Vector{<:AbstractStatistic},
                  n_actors::Int; ties::Symbol=:error, cache::Symbol=:auto,
                  chunk::Union{Nothing,Int}=nothing,
                  cache_bytes::Int=_DEFAULT_CACHE_BYTES, maxiter::Int=100,
                  tol::Float64=1e-8) where T
    check_tie_policy(ties, _OBPM_TIES_SUPPORTED; model=_OBPM_TIES_MODEL,
                     reasons=_OBPM_TIES_REASONS)
    model = OrdinalBPM(collect(AbstractStatistic, statistics), n_actors)
    p = length(statistics)
    isempty(events) && throw(ArgumentError("no events to fit"))

    sorted = sort(events, by=e -> e.time)
    blocks = _tie_blocks(sorted)
    has_ties = any(b -> length(b) > 1, blocks)
    ties === :error && has_ties && _reject_ties(sorted, blocks,
        "The ordinal likelihood is a likelihood over the ORDER of the events, " *
        "and a tie is precisely the statement that the order is unobserved: " *
        "sorting it would invent the very thing being modelled (and would let " *
        "whichever event is placed first enter the statistics of the ones placed " *
        "after it).",
        "Choose a policy explicitly: `ties=:efron` (the Efron correction, the " *
        "best approximation and `survival::coxph`'s default), `ties=:breslow` " *
        "(the Breslow correction), or `ties=:ordered` (the legacy behaviour — " *
        "arbitrary order, no correction).")
    tie_applied = has_ties ? ties : :none

    plan = _risk_set_plan(events, model.statistics, n_actors; ties=ties)
    rs = _risk_sets(plan; cache=cache, chunk=chunk, cache_bytes=cache_bytes)

    θ, se, ll, converged = _newton(_obpm_derivatives(rs), p; maxiter=maxiter,
                                   tol=tol)
    return OrdinalBPMResult(model, θ, se, ll, converged, length(events), tie_applied)
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
    # What was ACTUALLY done with tied event times: `:none` (the data had none —
    # which is what a continuous-time model expects), `:ordered` or `:batch`.
    tie_type::Symbol
end

TimingModelResult(model, coefficients, baseline_params, std_errors, loglik, converged) =
    TimingModelResult(model, coefficients, baseline_params, std_errors, loglik,
                      converged, :none)

function Base.show(io::IO, result::TimingModelResult)
    println(io, "Timing Model Results")
    println(io, "====================")
    println(io, "Baseline: $(result.model.baseline)")
    println(io, "Baseline rate λ₀: $(round(result.baseline_params[1], digits=6))")
    println(io, "Log-likelihood: $(round(result.loglik, digits=4))")
    println(io, "Converged: $(result.converged)")
    result.tie_type === :none ||
        println(io, "Tied event times: $(result.tie_type) " *
                    "(a continuous-time model gives a tie probability zero)")
    println(io)
    z, p = _wald_zp(result.coefficients, result.std_errors)
    print_coeftable(io, [name(stat) for stat in result.model.statistics],
                    result.coefficients, result.std_errors, p; z_values=z)
end

# The shared result-metadata protocol (see the OrdinalBPMResult block above).

estimand(::TimingModelResult) = :relational_event_timing

"""
    objective(::TimingModelResult) -> Symbol

`:likelihood` — the exact interval-timing likelihood over the **full risk set**:
`ℓ(λ₀, θ) = Σ_m [log λ₀ + θ'x_case − λ₀ Δt_m Σ_{ij} exp(θ'x_ij)]`, with the
per-dyad hazards of every dyad entering each waiting-time term. `(log λ₀, θ)` are
maximized jointly.
"""
objective(::TimingModelResult) = :likelihood

"""
    is_exact(result::TimingModelResult) -> Bool

`true` on data with no tied event times. The objective IS the exact likelihood of
the fitted model — the exponential-baseline proportional-hazards process with
statistics held constant between events (that piecewise-constant hazard is the
model's definition, not an approximation of it). Weibull/Gompertz baselines are
not fitted at all, so no `TimingModelResult` can carry an approximate likelihood.

`false` when the data carried **ties** (`tie_type != :none`). This is the sharp
case in the ecosystem: under a continuous-time model a tie has probability
**zero**, so tied data does not merely strain the likelihood, it contradicts the
process. Whatever the policy then computes (`:ordered` — a zero-length waiting
interval; `:batch` — a coarsened simultaneous batch) is the likelihood of a
*different* model from the one the user asked for.
"""
is_exact(result::TimingModelResult) = result.tie_type === :none

se_method(::TimingModelResult) = :hessian

missing_method(::TimingModelResult) = :none

"""
    tie_method(result::TimingModelResult) -> Symbol

What was ACTUALLY done with tied event times: `:none` (the data had none, which
is what a continuous-time model expects), `:ordered` (each tied event after the
first enters with a zero-length waiting interval, in an arbitrary order, while
still updating the history) or `:batch` (the tie is one simultaneous batch:
statistics frozen across it, one exposure interval for the block). `:error` — the
default — never appears: under it a tie throws. Breslow and Efron are not
available here at all; see `ties=` in [`fit_timing`](@ref).
"""
tie_method(result::TimingModelResult) = result.tie_type

function approximations(result::TimingModelResult)
    out = String[]
    result.converged ||
        push!(out, "the Newton-Raphson maximization did NOT converge: the reported " *
                   "estimates are not a stationary point of the likelihood")
    if result.tie_type === :ordered
        push!(out, "tied event times were ordered arbitrarily with NO correction " *
                   "(`ties=:ordered`): each tied event after the first enters as a " *
                   "ZERO-LENGTH waiting interval — it contributes an event term with " *
                   "no exposure — while still updating the statistics of the next, " *
                   "i.e. the fit claims one event caused another in no time at all. " *
                   "Under the continuous-time model being fitted, a tie has " *
                   "probability zero")
    elseif result.tie_type === :batch
        push!(out, "tied event times were read as a simultaneous BATCH " *
                   "(`ties=:batch`): the tied events could not have influenced one " *
                   "another (statistics frozen across the block) and the block " *
                   "consumes ONE exposure interval. This is the likelihood of a " *
                   "COARSENED observation process, not of the continuous-time model " *
                   "the exponential likelihood otherwise assumes — under which a tie " *
                   "has probability zero")
    end
    return out
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
               t0=zero(T), cache=:auto, chunk=nothing, maxiter=100, tol=1e-8)
        -> TimingModelResult

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

`t_end` is the observation *offset*, i.e. the time recording stopped. It
adds the right-censored final interval `[t_M, t_end]` — exposure with no
event — to the likelihood:

    ℓ += − λ₀ (t_end − t_M) Σ_{ij} exp(θ'x_ij).

`relevent::rem.dyad` **always** has this term (its last edgelist row is
the termination time, and any event on it is ignored), so `t_end` is
required to reproduce it: with `t_end = nothing` (the default) the
sequence is treated as ending at its last event, and the estimated
baseline rate λ₀ is biased upward, because the observation window is
implicitly shortened to exclude the eventless tail.

!!! note "Golden fixture"
    `test/fixtures/relevent_rem_dyad.toml` freezes `rem.dyad`'s temporal fit
    and pins both facts: with `t_end` the two agree to ~1e-6, and without
    it λ₀ differs in the second decimal.

# Risk-set caching (issue Relevent#2)

`cache=:auto|:all|:chunked|:none` (with `chunk` and `cache_bytes`) bounds the
memory the `E × n(n−1) × p` risk-set design matrices take, exactly as in
[`fit_obpm`](@ref) — same policies, same defaults, and the fit is bit-identical
under all of them.

# Tied event times (issue Relevent#1, review finding 12)

This is an **exact-time** likelihood. Under the continuous-time process it fits,
two events at one instant have probability **zero**: a tie is not an ambiguity in
the ordering, it is the model's own assumption failing. It is therefore either a
measurement artifact (a coarse clock) or a genuinely batched observation, and the
policies (`Networks.TIE_POLICIES`) say which:

- `ties=:error` (default) — name the tie and refuse.
- `ties=:ordered` — the legacy behaviour: each tied event after the first enters
  with a **zero-length waiting interval** (an event term with no exposure) while
  still updating the statistics of the next — the fit then claims one event
  caused another in no time at all.
- `ties=:batch` — read the tie as one simultaneous batch: the statistics are
  frozen across the block (no tied event can have influenced another) and the
  block consumes **one** exposure interval. This is the coarsened-observation
  reading, and the one to prefer when the clock is simply coarse.
- `ties=:breslow` / `ties=:efron` — **refused**, and not as a matter of taste:
  Breslow and Efron correct a *partial* likelihood, in which the baseline hazard
  is profiled out and only the order of events survives. There is no such
  denominator here. Fit `fit_obpm(...; ties=:efron)` if the order is what
  matters.

On tie-free data every policy gives the identical fit. `tie_method(fit)` reports
what happened; `is_exact(fit)` turns **false** as soon as a tie was in the data,
because no policy can make an exact-time likelihood exact for data the exact-time
model says cannot occur.
"""
function fit_timing(events::Vector{Event{T}}, statistics::Vector{<:AbstractStatistic},
                    n_actors::Int; baseline::Symbol=:exponential,
                    t0::T=zero(T), t_end::Union{Nothing,T}=nothing,
                    ties::Symbol=:error, cache::Symbol=:auto,
                    chunk::Union{Nothing,Int}=nothing,
                    cache_bytes::Int=_DEFAULT_CACHE_BYTES,
                    maxiter::Int=100, tol::Float64=1e-8) where T
    check_tie_policy(ties, _TIMING_TIES_SUPPORTED; model=_TIMING_TIES_MODEL,
                     reasons=_TIMING_TIES_REASONS)
    model = TimingModel(collect(AbstractStatistic, statistics); baseline=baseline)
    baseline == :exponential ||
        error("fit_timing implements the exponential-baseline likelihood; " *
              ":$baseline is available only for hazard_rate/survival_function")
    isempty(events) && throw(ArgumentError("no events to fit"))

    sorted = sort(events, by=e -> e.time)
    blocks = _tie_blocks(sorted)
    has_ties = any(b -> length(b) > 1, blocks)
    ties === :error && has_ties && _reject_ties(sorted, blocks,
        "This is an EXACT-TIME likelihood: under the continuous-time process it " *
        "fits, two events at one instant have probability ZERO, so a tie is not " *
        "an ambiguous ordering but the model's own assumption failing — the clock " *
        "is coarse, or the events are genuinely simultaneous.",
        "Say which: `ties=:batch` (a coarsened simultaneous batch — the tied " *
        "events cannot influence one another and the block consumes one exposure " *
        "interval) or `ties=:ordered` (the legacy behaviour — each tied event " *
        "after the first enters with a zero-length waiting interval). Breslow and " *
        "Efron do not apply to an exact-time likelihood; use " *
        "`fit_obpm(...; ties=:efron)` if only the order matters.")
    tie_applied = has_ties ? ties : :none

    p = length(statistics)
    plan = _risk_set_plan(events, model.statistics, n_actors;
                          t0=t0, t_end=t_end, ties=ties)
    rs = _risk_sets(plan; cache=cache, chunk=chunk, cache_bytes=cache_bytes)
    derivatives = _timing_derivatives(rs)

    β, se, ll, converged = _newton(derivatives, p + 1; maxiter=maxiter, tol=tol)

    λ0 = exp(β[1])
    return TimingModelResult(model, β[2:end], [λ0], se[2:end], ll, converged,
                             tie_applied)
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
`ties`, the risk-set cache policy `cache`/`chunk`/`cache_bytes`, and for the
interval likelihood `t0` and `t_end`) are forwarded to the underlying fitter.

`ties` defaults to `:error` in both, and the two accept **different** policies —
`:breslow`/`:efron` are defined only for the ordinal partial likelihood,
`:batch` only for the exact-time one — so a policy forwarded to the wrong
likelihood is refused with an explanation rather than quietly ignored. See
[`fit_obpm`](@ref) and [`fit_timing`](@ref).

# Example
```julia
fit_relevent(events, [PShift(:AB_BA), CovSnd(z)], n_actors)                # ordinal
fit_relevent(events, [PShift(:AB_BA)], n_actors; ordinal=false, t0=0.0,
             t_end=120.0)                                                  # timing
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
