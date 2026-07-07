"""
    Relevent.jl - Relational Event Models (Additional Features)

Extends REM.jl with features from R `relevent`'s `rem.dyad`:

- decay-weighted interaction-history statistics that plug directly into
  `REM.fit_rem` (they implement REM's 4-argument `compute` interface) and
  into this package's own full-risk-set estimators;
- **ordinal** estimation (`fit_obpm`): the exact multinomial partial
  likelihood over the full risk set (no case-control sampling);
- **interval-timing** estimation (`fit_timing`): the exponential-baseline
  proportional-hazards likelihood using the observed waiting times;
- baseline hazard/survival functions (exponential, Weibull, Gompertz) and
  cumulative decayed network state.

Port of the R relevent package from the StatNet collection.
"""
module Relevent

using LinearAlgebra
using REM
using Statistics

import REM: compute, name

# Additional REM statistics
export InteractionHistory, PriorInteraction
export SendingCapacity, ReceivingCapacity
export LocalInertia, Momentum

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
# Advanced REM Statistics
# =============================================================================
#
# Each statistic implements BOTH interfaces:
#
#   compute(stat, history::InteractionHistory, sender, receiver, current_time)
#       — used by this package's full-risk-set estimators;
#   compute(stat, state::REM.NetworkState, sender, receiver)
#       — REM.jl's interface, so these statistics work inside REM.fit_rem
#         and generate_observations.
#
# Decay is half-life parameterized: decay = log(2)/halflife.

_decay_sum(times, current_time, decay) =
    sum(exp(-decay * float(current_time - t)) for t in times; init=0.0)

"""
    PriorInteraction(halflife; direction=:outgoing) <: AbstractStatistic

Decayed count of prior interactions on the dyad (`:outgoing` = sender→
receiver events, `:incoming` = receiver→sender, `:both` = their sum).
"""
struct PriorInteraction <: AbstractStatistic
    halflife::Float64
    direction::Symbol

    function PriorInteraction(halflife::Float64; direction::Symbol=:outgoing)
        direction in (:outgoing, :incoming, :both) ||
            throw(ArgumentError("direction must be :outgoing, :incoming, or :both"))
        halflife > 0 || throw(ArgumentError("halflife must be positive"))
        new(halflife, direction)
    end
end

name(stat::PriorInteraction) = "prior_interaction_$(stat.direction)"

function compute(stat::PriorInteraction, history::InteractionHistory{T},
                 sender::Int, receiver::Int, current_time::T) where T
    decay = log(2) / stat.halflife
    value = 0.0

    if stat.direction in (:outgoing, :both)
        value += _decay_sum(get(history.pair_history, (sender, receiver), T[]),
                            current_time, decay)
    end
    if stat.direction in (:incoming, :both)
        value += _decay_sum(get(history.pair_history, (receiver, sender), T[]),
                            current_time, decay)
    end

    return value
end

function compute(stat::PriorInteraction, state::REM.NetworkState,
                 sender::Int, receiver::Int)
    decay = log(2) / stat.halflife
    t_now = state.current_time
    value = 0.0
    for (s, r, t, _) in state.event_history
        if stat.direction in (:outgoing, :both) && s == sender && r == receiver
            value += exp(-decay * float(t_now - t))
        end
        if stat.direction in (:incoming, :both) && s == receiver && r == sender
            value += exp(-decay * float(t_now - t))
        end
    end
    return value
end

"""
    SendingCapacity(halflife) <: AbstractStatistic

Decayed count of ALL of the sender's past outgoing events (sender
activity), regardless of receiver.
"""
struct SendingCapacity <: AbstractStatistic
    halflife::Float64
end

name(::SendingCapacity) = "sending_capacity"

function compute(stat::SendingCapacity, history::InteractionHistory{T},
                 sender::Int, receiver::Int, current_time::T) where T
    decay = log(2) / stat.halflife
    value = 0.0
    # Sum over every dyad the sender has used, over every event time
    for r in unique(get(history.sender_history, sender, Int[]))
        value += _decay_sum(get(history.pair_history, (sender, r), T[]),
                            current_time, decay)
    end
    return value
end

function compute(stat::SendingCapacity, state::REM.NetworkState,
                 sender::Int, receiver::Int)
    decay = log(2) / stat.halflife
    t_now = state.current_time
    value = 0.0
    for (s, _, t, _) in state.event_history
        s == sender && (value += exp(-decay * float(t_now - t)))
    end
    return value
end

"""
    ReceivingCapacity(halflife) <: AbstractStatistic

Decayed count of ALL of the receiver's past incoming events (receiver
popularity).
"""
struct ReceivingCapacity <: AbstractStatistic
    halflife::Float64
end

name(::ReceivingCapacity) = "receiving_capacity"

function compute(stat::ReceivingCapacity, history::InteractionHistory{T},
                 sender::Int, receiver::Int, current_time::T) where T
    decay = log(2) / stat.halflife
    value = 0.0
    for s in unique(get(history.receiver_history, receiver, Int[]))
        value += _decay_sum(get(history.pair_history, (s, receiver), T[]),
                            current_time, decay)
    end
    return value
end

function compute(stat::ReceivingCapacity, state::REM.NetworkState,
                 sender::Int, receiver::Int)
    decay = log(2) / stat.halflife
    t_now = state.current_time
    value = 0.0
    for (_, r, t, _) in state.event_history
        r == receiver && (value += exp(-decay * float(t_now - t)))
    end
    return value
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

function compute(stat::LocalInertia, state::REM.NetworkState,
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

    Momentum(halflife::Float64; normalize::Bool=false) = new(halflife, normalize)
end

name(::Momentum) = "momentum"

function compute(stat::Momentum, history::InteractionHistory{T},
                 sender::Int, receiver::Int, current_time::T) where T
    decay = log(2) / stat.halflife
    value = 0.0
    n_sender = 0
    for r in unique(get(history.sender_history, sender, Int[]))
        times = get(history.pair_history, (sender, r), T[])
        n_sender += length(times)
        value += _decay_sum(times, current_time, decay)
    end
    (stat.normalize && n_sender > 0) && (value /= n_sender)
    return value
end

function compute(stat::Momentum, state::REM.NetworkState,
                 sender::Int, receiver::Int)
    decay = log(2) / stat.halflife
    t_now = state.current_time
    value = 0.0
    n_sender = 0
    for (s, _, t, _) in state.event_history
        s == sender || continue
        n_sender += 1
        value += exp(-decay * float(t_now - t))
    end
    (stat.normalize && n_sender > 0) && (value /= n_sender)
    return value
end

# =============================================================================
# Full-risk-set machinery shared by the estimators
# =============================================================================

# For each event (in time order): the case's statistic vector and the
# matrix of statistics for every dyad in the full risk set. History is
# strictly pre-event (no look-ahead).
function _risk_set_stats(events::Vector{Event{T}}, statistics, n_actors::Int) where T
    sorted = sort(events, by=e -> e.time)
    if length(sorted) > 1 && !allunique(e.time for e in sorted)
        @warn "Event sequence contains tied timestamps; ties are ordered " *
              "arbitrarily and no tie correction is applied" maxlog = 1
    end

    p = length(statistics)
    dyads = [(s, r) for s in 1:n_actors for r in 1:n_actors if s != r]
    history = InteractionHistory{T}()

    case_idx = Vector{Int}(undef, length(sorted))
    X = Vector{Matrix{Float64}}(undef, length(sorted))
    waiting = Vector{Float64}(undef, length(sorted))
    t_prev = length(sorted) > 0 ? zero(sorted[1].time) : zero(T)

    for (m, ev) in enumerate(sorted)
        Xm = Matrix{Float64}(undef, length(dyads), p)
        ci = 0
        for (d, (s, r)) in enumerate(dyads)
            for (k, stat) in enumerate(statistics)
                Xm[d, k] = compute(stat, history, s, r, ev.time)
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

function Base.show(io::IO, result::OrdinalBPMResult)
    println(io, "Ordinal Butts-Park Model Results")
    println(io, "================================")
    println(io, "N actors: $(result.model.n_actors)")
    println(io, "N events: $(result.n_events)")
    println(io, "Log-likelihood: $(round(result.loglik, digits=4))")
    println(io, "Converged: $(result.converged)")
    println(io)
    println(io, "Coefficients:")
    for (i, stat) in enumerate(result.model.statistics)
        println(io, "  $(rpad(name(stat), 28)) $(lpad(round(result.coefficients[i], digits=4), 10)) " *
                    "(SE: $(round(result.std_errors[i], digits=4)))")
    end
end

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

    function derivatives(θ)
        ll = 0.0
        grad = zeros(p)
        hess = zeros(p, p)
        for m in 1:M
            Xm = X[m]
            η = Xm * θ
            ηmax = maximum(η)
            w = exp.(η .- ηmax)
            Z = sum(w)
            probs = w ./ Z

            ll += η[case_idx[m]] - ηmax - log(Z)

            x_exp = Xm' * probs
            grad .+= Xm[case_idx[m], :] .- x_exp
            hess .-= Xm' * (probs .* Xm) .- x_exp * x_exp'
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
    println(io, "Coefficients:")
    for (i, stat) in enumerate(result.model.statistics)
        println(io, "  $(rpad(name(stat), 28)) $(lpad(round(result.coefficients[i], digits=4), 10)) " *
                    "(SE: $(round(result.std_errors[i], digits=4)))")
    end
end

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
               maxiter=100, tol=1e-8) -> TimingModelResult

Fit the interval-timing relational event model with an exponential
baseline by exact maximum likelihood: with waiting time `Δt_m` before
event m and per-dyad hazards `λ₀·exp(θ'x_ij)` (statistics frozen between
events),

    ℓ(λ₀, θ) = Σ_m [ log λ₀ + θ'x_case − λ₀ Δt_m Σ_{ij} exp(θ'x_ij) ].

`(log λ₀, θ)` are estimated jointly by Newton-Raphson; standard errors
come from the observed information. Weibull/Gompertz baselines are not
fitted (an informative error is raised); use them with
[`hazard_rate`](@ref)/[`survival_function`](@ref).
"""
function fit_timing(events::Vector{Event{T}}, statistics::Vector{<:AbstractStatistic},
                    n_actors::Int; baseline::Symbol=:exponential,
                    maxiter::Int=100, tol::Float64=1e-8) where T
    model = TimingModel(collect(AbstractStatistic, statistics); baseline=baseline)
    baseline == :exponential ||
        error("fit_timing implements the exponential-baseline likelihood; " *
              ":$baseline is available only for hazard_rate/survival_function")
    isempty(events) && throw(ArgumentError("no events to fit"))

    p = length(statistics)
    _, case_idx, X, waiting = _risk_set_stats(events, model.statistics, n_actors)
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
