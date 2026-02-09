"""
    Relevent.jl - Relational Event Models (Additional Features)

Provides additional relational event model features complementing REM.jl,
including ordinal timing models (BPM), interaction history tracking,
and advanced REM statistics.

Port of the R relevent package from the StatNet collection.
"""
module Relevent

using DataFrames
using Distributions
using LinearAlgebra
using Network
using Optim
using REM
using Random
using Statistics
using StatsBase

# Additional REM statistics
export InteractionHistory, PriorInteraction
export SendingCapacity, ReceivingCapacity
export LocalInertia, Momentum

# Ordinal models
export OrdinalBPM, fit_obpm, OrdinalBPMResult
export rank_events, ordinal_likelihood

# Timing models
export TimingModel, fit_timing, TimingModelResult
export hazard_rate, survival_function

# History tracking
export EventHistory, update_history!
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

    # Update sender history
    senders = get!(history.sender_history, event.sender, Int[])
    push!(senders, event.receiver)

    # Update receiver history
    receivers = get!(history.receiver_history, event.receiver, Int[])
    push!(receivers, event.sender)

    # Update pair history
    pair = (event.sender, event.receiver)
    times = get!(history.pair_history, pair, T[])
    push!(times, event.time)

    # Update counts
    history.event_counts[pair] = get(history.event_counts, pair, 0) + 1

    return history
end

"""
    get_interaction_count(history::InteractionHistory, sender, receiver) -> Int

Get the number of times sender has sent to receiver.
"""
function get_interaction_count(history::InteractionHistory, sender::Int, receiver::Int)
    return get(history.event_counts, (sender, receiver), 0)
end

"""
    get_last_interaction(history::InteractionHistory, sender, receiver) -> Union{T, Nothing}

Get the time of the last interaction from sender to receiver.
"""
function get_last_interaction(history::InteractionHistory{T}, sender::Int, receiver::Int) where T
    times = get(history.pair_history, (sender, receiver), T[])
    return isempty(times) ? nothing : times[end]
end

# =============================================================================
# Advanced REM Statistics
# =============================================================================

"""
    PriorInteraction <: AbstractStatistic

Statistic for prior interaction between sender and receiver.
Uses half-life decay.

# Fields
- `halflife::Float64`: Half-life for decay
- `direction::Symbol`: :outgoing, :incoming, or :both
"""
struct PriorInteraction <: AbstractStatistic
    halflife::Float64
    direction::Symbol

    function PriorInteraction(halflife::Float64; direction::Symbol=:outgoing)
        direction in (:outgoing, :incoming, :both) ||
            throw(ArgumentError("direction must be :outgoing, :incoming, or :both"))
        new(halflife, direction)
    end
end

function compute(stat::PriorInteraction, history::InteractionHistory{T},
                 sender::Int, receiver::Int, current_time::T) where T
    decay = log(2) / stat.halflife
    value = 0.0

    if stat.direction in (:outgoing, :both)
        times = get(history.pair_history, (sender, receiver), T[])
        for t in times
            value += exp(-decay * (current_time - t))
        end
    end

    if stat.direction in (:incoming, :both)
        times = get(history.pair_history, (receiver, sender), T[])
        for t in times
            value += exp(-decay * (current_time - t))
        end
    end

    return value
end

"""
    SendingCapacity <: AbstractStatistic

Measures the sender's sending capacity based on past activity.
"""
struct SendingCapacity <: AbstractStatistic
    halflife::Float64
end

function compute(stat::SendingCapacity, history::InteractionHistory{T},
                 sender::Int, receiver::Int, current_time::T) where T
    decay = log(2) / stat.halflife
    count = 0.0

    receivers = get(history.sender_history, sender, Int[])
    pair_times = history.pair_history

    for (i, r) in enumerate(receivers)
        times = get(pair_times, (sender, r), T[])
        if !isempty(times) && i <= length(times)
            count += exp(-decay * (current_time - times[i]))
        end
    end

    return count
end

"""
    ReceivingCapacity <: AbstractStatistic

Measures the receiver's receiving capacity based on past activity.
"""
struct ReceivingCapacity <: AbstractStatistic
    halflife::Float64
end

function compute(stat::ReceivingCapacity, history::InteractionHistory{T},
                 sender::Int, receiver::Int, current_time::T) where T
    decay = log(2) / stat.halflife
    count = 0.0

    senders = get(history.receiver_history, receiver, Int[])
    pair_times = history.pair_history

    for (i, s) in enumerate(senders)
        times = get(pair_times, (s, receiver), T[])
        if !isempty(times) && i <= length(times)
            count += exp(-decay * (current_time - times[i]))
        end
    end

    return count
end

"""
    LocalInertia <: AbstractStatistic

Tendency for repeat interactions (same dyad).
"""
struct LocalInertia <: AbstractStatistic
    halflife::Float64
end

function compute(stat::LocalInertia, history::InteractionHistory{T},
                 sender::Int, receiver::Int, current_time::T) where T
    decay = log(2) / stat.halflife
    times = get(history.pair_history, (sender, receiver), T[])

    isempty(times) && return 0.0
    return exp(-decay * (current_time - times[end]))
end

"""
    Momentum <: AbstractStatistic

Sender's overall activity momentum.
"""
struct Momentum <: AbstractStatistic
    halflife::Float64
    normalize::Bool

    Momentum(halflife::Float64; normalize::Bool=false) = new(halflife, normalize)
end

function compute(stat::Momentum, history::InteractionHistory{T},
                 sender::Int, receiver::Int, current_time::T) where T
    decay = log(2) / stat.halflife

    count = 0.0
    receivers_list = get(history.sender_history, sender, Int[])

    for r in unique(receivers_list)
        times = get(history.pair_history, (sender, r), T[])
        for t in times
            count += exp(-decay * (current_time - t))
        end
    end

    if stat.normalize && count > 0
        count /= length(history.events)
    end

    return count
end

# =============================================================================
# Ordinal Butts-Park Model
# =============================================================================

"""
    OrdinalBPM

Ordinal Butts-Park Model for relational events.
Models the relative ordering of events rather than exact timing.

# Fields
- `statistics::Vector{AbstractStatistic}`: Statistics for the model
- `n_actors::Int`: Number of actors
"""
struct OrdinalBPM
    statistics::Vector{AbstractStatistic}
    n_actors::Int

    function OrdinalBPM(statistics::Vector{<:AbstractStatistic}, n::Int)
        new(statistics, n)
    end
end

"""
    OrdinalBPMResult

Results from fitting an Ordinal BPM.
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
        stat_name = string(typeof(stat).name.name)
        println(io, "  $(rpad(stat_name, 25)) $(lpad(round(result.coefficients[i], digits=4), 10)) " *
                    "(SE: $(round(result.std_errors[i], digits=4)))")
    end
end

"""
    rank_events(events::Vector{Event}) -> Vector{Int}

Convert events to ordinal ranks (1 = first event, 2 = second, etc.).
"""
function rank_events(events::Vector{Event{T}}) where T
    sorted_indices = sortperm([e.time for e in events])
    ranks = zeros(Int, length(events))
    for (rank, idx) in enumerate(sorted_indices)
        ranks[idx] = rank
    end
    return ranks
end

logaddexp(a, b) = a > b ? a + log1p(exp(b - a)) : b + log1p(exp(a - b))

"""
    fit_obpm(events, statistics, n_actors; kwargs...) -> OrdinalBPMResult

Fit an ordinal Butts-Park model.
"""
function fit_obpm(events::Vector{Event{T}}, statistics::Vector{<:AbstractStatistic},
                  n_actors::Int; maxiter::Int=100, tol::Float64=1e-6) where T

    model = OrdinalBPM(statistics, n_actors)
    n_stats = length(statistics)
    coef = zeros(n_stats)

    # Simple gradient descent (placeholder for full optimization)
    for iter in 1:maxiter
        # Compute gradient numerically
        eps = 1e-5
        grad = zeros(n_stats)

        for i in 1:n_stats
            coef_plus = copy(coef)
            coef_plus[i] += eps
            coef_minus = copy(coef)
            coef_minus[i] -= eps

            # Would compute likelihood here
            grad[i] = 0.0  # Placeholder
        end

        if maximum(abs.(grad)) < tol
            break
        end
    end

    se = fill(NaN, n_stats)
    return OrdinalBPMResult(model, coef, se, NaN, false, length(events))
end

# =============================================================================
# Timing Models
# =============================================================================

"""
    TimingModel

Model for inter-event times in relational event sequences.

# Fields
- `statistics::Vector{AbstractStatistic}`: Statistics affecting hazard
- `baseline::Symbol`: Baseline hazard (:exponential, :weibull, :gompertz)
"""
struct TimingModel
    statistics::Vector{AbstractStatistic}
    baseline::Symbol

    function TimingModel(statistics::Vector{<:AbstractStatistic};
                         baseline::Symbol=:exponential)
        baseline in (:exponential, :weibull, :gompertz) ||
            throw(ArgumentError("baseline must be :exponential, :weibull, or :gompertz"))
        new(statistics, baseline)
    end
end

"""
    TimingModelResult

Results from fitting a timing model.
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
    println(io, "Baseline params: $(result.baseline_params)")
    println(io, "Log-likelihood: $(round(result.loglik, digits=4))")
    println(io, "Converged: $(result.converged)")
end

"""
    hazard_rate(model::TimingModel, coef, baseline_params, t, x) -> Float64

Compute the hazard rate at time t given covariates x.
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
end

"""
    survival_function(model::TimingModel, coef, baseline_params, t, x) -> Float64

Compute survival probability at time t given covariates x.
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
end

"""
    fit_timing(events, statistics, n_actors; baseline=:exponential) -> TimingModelResult

Fit a timing model for inter-event times.
"""
function fit_timing(events::Vector{Event{T}}, statistics::Vector{<:AbstractStatistic},
                    n_actors::Int; baseline::Symbol=:exponential, maxiter::Int=100) where T

    model = TimingModel(statistics; baseline=baseline)
    n_stats = length(statistics)

    # Compute inter-event times
    times = [e.time for e in events]
    sort!(times)
    inter_times = diff(times)

    # Initialize baseline parameters
    coef = zeros(n_stats)
    baseline_params = baseline == :exponential ? [1.0 / mean(inter_times)] : [1.0, 1.0]
    se = fill(NaN, n_stats)

    return TimingModelResult(model, coef, baseline_params, se, NaN, false)
end

# =============================================================================
# Cumulative Network State
# =============================================================================

"""
    CumulativeState{T}

Track cumulative network state with decay for REM statistics.
"""
mutable struct CumulativeState{T}
    n_actors::Int
    adj_matrix::Matrix{Float64}
    outdegree::Vector{Float64}
    indegree::Vector{Float64}
    last_update::T
    decay::Float64

    function CumulativeState{T}(n::Int; halflife::Float64=Inf) where T
        decay = halflife == Inf ? 0.0 : log(2) / halflife
        new{T}(n, zeros(n, n), zeros(n), zeros(n), zero(T), decay)
    end
end

CumulativeState(n::Int; kwargs...) = CumulativeState{Float64}(n; kwargs...)

"""
    update_state!(state::CumulativeState, event::Event)

Update the cumulative state with a new event.
"""
function update_state!(state::CumulativeState{T}, event::Event{T}) where T
    if state.decay > 0 && event.time > state.last_update
        dt = event.time - state.last_update
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

get_outdegree_history(state::CumulativeState, actor::Int) = state.outdegree[actor]
get_indegree_history(state::CumulativeState, actor::Int) = state.indegree[actor]

end # module
