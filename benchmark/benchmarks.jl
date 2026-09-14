#!/usr/bin/env julia
# benchmark/benchmarks.jl — BenchmarkTools suite for Relevent.jl's hot loops.
#
# Covers the streaming decayed accumulators that replaced the per-evaluation
# event-history rescans: a full-risk-set sweep of the decay-weighted
# statistics (the inner loop of both likelihoods — each event is absorbed
# once, reads are decayed lazily) and the ordinal `fit_obpm` estimator built
# on top of them.
#
# Defines the standard `SUITE::BenchmarkGroup`. Run standalone with
#     julia --project=benchmark benchmark/benchmarks.jl
# which tunes + runs the suite and prints one tab-separated `BENCHJL` line
# per benchmark (consumed by the site repo's tools/run_benchmarks.jl).

using BenchmarkTools
using Random
using Relevent
using REM: Event, compute

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

const N_ACTORS = 20
const N_EVENTS = 400

function make_events(rng::AbstractRNG, n_actors::Int, n_events::Int)
    events = Event{Float64}[]
    t = 0.0
    for _ in 1:n_events
        t += -log(rand(rng))
        s = rand(rng, 1:n_actors)
        r = rand(rng, 1:(n_actors - 1))
        r >= s && (r += 1)
        push!(events, Event(s, r, t))
    end
    return events
end

const EVENTS = make_events(Random.Xoshiro(20260712), N_ACTORS, N_EVENTS)

# Decay-weighted statistics backed by the streaming accumulators, plus the
# p-shift (pure last-event lookup) that shares the inner loop with them.
make_stats() = [LocalInertia(5.0), Momentum(1.0), SendingCapacity(1.0),
                ReceivingCapacity(1.0), PShift(:AB_BA)]

"""
Full-risk-set statistics sweep: replay the event stream, evaluating every
statistic for every candidate dyad before absorbing each event — exactly the
work pattern of the likelihood evaluations. Fresh statistics (empty
accumulator caches) are built per call so repeated evaluations are
identical.
"""
function statistics_sweep(events, n_actors)
    stats = make_stats()
    history = InteractionHistory()
    s = 0.0
    for event in events
        t = event.time
        for stat in stats
            for i in 1:n_actors, j in 1:n_actors
                i == j && continue
                s += compute(stat, history, i, j, t)
            end
        end
        update_history!(history, event)
    end
    return s
end

# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

const SUITE = BenchmarkGroup()

let g = addgroup!(SUITE, "statistics")
    g["fullriskset_sweep_$(N_EVENTS)ev"] =
        @benchmarkable statistics_sweep($EVENTS, $N_ACTORS)
end

let g = addgroup!(SUITE, "estimation")
    g["fit_obpm_$(N_EVENTS)ev"] =
        @benchmarkable fit_obpm($EVENTS, stats, $N_ACTORS) setup =
            (stats = make_stats())

    # The risk-set cache policies (issue Relevent#2). `:all` materializes an
    # n(n−1) × p design matrix per event — O(E · n² · p), 906 MB at
    # (n, E, p) = (100, 2000, 6) — while `:chunked`/`:none` recompute them into a
    # bounded cache on each pass. The fits are bit-identical; only cost moves.
    #
    # READ THE `memory` COLUMN WITH CARE. BenchmarkTools reports TOTAL bytes
    # allocated, not the peak LIVE footprint — and the trade here runs the other
    # way on that measure: the streamed policies allocate MORE in total (they
    # replay the interaction history on every pass, and that garbage is counted)
    # while holding far less live. The quantity the issue is about is the live
    # one, and it is not in this table: it is `k * Relevent._design_bytes(plan)`
    # for the `(mode, k)` that `Relevent._resolve_cache` returns — exact, and
    # computable without allocating any of it. What this table measures honestly
    # is the TIME each policy costs.
    for cache in (:all, :chunked, :none)
        g["fit_obpm_$(N_EVENTS)ev_cache_$(cache)"] =
            @benchmarkable fit_obpm($EVENTS, stats, $N_ACTORS; cache=$cache,
                                    chunk=32) setup = (stats = make_stats())
    end
end

# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

function print_benchjl(results::BenchmarkGroup)
    for (path, trial) in BenchmarkTools.leaves(results)
        est = median(trial)
        println("BENCHJL\t", join(path, "/"), "\t",
                BenchmarkTools.time(est), "\t",
                BenchmarkTools.allocs(est), "\t",
                BenchmarkTools.memory(est))
    end
end

function main()
    tune!(SUITE)
    results = run(SUITE; verbose=false, seconds=1)
    print_benchjl(results)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
