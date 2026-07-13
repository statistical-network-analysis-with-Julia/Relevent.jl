using Relevent
using REM
import Networks
using Distributions
using LinearAlgebra
using Random
using Statistics
using Test

function history_fixture()
    h = InteractionHistory{Float64}()
    update_history!(h, Event(1, 2, 1.0))
    update_history!(h, Event(1, 2, 2.0))
    update_history!(h, Event(2, 1, 3.0))
    update_history!(h, Event(1, 3, 4.0))
    return h
end

@testset "Relevent.jl" begin
    @testset "InteractionHistory" begin
        h = history_fixture()
        @test get_interaction_count(h, 1, 2) == 2
        @test get_interaction_count(h, 3, 1) == 0
        @test get_last_interaction(h, 1, 2) == 2.0
        @test get_last_interaction(h, 3, 1) === nothing
        @test length(h.events) == 4
    end

    @testset "History-based statistics (hand values)" begin
        h = history_fixture()
        t = 5.0
        d = log(2) / 10.0  # halflife 10

        # PriorInteraction outgoing on (1,2): events at 1.0 and 2.0
        expected_out = exp(-d * 4.0) + exp(-d * 3.0)
        @test compute(PriorInteraction(10.0), h, 1, 2, t) ≈ expected_out
        # incoming adds the (2,1) event at 3.0
        expected_in = exp(-d * 2.0)
        @test compute(PriorInteraction(10.0; direction=:both), h, 1, 2, t) ≈
              expected_out + expected_in

        # SendingCapacity of 1: all three outgoing events (1.0, 2.0, 4.0)
        # (the old implementation dropped most of these terms)
        expected_sc = exp(-d * 4.0) + exp(-d * 3.0) + exp(-d * 1.0)
        @test compute(SendingCapacity(10.0), h, 1, 99, t) ≈ expected_sc

        # ReceivingCapacity of 2: one incoming event at min(1,2)... events
        # into 2: (1,2)@1.0 and (1,2)@2.0
        @test compute(ReceivingCapacity(10.0), h, 99, 2, t) ≈
              exp(-d * 4.0) + exp(-d * 3.0)

        # LocalInertia uses the most recent dyad event only
        @test compute(LocalInertia(10.0), h, 1, 2, t) ≈ exp(-d * 3.0)
        @test compute(LocalInertia(10.0), h, 3, 1, t) == 0.0

        # Momentum normalization divides by the sender's own event count
        m_raw = compute(Momentum(10.0), h, 1, 99, t)
        m_norm = compute(Momentum(10.0; normalize=true), h, 1, 99, t)
        @test m_raw ≈ expected_sc
        @test m_norm ≈ expected_sc / 3

        @test_throws ArgumentError PriorInteraction(10.0; direction=:bogus)
        @test_throws ArgumentError PriorInteraction(-1.0)
    end

    @testset "REM.EventNetworkState interface and fit_rem integration" begin
        events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0),
                  Event(2, 3, 4.0), Event(3, 1, 5.0), Event(1, 3, 6.0)]
        seq = EventSequence(events)

        # 4-arg compute agrees with the history-based 5-arg compute
        state = REM.EventNetworkState(seq)
        h = InteractionHistory{Float64}()
        for e in events[1:4]
            REM.update!(state, e)
            update_history!(h, e)
        end
        state.current_time = 5.0

        for stat in (PriorInteraction(8.0; direction=:both), SendingCapacity(8.0),
                     ReceivingCapacity(8.0), LocalInertia(8.0),
                     Momentum(8.0; normalize=true))
            @test compute(stat, state, 1, 2) ≈ compute(stat, h, 1, 2, 5.0) atol = 1e-12
        end

        # The README workflow: Relevent statistics inside REM.fit_rem
        # (this used to throw "compute() not implemented")
        result = fit_rem(seq, [Repetition(), LocalInertia(5.0)];
                         n_controls=10, seed=1)
        @test result isa REM.REMResult
        @test length(coef(result)) == 2
        @test all(isfinite, coef(result))
    end

    @testset "Streaming accumulators match a direct-scan reference" begin
        # The decay-weighted statistics are computed from lazily decayed
        # accumulators (each event absorbed once); on a random stream they
        # must agree with the eager reference that rescans the full
        # history on every evaluation (the old implementation).
        function ref_scan(events, t_now, d; keep)
            return sum(exp(-d * (t_now - e.time)) for e in events if keep(e);
                       init=0.0)
        end

        rng = Random.Xoshiro(42)
        n = 6
        halflife = 9.0
        d = log(2) / halflife

        events = Event{Float64}[]
        t = 0.0
        while length(events) < 250
            s, r = rand(rng, 1:n), rand(rng, 1:n)
            s == r && continue
            t += 2 * rand(rng)
            push!(events, Event(s, r, t))
        end

        h = InteractionHistory{Float64}()
        seq = EventSequence(events)
        state = REM.EventNetworkState(seq)

        for (m, ev) in enumerate(events)
            t_now = ev.time
            state.current_time = t_now
            # Evaluate on the strictly-pre-event history, several dyads
            for (s, r) in ((1, 2), (2, 1), (ev.sender, ev.receiver), (5, 3))
                s == r && continue
                past = events[1:(m - 1)]
                out = ref_scan(past, t_now, d; keep=e -> e.sender == s && e.receiver == r)
                inc = ref_scan(past, t_now, d; keep=e -> e.sender == r && e.receiver == s)
                snd = ref_scan(past, t_now, d; keep=e -> e.sender == s)
                rcv = ref_scan(past, t_now, d; keep=e -> e.receiver == r)
                n_snd = count(e -> e.sender == s, past)

                for (src, args) in ((h, (s, r, t_now)), (state, (s, r)))
                    @test compute(PriorInteraction(halflife), src, args...) ≈
                          out atol = 1e-10
                    @test compute(PriorInteraction(halflife; direction=:incoming),
                                  src, args...) ≈ inc atol = 1e-10
                    @test compute(PriorInteraction(halflife; direction=:both),
                                  src, args...) ≈ out + inc atol = 1e-10
                    @test compute(SendingCapacity(halflife), src, args...) ≈
                          snd atol = 1e-10
                    @test compute(ReceivingCapacity(halflife), src, args...) ≈
                          rcv atol = 1e-10
                    @test compute(Momentum(halflife), src, args...) ≈
                          snd atol = 1e-10
                    @test compute(Momentum(halflife; normalize=true), src, args...) ≈
                          (n_snd > 0 ? snd / n_snd : 0.0) atol = 1e-10
                end
            end

            update_history!(h, ev)
            REM.update!(state, ev)
        end

        # Resetting the underlying state invalidates the cached accumulators
        REM.reset!(state)
        state.current_time = 1.0
        @test compute(SendingCapacity(halflife), state, 1, 2) == 0.0
        REM.update!(state, Event(1, 2, 1.0))
        state.current_time = 1.0
        @test compute(SendingCapacity(halflife), state, 1, 3) ≈ 1.0
    end

    @testset "Participation shifts (Gibson 2003)" begin
        @test length(pshift_types()) == 13
        @test Relevent.name(PShift(:AB_BA)) == "PSAB-BA"
        @test PShift("PSAB-XY").shift == :AB_XY
        @test PShift("AB-B0").shift == :AB_B0
        @test_throws ArgumentError PShift(:AB_ZZ)

        # Hand-computed indicators. After the dyadic event 1→2
        # (A = 1, B = 2), each candidate realizes exactly one shift
        # (or none: repeating 1→2 is not a participation shift).
        h = InteractionHistory{Float64}()
        update_history!(h, Event(1, 2, 1.0))

        expected_dyadic = Dict(
            (2, 1) => :AB_BA,   # turn receiving: B answers A
            (2, 3) => :AB_BY,   # turn receiving: B addresses someone new
            (2, 0) => :AB_B0,   # turn receiving: B addresses the group
            (1, 3) => :AB_AY,   # turn continuing: A addresses someone new
            (1, 0) => :AB_A0,   # turn continuing: A addresses the group
            (3, 1) => :AB_XA,   # turn usurping: outsider addresses A
            (3, 2) => :AB_XB,   # turn usurping: outsider addresses B
            (3, 4) => :AB_XY,   # turn usurping: outsider addresses outsider
            (3, 0) => :AB_X0,   # turn usurping: outsider addresses the group
            (1, 2) => nothing,  # repetition of A→B: no shift
        )
        for ((i, j), shift) in expected_dyadic, ps in pshift_types()
            @test compute(PShift(ps), h, i, j, 2.0) ==
                  (ps === shift ? 1.0 : 0.0)
        end

        # After the group-directed event 1→0 (A = 1, null receiver),
        # only the A0-* shifts can fire
        h0 = InteractionHistory{Float64}()
        update_history!(h0, Event(1, 0, 1.0))

        expected_group = Dict(
            (2, 0) => :A0_X0,   # turn claiming: outsider addresses the group
            (2, 1) => :A0_XA,   # turn claiming: outsider answers A
            (2, 3) => :A0_XY,   # turn claiming: outsider addresses outsider
            (1, 3) => :A0_AY,   # turn continuing: A addresses someone
        )
        for ((i, j), shift) in expected_group, ps in pshift_types()
            @test compute(PShift(ps), h0, i, j, 2.0) ==
                  (ps === shift ? 1.0 : 0.0)
        end

        # No previous event → all shifts are 0
        h_empty = InteractionHistory{Float64}()
        @test all(compute(PShift(ps), h_empty, 1, 2, 1.0) == 0.0
                  for ps in pshift_types())

        # The REM.EventNetworkState interface agrees with the history one
        seq = EventSequence([Event(1, 2, 1.0)])
        state = REM.EventNetworkState(seq)
        REM.update!(state, seq[1])
        state.current_time = 2.0
        for ps in pshift_types(), (i, j) in keys(expected_dyadic)
            @test compute(PShift(ps), state, i, j) ==
                  compute(PShift(ps), h, i, j, 2.0)
        end
        @test compute(PShift(:AB_BA), REM.EventNetworkState{Float64}(), 1, 2) == 0.0

        # P-shifts fit inside both estimators
        rng = Random.Xoshiro(11)
        events = Event{Float64}[]
        t = 0.0
        prev = (1, 2)
        for m in 1:80
            t += rand(rng)
            # strongly turn-receiving stream: mostly answer the last event
            if rand(rng) < 0.7
                s, r = prev[2], prev[1]
            else
                s, r = rand(rng, 1:5), rand(rng, 1:5)
                s == r && continue
            end
            push!(events, Event(s, r, t))
            prev = (s, r)
        end
        result = fit_obpm(events, [PShift(:AB_BA)], 5)
        @test result.converged
        @test result.coefficients[1] > 0   # answering is over-represented

        seq2 = EventSequence(events)
        rem_fit = REM.fit_rem(seq2, [Repetition(), PShift(:AB_BA)];
                              n_controls=10, seed=3)
        @test all(isfinite, coef(rem_fit))
    end

    @testset "Covariate effects (CovSnd/CovRec/CovInt)" begin
        x = [1.5, -0.5, 2.0]
        h = InteractionHistory{Float64}()
        state = REM.EventNetworkState{Float64}()

        @test Relevent.name(CovSnd(x)) == "CovSnd"
        @test Relevent.name(CovRec(x; name="CovRec.age")) == "CovRec.age"

        # Hand values on both interfaces: CovSnd = x_i, CovRec = x_j,
        # CovInt = x_i + x_j (relevent: effect on outgoing AND incoming)
        @test compute(CovSnd(x), h, 1, 2, 0.0) == 1.5
        @test compute(CovSnd(x), state, 1, 2) == 1.5
        @test compute(CovRec(x), h, 1, 2, 0.0) == -0.5
        @test compute(CovRec(x), state, 1, 2) == -0.5
        @test compute(CovInt(x), h, 3, 2, 0.0) == 1.5
        @test compute(CovInt(x), state, 3, 2) == 1.5

        @test_throws ArgumentError compute(CovSnd(x), state, 4, 2)
        @test_throws ArgumentError compute(CovRec(x), h, 1, 4, 0.0)

        # Sender effect is recovered with the right sign: actor 3 sends at
        # a much higher rate
        rng = Random.Xoshiro(5)
        z = [0.0, 0.0, 2.0, 0.0]
        events = Event{Float64}[]
        for m in 1:120
            t = Float64(m)
            s = rand(rng) < 0.75 ? 3 : rand(rng, [1, 2, 4])
            r = rand(rng, setdiff(1:4, s))
            push!(events, Event(s, r, t))
        end
        result = fit_obpm(events, [CovSnd(z)], 4)
        @test result.converged
        @test result.coefficients[1] > 0
    end

    @testset "rank_events" begin
        events = [Event(1, 2, 3.0), Event(2, 1, 1.0), Event(1, 3, 2.0)]
        @test rank_events(events) == [3, 1, 2]
    end

    @testset "fit_obpm recovers a known coefficient" begin
        # Simulate ordinal events from the multinomial model with a
        # LocalInertia effect, then fit by full-risk-set ML
        rng = Random.Xoshiro(2026)
        n = 6
        θ_true = 1.2
        stat = LocalInertia(5.0)
        dyads = [(s, r) for s in 1:n for r in 1:n if s != r]
        h = InteractionHistory{Float64}()
        events = Event{Float64}[]

        for m in 1:400
            t = Float64(m)
            η = [θ_true * compute(stat, h, s, r, t) for (s, r) in dyads]
            w = exp.(η .- maximum(η))
            w ./= sum(w)
            u = rand(rng)
            acc = 0.0
            pick = length(dyads)
            for (k, p) in enumerate(w)
                acc += p
                if u <= acc
                    pick = k
                    break
                end
            end
            ev = Event(dyads[pick][1], dyads[pick][2], t)
            push!(events, ev)
            update_history!(h, ev)
        end

        result = fit_obpm(events, [stat], n)
        @test result.converged
        @test isfinite(result.loglik)
        @test result.loglik < 0
        @test result.std_errors[1] > 0
        @test result.coefficients[1] ≈ θ_true atol = 0.25

        # StatsAPI accessors (same generics as REM/StatsBase)
        @test coef(result) == result.coefficients
        @test stderror(result) == result.std_errors

        # Null model sanity: a constant-hazard fit has loglik
        # M · log(1/n_dyads) when the statistic carries no signal... use
        # zero-information data: single event → coefficient ≈ 0 pull
        @test fit_obpm(events[1:1], [stat], n).loglik ≈ log(1 / 30) atol = 1e-6
    end

    @testset "fit_timing recovers rate and coefficient" begin
        # Simulate a proportional-hazards event stream: waiting times
        # exponential with rate λ₀·Σexp(θ'x), dyad picked ∝ exp(θ'x)
        rng = Random.Xoshiro(7)
        n = 5
        λ0_true = 0.4
        θ_true = 0.9
        stat = LocalInertia(6.0)
        dyads = [(s, r) for s in 1:n for r in 1:n if s != r]
        h = InteractionHistory{Float64}()
        events = Event{Float64}[]
        t = 0.0

        for m in 1:500
            x = [compute(stat, h, s, r, t) for (s, r) in dyads]
            w = exp.(θ_true .* x)
            total_rate = λ0_true * sum(w)
            t += rand(rng, Exponential(1 / total_rate))
            # Statistics decay between events; recompute at the event time
            x = [compute(stat, h, s, r, t) for (s, r) in dyads]
            w = exp.(θ_true .* x)
            probs = w ./ sum(w)
            u = rand(rng)
            acc = 0.0
            pick = length(dyads)
            for (k, p) in enumerate(probs)
                acc += p
                if u <= acc
                    pick = k
                    break
                end
            end
            ev = Event(dyads[pick][1], dyads[pick][2], t)
            push!(events, ev)
            update_history!(h, ev)
        end

        result = fit_timing(events, [stat], n)
        @test result.converged
        @test isfinite(result.loglik)
        @test result.baseline_params[1] ≈ λ0_true rtol = 0.2
        @test result.coefficients[1] ≈ θ_true atol = 0.3

        # StatsAPI accessors (same generics as REM/StatsBase)
        @test coef(result) == result.coefficients
        @test stderror(result) == result.std_errors

        # Non-exponential baselines are not silently mis-fit
        @test_throws ErrorException fit_timing(events, [stat], n;
                                               baseline=:weibull)

        # t0 (observation onset): shifting the whole timeline by c and
        # setting t0 = c leaves every waiting time — hence the fit —
        # unchanged (statistics depend on time differences only)
        c = 25.0
        shifted = [Event(e.sender, e.receiver, e.time + c) for e in events]
        shifted_fit = fit_timing(shifted, [stat], n; t0=c)
        @test shifted_fit.coefficients ≈ result.coefficients atol = 1e-8
        @test shifted_fit.baseline_params[1] ≈ result.baseline_params[1] atol = 1e-8
        @test shifted_fit.loglik ≈ result.loglik atol = 1e-6

        # Without t0 the first interval is overstated by c, biasing λ₀ down
        default_fit = fit_timing(shifted, [stat], n)
        @test default_fit.baseline_params[1] < shifted_fit.baseline_params[1]

        # The onset must precede the first event
        @test_throws ArgumentError fit_timing(shifted, [stat], n;
                                              t0=shifted[1].time + 1.0)
    end

    @testset "Golden: relevent::rem.dyad (ordinal + timing)" begin
        # Frozen output of R relevent::rem.dyad, with provenance and a stated,
        # justified tolerance (see test/fixtures/relevent_rem_dyad.toml and the
        # generating script test/fixtures/r/relevent_rem_dyad.R).
        #
        # Both likelihoods are EXACT — no Monte Carlo — so this is a real
        # equality check, not a "close enough" one: 1e-6 is the reference
        # optimizer's termination slack, nothing more.
        g = Networks.load_golden(joinpath(@__DIR__, "fixtures",
                                          "relevent_rem_dyad.toml"))
        report(key, actual) = begin
            ok = Networks.check_golden(g, key, actual)
            ok || println(stderr, Networks.golden_report(g, key, actual))
            ok
        end

        n = Int(g.values["n_actors"])
        times = Float64.(g.values["input_time"])
        senders = Int.(g.values["input_sender"])
        receivers = Int.(g.values["input_receiver"])
        z = Float64.(g.values["input_covariate"])
        t_end = Float64(g.values["t_end"])
        events = [Event(senders[i], receivers[i], times[i]) for i in eachindex(times)]

        # In R's coefficient order (CovSnd, CovRec, then the p-shifts)
        stats = AbstractStatistic[CovSnd(z), CovRec(z),
                                  PShift(:AB_BA), PShift(:AB_BY),
                                  PShift(:AB_XB), PShift(:AB_AY)]
        @test g.values["ordinal_names"] ==
              ["CovSnd.1", "CovRec.1", "PSAB-BA", "PSAB-BY", "PSAB-XB", "PSAB-AY"]

        # --- ordinal likelihood -------------------------------------------
        ord = fit_obpm(events, stats, n; tol=1e-12)
        @test ord.converged
        @test report("ordinal_coefficients", ord.coefficients)
        @test report("ordinal_std_errors", ord.std_errors)
        @test report("ordinal_loglik", ord.loglik)

        # --- temporal likelihood ------------------------------------------
        # rem.dyad has no intercept, so R gets a constant CovSnd column whose
        # coefficient IS log(λ₀); and its last edgelist row is the end of the
        # observation window, so the likelihood carries a right-censored final
        # interval. `t_end` is what reproduces that.
        tim = fit_timing(events, stats, n; t_end=t_end, tol=1e-12)
        @test tim.converged
        @test report("timing_log_baseline", log(tim.baseline_params[1]))
        @test report("timing_coefficients", tim.coefficients)
        @test report("timing_std_errors", tim.std_errors)
        @test report("timing_loglik", tim.loglik)

        # ...and the gap that `t_end` closes, pinned so it cannot come back
        # silently: drop the eventless tail and the fitted baseline rate is
        # biased UPWARD (the same events in a shorter window), here by ~0.047
        # in log λ₀ — 350x the tolerance above.
        no_end = fit_timing(events, stats, n; tol=1e-12)
        gap = log(no_end.baseline_params[1]) - Float64(g.values["timing_log_baseline"])
        @test gap > 0.04
        @test !Networks.check_golden(g, "timing_log_baseline",
                                     log(no_end.baseline_params[1]))

        # --- and the SAME fixture under every cache policy (Relevent#2) -----
        # The point of the cache modes is that they change memory and time and
        # NOTHING else. Held to the golden numbers, not merely to each other.
        for cache in (:all, :chunked, :none)
            o = fit_obpm(events, stats, n; tol=1e-12, cache=cache, chunk=8)
            t = fit_timing(events, stats, n; t_end=t_end, tol=1e-12, cache=cache,
                           chunk=8)
            @test report("ordinal_coefficients", o.coefficients)
            @test report("timing_coefficients", t.coefficients)
            # bit-identical, not merely within the golden tolerance
            @test o.coefficients == ord.coefficients
            @test o.std_errors == ord.std_errors
            @test o.loglik === ord.loglik
            @test t.coefficients == tim.coefficients
            @test t.std_errors == tim.std_errors
            @test t.baseline_params == tim.baseline_params
            @test t.loglik === tim.loglik
        end
    end

    # ------------------------------------------------------------------
    # Risk-set cache policies (issue Relevent#2)
    #
    # `_risk_set_stats` materialized an n(n−1) × p design matrix for EVERY
    # event: O(E · n² · p) doubles, which put a ceiling on exact full-risk-set
    # estimation well below the point where the arithmetic gets hard. `cache=`
    # bounds it. The claim these tests police is that it bounds ONLY that: the
    # intervals are visited in the same order and the statistics read off the
    # same histories, so every accumulation is summed identically and the fits
    # are bit-identical — `==`, not `≈`.
    # ------------------------------------------------------------------
    @testset "Risk-set cache policies (:all / :chunked / :none)" begin
        rng = MersenneTwister(31337)
        n = 9
        events = Event{Float64}[]
        t = 0.0
        s, r = 1, 2
        for k in 1:80
            t += 0.05 + rand(rng)
            u = rand(rng)
            if k > 1 && u < 0.3
                s, r = r, s                      # reciprocity, so PShift bites
            elseif k > 1 && u >= 0.5
                s = rand(rng, 1:n)
                r = rand(rng, filter(!=(s), 1:n))
            end
            push!(events, Event(s, r, t))
        end
        stats = AbstractStatistic[PriorInteraction(4.0), PShift(:AB_BA),
                                  SendingCapacity(3.0)]

        ref_o = fit_obpm(events, stats, n)
        ref_t = fit_timing(events, stats, n; t_end=t + 5.0)
        @test ref_o.converged && ref_t.converged

        for cache in (:auto, :all, :chunked, :none), chunk in (nothing, 1, 3, 17, 500)
            cache === :chunked || chunk === nothing || continue   # chunk only bites there
            o = fit_obpm(events, stats, n; cache=cache, chunk=chunk)
            @test o.coefficients == ref_o.coefficients
            @test o.std_errors == ref_o.std_errors
            @test o.loglik === ref_o.loglik

            tm = fit_timing(events, stats, n; t_end=t + 5.0, cache=cache, chunk=chunk)
            @test tm.coefficients == ref_t.coefficients
            @test tm.baseline_params == ref_t.baseline_params
            @test tm.std_errors == ref_t.std_errors
            @test tm.loglik === ref_t.loglik
        end

        # ...including under a tie correction, where `:all` SHARES one design
        # matrix across a frozen block and the streamed policies recompute it.
        # Same history, same read time, so the same numbers.
        tied = copy(events)
        push!(tied, Event(4, 5, tied[10].time))     # a tie, distinct dyads
        ref_b = fit_obpm(tied, stats, n; ties=:breslow)
        ref_e = fit_obpm(tied, stats, n; ties=:efron)
        for cache in (:all, :chunked, :none)
            b = fit_obpm(tied, stats, n; ties=:breslow, cache=cache, chunk=2)
            e = fit_obpm(tied, stats, n; ties=:efron, cache=cache, chunk=2)
            @test b.coefficients == ref_b.coefficients
            @test e.coefficients == ref_e.coefficients   # the Efron weights too
            @test e.loglik === ref_e.loglik
        end
        @test ref_b.coefficients != ref_e.coefficients   # the corrections differ

        # The memory the policy actually buys, without allocating any of it.
        plan = Relevent._risk_set_plan(events, Tuple(stats), n)
        one_matrix = Relevent._design_bytes(plan)
        @test one_matrix == n * (n - 1) * length(stats) * sizeof(Float64)
        @test Relevent._resolve_cache(plan; cache=:all) == (:all, plan.n_int)
        @test Relevent._resolve_cache(plan; cache=:none) == (:none, 1)
        @test Relevent._resolve_cache(plan; cache=:chunked, chunk=5) == (:chunked, 5)
        # a chunk covering everything IS :all — same memory, no recomputation
        @test Relevent._resolve_cache(plan; cache=:chunked,
                                      chunk=plan.n_int) == (:all, plan.n_int)
        # :auto caches everything under budget and chunks above it
        @test Relevent._resolve_cache(plan; cache=:auto)[1] === :all
        @test Relevent._resolve_cache(plan; cache=:auto,
                                      cache_bytes=4 * one_matrix) == (:chunked, 4)

        @test_throws ArgumentError fit_obpm(events, stats, n; cache=:some)
        @test_throws ArgumentError fit_obpm(events, stats, n; cache=:chunked, chunk=0)
    end

    # ------------------------------------------------------------------
    # Allocation regression on the derivative loop (review finding 15)
    #
    # `fit_timing`'s per-interval `Xm' * (w .* Xm)` allocated a full weighted
    # copy of the risk-set design matrix AND a fresh p×p outer product on every
    # event of every Newton evaluation. Both likelihoods now run on preallocated
    # workspaces: what a derivative evaluation allocates must not scale with the
    # number of events or the size of the risk set.
    # ------------------------------------------------------------------
    @testset "Derivative evaluations allocate O(p²), not O(E · n² · p)" begin
        stats = AbstractStatistic[PriorInteraction(4.0), PShift(:AB_BA)]
        p = length(stats)

        # The REAL closures the fitters hand to `_newton` — not a copy of the
        # loop, which would only test the test.
        function derivative_allocs(n, E; cache=:all)
            rng = MersenneTwister(4)
            events = Event{Float64}[]
            t = 0.0
            for _ in 1:E
                t += 0.1 + rand(rng)
                s = rand(rng, 1:n)
                r = rand(rng, filter(!=(s), 1:n))
                push!(events, Event(s, r, t))
            end
            # the ordinal likelihood has no censored tail; the timing one does
            plan_o = Relevent._risk_set_plan(events, Tuple(stats), n)
            plan_t = Relevent._risk_set_plan(events, Tuple(stats), n; t_end=t + 1.0)
            dO = Relevent._obpm_derivatives(Relevent._risk_sets(plan_o; cache=cache))
            dT = Relevent._timing_derivatives(Relevent._risk_sets(plan_t; cache=cache))
            θ = fill(0.05, p)
            β = fill(0.05, p + 1)
            dO(θ); dT(β)                   # warm up: @allocated on a first call
            return (@allocated dO(θ)),     #          would measure compilation
                   (@allocated dT(β))
        end

        # 30 dyads / 21 intervals, then 182 dyads / 201 intervals: 10x the events
        # and 6x the risk set, i.e. 60x the work. The workspaces are reused, so
        # the allocations do not move — only the p-sized gradient and p×p Hessian
        # returned to the optimizer are allocated per evaluation.
        o_small, t_small = derivative_allocs(6, 20)
        o_big, t_big = derivative_allocs(14, 200)

        @test o_small <= 512      # 2 + 2x2 doubles, plus two array headers
        @test t_small <= 512
        @test o_big <= 512
        @test t_big <= 512
        @test o_big <= o_small + 64
        @test t_big <= t_small + 64

        # ...and the streamed policies recompute the design matrices into their
        # bounded buffers rather than allocating one per interval. They are NOT
        # allocation-free — replaying the interaction history allocates, O(E) —
        # but what they allocate is the history, not the risk sets: an order of
        # magnitude below the design-matrix cache they exist to avoid.
        o_none, t_none = derivative_allocs(14, 200; cache=:none)
        plan = Relevent._risk_set_plan(Event{Float64}[Event(1, 2, float(k))
                                                      for k in 1:200],
                                       Tuple(stats), 14)
        cached = Relevent._design_bytes(plan) * plan.n_int   # what :all holds
        @test o_none < cached ÷ 10
        @test t_none < cached ÷ 10
    end

    @testset "fit_relevent / rem_dyad standardized entry point" begin
        rng = Random.Xoshiro(13)
        n = 5
        events = Event{Float64}[]
        t = 0.0
        for m in 1:60
            t += rand(rng)
            s = rand(rng, 1:n)
            r = rand(rng, setdiff(1:n, s))
            push!(events, Event(s, r, t))
        end
        stats = [PShift(:AB_BA), LocalInertia(5.0)]

        # ordinal=true (the default, as in relevent::rem.dyad) -> fit_obpm
        r_ord = fit_relevent(events, stats, n)
        @test r_ord isa OrdinalBPMResult
        ref_ord = fit_obpm(events, stats, n)
        @test r_ord.coefficients == ref_ord.coefficients
        @test r_ord.loglik == ref_ord.loglik

        # ordinal=false -> the interval (timing) likelihood, kwargs forwarded
        r_tim = fit_relevent(events, stats, n; ordinal=false, t0=0.0)
        @test r_tim isa TimingModelResult
        ref_tim = fit_timing(events, stats, n; t0=0.0)
        @test r_tim.coefficients == ref_tim.coefficients
        @test r_tim.baseline_params == ref_tim.baseline_params

        # rem_dyad is the R-faithful alias
        r_alias = rem_dyad(events, stats, n)
        @test r_alias isa OrdinalBPMResult
        @test r_alias.coefficients == r_ord.coefficients

        # show() renders the shared ecosystem coefficient table
        # (Networks.jl print_coeftable: z / Pr(>|z|) columns + signif codes)
        for (res, heading) in ((r_ord, "Ordinal Butts-Park Model Results"),
                               (r_tim, "Timing Model Results"))
            out = sprint(show, res)
            @test occursin(heading, out)
            @test occursin("PSAB-BA", out)
            @test occursin("Pr(>|z|)", out)
            @test occursin("Signif. codes", out)
        end
    end

    @testset "Hazard/survival consistency" begin
        stat = LocalInertia(5.0)
        x = [0.7]
        coefs = [0.5]

        for (baseline, params) in ((:exponential, [0.3]),
                                   (:weibull, [2.0, 1.5]),
                                   (:gompertz, [0.2, 0.1]))
            model = TimingModel([stat]; baseline=baseline)
            # S(t) = exp(-∫₀ᵗ h(u)du): check numerically
            t_max = 2.0
            grid = range(1e-6, t_max, length=20001)
            H = sum(hazard_rate(model, coefs, params, u, x) for u in grid) *
                (t_max / length(grid))
            S = survival_function(model, coefs, params, t_max, x)
            @test S ≈ exp(-H) rtol = 1e-2
        end

        @test_throws ArgumentError TimingModel([stat]; baseline=:bogus)
    end

    @testset "CumulativeState" begin
        state = CumulativeState(3; halflife=1.0)
        update_state!(state, Event(1, 2, 0.0))
        update_state!(state, Event(1, 3, 1.0))  # one halflife later

        # First event decayed to 1/2, plus the new event
        @test get_outdegree_history(state, 1) ≈ 0.5 + 1.0
        @test get_indegree_history(state, 2) ≈ 0.5
        @test state.adj_matrix[1, 3] ≈ 1.0

        @test_throws ArgumentError update_state!(state, Event(1, 9, 2.0))
    end

    @testset "Result metadata protocol" begin
        # Relevent's distinguishing property against REM.fit_rem: both models
        # enumerate the FULL risk set, so their objectives ARE the exact
        # likelihoods. The protocol has to say so — and, on THIS data, has to say
        # that nothing else is in the way either: the event times are strictly
        # ordered, so no tie policy could bite and there is no caveat to carry.
        events = [Event(1, 2, 1.0), Event(2, 3, 2.0), Event(3, 1, 3.0),
                  Event(2, 1, 4.0), Event(1, 3, 5.0), Event(3, 2, 6.0)]
        n = 4

        ord = fit_obpm(events, [PriorInteraction(2.0)], n)
        md = Networks.fit_metadata(ord)
        @test md.estimand == :relational_event
        @test md.objective == :likelihood
        @test md.is_exact                      # full risk set, no sampling
        @test md.se_method == :hessian
        @test md.missing_method == :none
        @test md.tie_method == :none           # the data has no tied timestamps
        @test isempty(md.approximations)

        tim = fit_timing(events, [PriorInteraction(2.0)], n)
        mdt = Networks.fit_metadata(tim)
        @test mdt.estimand == :relational_event_timing
        @test mdt.objective == :likelihood
        @test mdt.is_exact                     # exact exponential-baseline likelihood
        @test mdt.se_method == :hessian
        @test mdt.tie_method == :none
        @test isempty(mdt.approximations)

        # Contrast with REM's default case-control fit of the same events: the
        # SAME family of objective, but a SAMPLED risk set, so not exact.
        seq = EventSequence(events; actors=1:n)
        rem_fit = REM.fit_rem(seq, [REM.Repetition()]; n_controls=2, seed=3)
        @test Networks.objective(rem_fit) == :partial_likelihood
        @test !Networks.is_exact(rem_fit)
        @test Networks.is_exact(ord)
    end

    @testset "Tied event times: the two likelihoods take different policies" begin
        # Issue Relevent#1 / review finding 12. ONE vocabulary
        # (`Networks.TIE_POLICIES`), ONE keyword (`ties=`), but the two
        # likelihoods here claim DIFFERENT things and therefore accept different
        # subsets of it — and refuse the rest loudly rather than no-op:
        #
        #   fit_obpm   likelihood over the ORDER: :error :ordered :breslow :efron
        #   fit_timing exact-TIME likelihood:     :error :ordered :batch
        n = 4
        # PShift(:AB_BA) is the sharp statistic here: under an arbitrary ordering,
        # the 2→1 event tied with 1→2 "responds" to it instantly — the p-shift
        # fires on information the tie does not contain.
        stats = [PShift(:AB_BA), PriorInteraction(4.0)]
        tied = [Event(1, 2, 1.0), Event(2, 1, 1.0),      # tie at t = 1
                Event(1, 3, 2.0),
                Event(3, 1, 3.0), Event(1, 3, 3.0),      # tie at t = 3
                Event(2, 3, 4.0)]
        untied = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0),
                  Event(3, 1, 4.0), Event(1, 3, 5.0), Event(2, 3, 6.0)]

        # --- the default REFUSES, in BOTH likelihoods, and names the tie ------
        e_o = try; fit_obpm(tied, stats, n); catch e; e; end
        e_t = try; fit_timing(tied, stats, n); catch e; e; end
        @test e_o isa ArgumentError && e_t isa ArgumentError
        m_o, m_t = sprint(showerror, e_o), sprint(showerror, e_t)
        for m in (m_o, m_t)
            @test occursin("tied timestamps", m)
            @test occursin("events 1–2", m)              # WHICH events
            @test occursin("t = 1.0", m)                 # at WHICH time
        end
        # ... but for DIFFERENT reasons, and the messages say which
        @test occursin("likelihood over the ORDER", m_o)
        @test occursin(":breslow", m_o) && occursin(":efron", m_o)
        @test occursin("EXACT-TIME likelihood", m_t)
        @test occursin("probability ZERO", m_t)
        @test occursin(":batch", m_t)

        # --- each model refuses the policies that are not DEFINED for it ------
        # (the sprint's governing rule: an unimplemented option fails loudly)
        e_batch = try; fit_obpm(tied, stats, n; ties=:batch); catch e; e; end
        @test e_batch isa ArgumentError
        @test occursin("`:batch` is not defined", sprint(showerror, e_batch))
        @test occursin("IS the Breslow correction", sprint(showerror, e_batch))

        for bad in (:breslow, :efron)
            e_bad = try; fit_timing(tied, stats, n; ties=bad); catch e; e; end
            @test e_bad isa ArgumentError
            m = sprint(showerror, e_bad)
            @test occursin("`:$bad` is not defined", m)
            # ... and WHY: there is no partial-likelihood denominator to re-weight
            @test occursin("PARTIAL likelihood", m)
            @test occursin("fit_obpm", m)                # where it DOES apply
        end

        # ... and both refuse a symbol outside the shared vocabulary
        for f in (fit_obpm, fit_timing)
            e_junk = try; f(tied, stats, n; ties=:jitter); catch e; e; end
            @test e_junk isa ArgumentError
            @test occursin("unknown tie policy", sprint(showerror, e_junk))
            @test occursin("Networks.TIE_POLICIES", sprint(showerror, e_junk))
        end

        # --- on TIE-FREE data every policy is a no-op -------------------------
        # A tie correction that changes anything on data without ties is not a
        # tie correction. This is the sharpest check available, in both models.
        u_ord = [fit_obpm(untied, stats, n; ties=t)
                 for t in (:error, :ordered, :breslow, :efron)]
        for f in u_ord[2:end]
            @test coef(f) == coef(u_ord[1])
            @test stderror(f) == stderror(u_ord[1])
            @test f.loglik == u_ord[1].loglik
        end
        u_tim = [fit_timing(untied, stats, n; ties=t, t_end=7.0)
                 for t in (:error, :ordered, :batch)]
        for f in u_tim[2:end]
            @test coef(f) == coef(u_tim[1])
            @test f.baseline_params == u_tim[1].baseline_params
            @test f.loglik == u_tim[1].loglik
        end
        # ... and none of them claims to have corrected anything
        @test all(Networks.tie_method(f) === :none for f in vcat(u_ord, u_tim))
        @test all(Networks.is_exact(f) for f in vcat(u_ord, u_tim))
        @test all(isempty(Networks.approximations(f)) for f in vcat(u_ord, u_tim))

        # --- ordinal: the three policies genuinely differ on tied data --------
        o = fit_obpm(tied, stats, n; ties=:ordered)
        b = fit_obpm(tied, stats, n; ties=:breslow)
        ef = fit_obpm(tied, stats, n; ties=:efron)
        @test coef(o) != coef(b) && coef(b) != coef(ef)
        @test Networks.tie_method(o) === :ordered
        @test Networks.tie_method(b) === :breslow
        @test Networks.tie_method(ef) === :efron
        # A tied fit is no longer EXACT: the order the likelihood is over is not
        # in the data, whichever policy is used to stand in for it
        @test !Networks.is_exact(o) && !Networks.is_exact(b) && !Networks.is_exact(ef)
        @test any(occursin("BRESLOW correction", a) for a in Networks.approximations(b))
        @test any(occursin("EFRON correction", a) for a in Networks.approximations(ef))
        @test any(occursin("ordered arbitrarily", a) for a in Networks.approximations(o))
        @test occursin("Tied event times: breslow", sprint(show, b))
        @test !occursin("Tied event times", sprint(show, u_ord[1]))

        # --- what the corrections actually DO: freeze the tie block -----------
        # Under `:ordered` the 2→1 event tied with 1→2 scores a p-shift AB-BA of
        # 1: it "responded" to an event that did not precede it. Under a
        # correction the history is frozen across the block and it cannot.
        _, ci_o, X_o, _, W_o = Relevent._risk_set_stats(tied, Tuple(stats), n;
                                                        ties=:ordered)
        _, ci_b, X_b, _, W_b = Relevent._risk_set_stats(tied, Tuple(stats), n;
                                                        ties=:breslow)
        @test X_o[2][ci_o[2], 1] == 1.0     # the invented instantaneous response
        @test X_b[2][ci_b[2], 1] == 0.0     # a tie cannot see itself
        @test W_o === nothing && W_b === nothing        # neither is weighted
        # ... and the block is absorbed as a WHOLE (frozen ≠ dropped): by the
        # third event, both tied events are in the history
        @test X_b[3][ci_b[3], 2] == 0.0     # 1→3 has no prior 1→3 yet ...
        @test X_o[6][ci_o[6], 2] == 0.0     # (sanity: 2→3 has no prior 2→3)
        # the (1,2) dyad's PriorInteraction in interval 3 is positive under BOTH:
        # freezing suppresses the tie WITHIN its block, not afterwards
        d12 = findfirst(==((1, 2)), [(s, r) for s in 1:n for r in 1:n if s != r])
        @test X_b[3][d12, 2] > 0.0

        # --- Efron's weights ---------------------------------------------------
        _, _, _, _, W_e = Relevent._risk_set_stats(tied, Tuple(stats), n; ties=:efron)
        @test W_e !== nothing
        @test all(==(1.0), W_e[1])                       # j = 1: weight 1 − 0/2
        @test count(==(0.5), W_e[2]) == 2                # j = 2: the two tied cases
        @test count(==(1.0), W_e[2]) == n * (n - 1) - 2  # ... and nobody else
        @test all(==(1.0), W_e[3])                       # untied interval, untouched
        # Efron needs the tied cases to be DISTINCT dyads: a dyad competing with
        # itself has no fractional weight, so refuse rather than invent one
        dup = [Event(1, 2, 1.0), Event(1, 2, 1.0), Event(2, 3, 2.0)]
        e_dup = try; fit_obpm(dup, stats, n; ties=:efron); catch e; e; end
        @test e_dup isa ArgumentError
        @test occursin("distinct dyads", sprint(showerror, e_dup))
        @test fit_obpm(dup, stats, n; ties=:breslow).converged   # Breslow is fine

        # --- timing: :ordered vs :batch ---------------------------------------
        # For the exact-time likelihood the tied intervals have Δt = 0, so the
        # policies can only differ through the tied events' OWN statistics — which
        # is exactly where the arbitrary order does its damage (the AB-BA p-shift
        # above). They differ here, and `:batch` is the one that does not claim an
        # event caused another in no time at all.
        to = fit_timing(tied, stats, n; ties=:ordered, t_end=5.0)
        tb = fit_timing(tied, stats, n; ties=:batch, t_end=5.0)
        @test coef(to) != coef(tb)
        @test Networks.tie_method(to) === :ordered
        @test Networks.tie_method(tb) === :batch
        # An exact-TIME likelihood on tied data is never exact: the data
        # contradicts the process, it does not merely under-determine it
        @test !Networks.is_exact(to) && !Networks.is_exact(tb)
        @test any(occursin("ZERO-LENGTH waiting interval", a)
                  for a in Networks.approximations(to))
        @test any(occursin("simultaneous BATCH", a) for a in Networks.approximations(tb))
        @test any(occursin("COARSENED observation process", a)
                  for a in Networks.approximations(tb))
        @test occursin("probability zero", sprint(show, tb))

        # --- fit_relevent forwards the policy to whichever likelihood it picks -
        @test Networks.tie_method(
            fit_relevent(tied, stats, n; ties=:efron)) === :efron
        @test Networks.tie_method(
            fit_relevent(tied, stats, n; ordinal=false, ties=:batch)) === :batch
        @test_throws ArgumentError fit_relevent(tied, stats, n)               # ordinal
        @test_throws ArgumentError fit_relevent(tied, stats, n; ordinal=false)
        # ... and a policy sent to the likelihood it is not defined for is
        # refused, not quietly ignored
        @test_throws ArgumentError fit_relevent(tied, stats, n; ordinal=false,
                                                ties=:efron)
        @test_throws ArgumentError rem_dyad(tied, stats, n; ties=:batch)

        # --- ONE vocabulary, shared with REM.jl --------------------------------
        # The same symbols mean the same things in `REM.fit_rem`, which is the
        # point of defining them in Networks.jl rather than twice.
        seq = EventSequence(tied; actors=1:n)
        rstats = [REM.Repetition(), REM.Reciprocity()]
        @test_throws ArgumentError REM.fit_rem(seq, rstats; n_controls=11)
        rem_ef = REM.fit_rem(seq, rstats; n_controls=11, ties=:efron)
        @test Networks.tie_method(rem_ef) === :efron
        @test !Networks.is_exact(rem_ef)
    end
end
