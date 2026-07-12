using Relevent
using REM
using Distributions
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
        # (Network.jl print_coeftable: z / Pr(>|z|) columns + signif codes)
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
end
