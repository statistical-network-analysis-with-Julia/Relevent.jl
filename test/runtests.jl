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

    @testset "REM.NetworkState interface and fit_rem integration" begin
        events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0),
                  Event(2, 3, 4.0), Event(3, 1, 5.0), Event(1, 3, 6.0)]
        seq = EventSequence(events)

        # 4-arg compute agrees with the history-based 5-arg compute
        state = REM.NetworkState(seq)
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

        # Non-exponential baselines are not silently mis-fit
        @test_throws ErrorException fit_timing(events, [stat], n;
                                               baseline=:weibull)
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
