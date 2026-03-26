using Relevent
using Test

@testset "Relevent.jl" begin
    @testset "Module loading" begin
        @test @isdefined(Relevent)
    end

    @testset "InteractionHistory construction" begin
        h = InteractionHistory()
        @test h isa InteractionHistory{Float64}
        @test isempty(h.events)
        @test isempty(h.event_counts)

        h2 = InteractionHistory{Int}()
        @test h2 isa InteractionHistory{Int}
    end

    @testset "InteractionHistory queries" begin
        h = InteractionHistory()
        @test get_interaction_count(h, 1, 2) == 0
        @test get_last_interaction(h, 1, 2) === nothing
    end

    @testset "Advanced REM statistics" begin
        @test PriorInteraction(1.0) isa PriorInteraction
        @test PriorInteraction(1.0; direction=:incoming) isa PriorInteraction
        @test_throws ArgumentError PriorInteraction(1.0; direction=:invalid)

        @test SendingCapacity(1.0) isa SendingCapacity
        @test ReceivingCapacity(1.0) isa ReceivingCapacity
        @test LocalInertia(1.0) isa LocalInertia
        @test Momentum(1.0) isa Momentum
        @test Momentum(1.0; normalize=true) isa Momentum
    end

    @testset "Ordinal BPM" begin
        @test isdefined(Relevent, :OrdinalBPM)
        @test isdefined(Relevent, :OrdinalBPMResult)
        @test isdefined(Relevent, :fit_obpm)
        @test isdefined(Relevent, :rank_events)
    end

    @testset "Timing models" begin
        tm = TimingModel(Relevent.AbstractStatistic[])
        @test tm isa TimingModel
        @test tm.baseline == :exponential

        tm2 = TimingModel(Relevent.AbstractStatistic[]; baseline=:weibull)
        @test tm2.baseline == :weibull

        @test_throws ArgumentError TimingModel(Relevent.AbstractStatistic[]; baseline=:invalid)
    end

    @testset "CumulativeState construction" begin
        cs = CumulativeState(5)
        @test cs isa CumulativeState{Float64}
        @test cs.n_actors == 5
        @test size(cs.adj_matrix) == (5, 5)

        @test get_outdegree_history(cs, 1) == 0.0
        @test get_indegree_history(cs, 1) == 0.0
    end

    @testset "Hazard and survival functions" begin
        @test isdefined(Relevent, :hazard_rate)
        @test isdefined(Relevent, :survival_function)
    end
end
