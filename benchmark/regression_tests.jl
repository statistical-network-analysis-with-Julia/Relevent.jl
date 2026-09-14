# Bounded performance gate: exercises actual likelihood closures; no tuning or
# machine-dependent wall-clock thresholds. Full throughput suite is separate.
using Test, Random, Relevent, REM

function derivative_allocations(n, m; cache=:all)
    rng = Xoshiro(20260914)
    events = Event{Float64}[]
    for i in 1:m
        s = rand(rng, 1:n)
        r = rand(rng, 1:n-1)
        push!(events, Event(s, r >= s ? r + 1 : r, Float64(i)))
    end
    stats = [PShift(:AB_BA), PriorInteraction(4.0)]
    rs = Relevent._risk_sets(Relevent._risk_set_plan(events, stats, n); cache=cache)
    ordinal = Relevent._obpm_derivatives(rs)
    timing = Relevent._timing_derivatives(rs)
    θ, β = zeros(2), zeros(3)
    ordinal(θ); timing(β)
    return (@allocated ordinal(θ)), (@allocated timing(β))
end

@testset "Derivative allocation bounds" begin
    small = derivative_allocations(6, 20)
    large = derivative_allocations(14, 200)
    @test all(x -> x <= 512, small)
    @test all(x -> x <= 512, large)
    @test all(large .<= small .+ 64)
    streamed = derivative_allocations(14, 200; cache=:none)
    design_bytes = 200 * 14 * 13 * 2 * sizeof(Float64)
    @test all(streamed .< design_bytes ÷ 10)
    println("Derivative bytes (ordinal,timing): small=$small large=$large streamed=$streamed")
end
