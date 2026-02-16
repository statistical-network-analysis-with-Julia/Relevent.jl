# Interaction History

Relevent.jl provides the `InteractionHistory` type for detailed tracking of relational event sequences. This enables advanced statistics that depend on the full history of interactions between actors, not just aggregate counts.

## Overview

The interaction history maintains:

- A chronological record of all events
- Per-sender history: which receivers each actor has contacted
- Per-receiver history: which senders have contacted each actor
- Per-dyad history: all event times for each sender-receiver pair
- Per-dyad counts: total number of events for each pair

## Creating an InteractionHistory

```julia
using REM
using Relevent

# Create an empty history tracker
history = InteractionHistory{Float64}()

# Or with a specific timestamp type
history_dt = InteractionHistory{DateTime}()
history_int = InteractionHistory{Int}()
```

The type parameter matches the timestamp type of your events.

## Adding Events

Use `update_history!` to add events one at a time:

```julia
events = [
    Event(1, 2, 1.0),
    Event(2, 1, 2.0),
    Event(1, 3, 3.0),
    Event(3, 2, 4.0),
    Event(1, 2, 5.0),
]

for event in events
    update_history!(history, event)
end
```

Each call to `update_history!` updates all internal tracking structures atomically.

## Querying History

### Event Counts

```julia
# How many times has actor 1 sent to actor 2?
get_interaction_count(history, 1, 2)  # 2

# How many times has actor 2 sent to actor 1?
get_interaction_count(history, 2, 1)  # 1

# How many times has actor 1 sent to actor 3?
get_interaction_count(history, 1, 3)  # 1

# Non-existent dyad
get_interaction_count(history, 3, 1)  # 0
```

### Last Interaction Time

```julia
# When did actor 1 last send to actor 2?
get_last_interaction(history, 1, 2)  # 5.0

# When did actor 2 last send to actor 1?
get_last_interaction(history, 2, 1)  # 2.0

# Non-existent dyad returns nothing
get_last_interaction(history, 3, 1)  # nothing
```

## Accessing Internal Data

### Sender History

A dictionary mapping each actor to a list of their receivers (in chronological order):

```julia
# All receivers of actor 1 (in order)
history.sender_history[1]  # [2, 3, 2]

# All receivers of actor 2
history.sender_history[2]  # [1]

# All receivers of actor 3
history.sender_history[3]  # [2]
```

### Receiver History

A dictionary mapping each actor to a list of senders who contacted them:

```julia
# All senders to actor 2
history.receiver_history[2]  # [1, 3, 1]

# All senders to actor 1
history.receiver_history[1]  # [2]

# All senders to actor 3
history.receiver_history[3]  # [1]
```

### Pair History

A dictionary mapping each (sender, receiver) pair to their event times:

```julia
# All event times for dyad (1, 2)
history.pair_history[(1, 2)]  # [1.0, 5.0]

# All event times for dyad (2, 1)
history.pair_history[(2, 1)]  # [2.0]

# Check if a dyad has any history
haskey(history.pair_history, (3, 1))  # false
```

### Event Counts

A dictionary with the total count per dyad:

```julia
history.event_counts[(1, 2)]  # 2
history.event_counts[(2, 1)]  # 1
```

### All Events

The complete event list in chronological order:

```julia
length(history.events)  # 5
history.events[1]       # Event(1, 2, 1.0)
history.events[end]     # Event(1, 2, 5.0)
```

## Use Cases

### Tracking Communication Patterns

```julia
# Which actors communicate most?
all_dyads = collect(keys(history.event_counts))
sorted = sort(all_dyads, by=d -> history.event_counts[d], rev=true)

for dyad in sorted
    println("$(dyad[1]) -> $(dyad[2]): $(history.event_counts[dyad]) events")
end
```

### Finding Reciprocated Relationships

```julia
# Check for reciprocity
for (sender, receiver) in keys(history.event_counts)
    reverse_count = get(history.event_counts, (receiver, sender), 0)
    if reverse_count > 0
        forward = history.event_counts[(sender, receiver)]
        println("$sender <-> $receiver: $forward forward, $reverse_count reverse")
    end
end
```

### Computing Time Between Interactions

```julia
# Inter-event times for a specific dyad
times = history.pair_history[(1, 2)]
if length(times) >= 2
    inter_times = diff(times)
    println("Inter-event times for 1->2: ", inter_times)
    println("Mean inter-event time: ", sum(inter_times) / length(inter_times))
end
```

### Actor Activity Profiles

```julia
# Activity profile for actor 1
n_sent = length(get(history.sender_history, 1, Int[]))
n_received = length(get(history.receiver_history, 1, Int[]))
unique_receivers = length(unique(get(history.sender_history, 1, Int[])))
unique_senders = length(unique(get(history.receiver_history, 1, Int[])))

println("Actor 1:")
println("  Events sent: $n_sent")
println("  Events received: $n_received")
println("  Unique receivers: $unique_receivers")
println("  Unique senders: $unique_senders")
```

## Cumulative Network State

For statistics that need a decaying adjacency representation, use `CumulativeState`:

```julia
# Create with 10-unit halflife
state = CumulativeState{Float64}(n_actors; halflife=10.0)

# Update with events
for event in events
    update_state!(state, event)
end

# Query decayed degrees
get_outdegree_history(state, 1)  # Decayed out-degree of actor 1
get_indegree_history(state, 2)   # Decayed in-degree of actor 2

# Decayed adjacency matrix
state.adj_matrix  # n_actors x n_actors matrix
```

### How Decay Works

Each time `update_state!` is called:

1. All existing weights are multiplied by $\exp(-\lambda \cdot \Delta t)$ where $\Delta t$ is the time elapsed
2. The new event adds 1.0 to the appropriate adjacency entry
3. Degrees are updated similarly

```julia
# Example: halflife = 10
state = CumulativeState{Float64}(3; halflife=10.0)

update_state!(state, Event(1, 2, 0.0))
println(state.adj_matrix[1, 2])  # 1.0

update_state!(state, Event(1, 2, 10.0))
# First event decayed: exp(-log(2)/10 * 10) = 0.5
# New event: 1.0
println(state.adj_matrix[1, 2])  # 1.5
```

## Integration with Statistics

InteractionHistory is used by Relevent.jl's advanced statistics:

```julia
# Statistics that use history
prior = PriorInteraction(10.0; direction=:outgoing)
inertia = LocalInertia(10.0)
momentum = Momentum(10.0)

# Compute using history
value = compute(prior, history, sender, receiver, current_time)
```

See [Advanced Statistics](statistics.md) for details on each statistic.

## Best Practices

1. **Process events chronologically**: Update history in time order for consistency
2. **Match timestamp types**: Use the same type for history and events
3. **Choose appropriate halflife**: Match the temporal scale of your data
4. **Query efficiently**: Use `get_interaction_count` and `get_last_interaction` for common queries
5. **Avoid modifying internals**: Use `update_history!` to add events
