# Advanced Statistics

Relevent.jl provides advanced statistics for relational event models that complement the basic statistics in REM.jl. All statistics use half-life decay and operate on the full interaction history.

## Statistics Interface

All Relevent.jl statistics implement:

```julia
compute(stat, history, sender, receiver, current_time) -> Float64
```

The `history` is an `InteractionHistory` and `current_time` is the time at which the statistic is evaluated. All statistics apply exponential decay based on their `halflife` parameter.

## PriorInteraction

Decayed count of prior interactions between two actors. This is the most general interaction history statistic.

```julia
# Count prior outgoing events from sender to receiver
PriorInteraction(halflife; direction=:outgoing)

# Count prior incoming events (receiver to sender)
PriorInteraction(halflife; direction=:incoming)

# Count events in both directions
PriorInteraction(halflife; direction=:both)
```

### How It Works

For `direction=:outgoing`, the statistic computes:

$$\text{PI}(s, r, t) = \sum_{k: s_k=s, r_k=r, t_k < t} \exp\left(-\frac{\log 2}{h} \cdot (t - t_k)\right)$$

Where $h$ is the halflife and the sum is over all past events from sender $s$ to receiver $r$.

### Interpretation

- **Positive coefficient**: Actors who have interacted before are more likely to interact again
- **With direction=:both**: Captures overall familiarity between actors
- **With direction=:incoming**: Captures reciprocity-like effects with decay

### Example

```julia
prior = PriorInteraction(10.0; direction=:outgoing)

# After events: 1->2 at t=0, 1->2 at t=5
# At t=10:
# Event at t=0: weight = exp(-log(2)/10 * 10) = 0.5
# Event at t=5: weight = exp(-log(2)/10 * 5) = 0.707
# Total: 1.207

value = compute(prior, history, 1, 2, 10.0)
```

## LocalInertia

Measures the recency of the last interaction on a specific dyad. Captures the tendency for repeat interactions -- actors who just communicated are likely to communicate again.

```julia
LocalInertia(halflife)
```

### How It Works

$$\text{LI}(s, r, t) = \exp\left(-\frac{\log 2}{h} \cdot (t - t_{\text{last}})\right)$$

Where $t_{\text{last}}$ is the time of the most recent event from $s$ to $r$. Returns 0.0 if no prior event exists.

### Interpretation

- **Positive coefficient**: Recent interactions increase the rate of future interaction
- **Halflife controls memory**: Short halflife = only very recent contacts matter
- **Compared to PriorInteraction**: LocalInertia only considers the last event, not all past events

### Example

```julia
inertia = LocalInertia(5.0)

# If last event 1->2 was at t=3, and current time is t=8:
# value = exp(-log(2)/5 * (8-3)) = exp(-log(2)) = 0.5
value = compute(inertia, history, 1, 2, 8.0)
```

## SendingCapacity

Measures the sender's overall activity level based on past sending behavior.

```julia
SendingCapacity(halflife)
```

### How It Works

Computes the decayed count of all events sent by the sender to any receiver:

$$\text{SC}(s, t) = \sum_{k: s_k=s, t_k < t} \exp\left(-\frac{\log 2}{h} \cdot (t - t_k)\right)$$

### Interpretation

- **Positive coefficient**: Active senders continue to be active
- **Matthew effect**: High-activity actors generate more events
- **Controls for baseline activity**: Important to include alongside dyad-level statistics

### Example

```julia
send_cap = SendingCapacity(10.0)

# Measures how active the sender has been recently
value = compute(send_cap, history, 1, 2, 10.0)
```

## ReceivingCapacity

Measures the receiver's popularity based on past events received.

```julia
ReceivingCapacity(halflife)
```

### How It Works

Computes the decayed count of all events received by the receiver from any sender:

$$\text{RC}(r, t) = \sum_{k: r_k=r, t_k < t} \exp\left(-\frac{\log 2}{h} \cdot (t - t_k)\right)$$

### Interpretation

- **Positive coefficient**: Popular receivers continue to attract interactions
- **Preferential attachment**: More popular actors receive more events
- **Controls for popularity differences**: Important baseline control

### Example

```julia
recv_cap = ReceivingCapacity(10.0)

# Measures how popular the receiver is
value = compute(recv_cap, history, 1, 2, 10.0)
```

## Momentum

Measures the sender's overall activity momentum across all receivers.

```julia
Momentum(halflife; normalize=false)
```

### How It Works

Similar to SendingCapacity but sums decayed event counts across all unique receivers:

$$\text{Mom}(s, t) = \sum_{r} \sum_{k: s_k=s, r_k=r, t_k < t} \exp\left(-\frac{\log 2}{h} \cdot (t - t_k)\right)$$

If `normalize=true`, divides by the total number of events observed.

### Interpretation

- **Positive coefficient**: Senders with recent bursts of activity continue to be active
- **normalize=true**: Controls for overall event volume
- **Compared to SendingCapacity**: Momentum aggregates across unique receivers

### Example

```julia
# Without normalization
mom = Momentum(10.0)
value = compute(mom, history, 1, 2, 10.0)

# With normalization
mom_norm = Momentum(10.0; normalize=true)
value_norm = compute(mom_norm, history, 1, 2, 10.0)
```

## Using Statistics in Models

### Building a Comprehensive Model

```julia
stats = [
    # Dyad-level effects
    LocalInertia(10.0),                           # Recent repeat interaction
    PriorInteraction(10.0; direction=:outgoing),  # Past contact
    PriorInteraction(10.0; direction=:incoming),  # Reciprocity

    # Actor-level effects
    SendingCapacity(10.0),                        # Sender activity
    ReceivingCapacity(10.0),                      # Receiver popularity
    Momentum(10.0),                               # Activity momentum
]
```

### Combining with REM.jl Statistics

Relevent.jl statistics work alongside REM.jl statistics:

```julia
using REM

stats = [
    # REM.jl statistics
    Repetition(),
    Reciprocity(),
    TransitiveClosure(),

    # Relevent.jl statistics
    LocalInertia(10.0),
    PriorInteraction(10.0; direction=:both),
]
```

## Choosing Halflife Values

### By Application Domain

| Domain | Typical Halflife |
|--------|------------------|
| Real-time messaging | 1-10 minutes |
| Email | 1-24 hours |
| Social media posts | 1-7 days |
| Business relationships | 1-4 weeks |
| Academic collaboration | 1-5 years |

### Sensitivity Analysis

Try multiple halflife values and compare:

```julia
for hl in [1.0, 5.0, 10.0, 50.0, 100.0]
    stats = [
        LocalInertia(hl),
        PriorInteraction(hl; direction=:outgoing),
        SendingCapacity(hl),
    ]

    # Fit and compare results
    result = fit_rem(seq, stats; n_controls=50, seed=42)
    println("Halflife $hl: LL=$(round(result.log_likelihood, digits=2))")
end
```

## Best Practices

1. **Include both dyad and actor effects**: Combine LocalInertia/PriorInteraction with Capacity statistics
2. **Match halflife to data**: Use domain knowledge to set appropriate decay rates
3. **Test multiple directions**: For PriorInteraction, try :outgoing, :incoming, and :both
4. **Normalize when appropriate**: Use `Momentum(hl; normalize=true)` for networks with many events
5. **Avoid collinearity**: LocalInertia and PriorInteraction may be correlated -- check VIFs
6. **Sufficient history**: Statistics need enough prior events to be informative
