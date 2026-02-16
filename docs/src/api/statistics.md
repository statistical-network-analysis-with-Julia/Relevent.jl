# Statistics API Reference

This page documents all statistics available in Relevent.jl.

## Dyad-Level Statistics

Statistics based on the detailed interaction history between the focal sender-receiver pair.

```@docs
PriorInteraction
LocalInertia
```

## Actor-Level Statistics

Statistics based on actor activity and popularity levels with exponential decay.

```@docs
SendingCapacity
ReceivingCapacity
Momentum
```

## Utility Functions

```@docs
rank_events
```
