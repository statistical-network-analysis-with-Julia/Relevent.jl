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

## Participation Shifts

The 13 Gibson (2003) participation-shift indicators, named as in R
relevent (`PSAB-BA`, ...).

```@docs
PShift
pshift_types
```

## Covariate Effects

Actor-covariate effects as in R relevent.

```@docs
CovSnd
CovRec
CovInt
```

## Utility Functions

```@docs
rank_events
```

## Cumulative R Catalogue

See [R effect coverage](../guide/r_catalogue.md) for mathematical conventions and limits.

```@docs
NIDSnd
NIDRec
NODSnd
NODRec
NTDegSnd
NTDegRec
RRecSnd
RSndSnd
OTPSnd
ITPSnd
ISPSnd
FESnd
FERec
FEInt
CovEvent
```
