# Marrow - behavioural user verification prototype

Decides whether a session belongs to the enrolled user from ordinary input events only (pointer movement, clicks, text), with no audio, camera or device assumption.

Method: one behavioural fingerprint per recording (median/p90 pointer speed, mean direction change, share of long pauses, median text interval, interval CV, correction rate, burst length); per-feature median/MAD robust z-scores; score = RMS of the three largest deviations; threshold fixed at enrolment by leave-one-session-out and frozen before testing. Synthetic data only.

## Measured under ingest jitter

Constant delay is invisible: every feature is a difference, so a uniform offset cancels (distance 0.9800 at +0, +500 and +5000 ms). Per-event arrival jitter is not invisible:

| added jitter | distance (owner session 6) | verdict |
| --- | --- | --- |
| 0 ms | 0.98 | accept |
| +15 ms | 1.29 | accept |
| +30 ms | 3.58 | reject |
| +60 ms | 5.15 | reject |
| +120 ms | 6.00 | reject |

Session FRR 25% -> 100% from +30 ms on style A; FAR stayed 0.0% at every level, so the model fails closed.

Reference implementation: `model.mjs`.
