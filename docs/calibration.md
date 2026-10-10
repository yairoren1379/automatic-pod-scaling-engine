# Calibrating the mock environment against the real cluster

The DQN is pre-trained in the mock environment, so the environment must behave like the real
system where it matters. The first version was designed from the proposal alone; this page records
how it was measured against the real target app on Minikube and corrected (stage 1 → 2 of the
sim-to-real pipeline).

## Measurement

- Target app (`back/go/targetapp`): 2 ms of CPU work + 20 ms simulated downstream call per request,
  512 KB working buffer per in-flight request, 128 KB response cache kept for 10 s, at most 200
  requests in flight. Pod limits: 200m CPU, 256Mi memory.
- The Deployment was pinned to 2 pods (`kubectl scale` + guardrails min = max = 2) and the load
  generator stepped the request rate; each level ran for 100 s and the controller's readings
  (30 s apart) after the first 35 s were averaged.
- Raw numbers: [`calibration_measurements.json`](calibration_measurements.json).

| Requests/s per pod | CPU | Memory | Latency | Errors | Notes |
|---|---|---|---|---|---|
| 30 | 39% | 27% | 23 ms | 0% | |
| 60 | 61% | 61% | 25 ms | 0% | latency still flat |
| 90 | 88% | 72% | 1,336 ms | 15.6% | the cliff: requests queue behind the CPU limit |
| 110 | 50% | 43% | 611 ms | 15.6% | pods restarting (liveness probe failures) |
| 130 | 53% | 48% | 1,390 ms | 43.8% | pods restarting |
| 150 | 100% | 86% | 905 ms | 29.4% | |

Both pods restarted 3 times under overload (reason `Error`: liveness probes timed out while the
pod was CPU-starved), and the controller reported `crashed=true`.

## What did not match the first simulator

| | First simulator | Real system |
|---|---|---|
| Latency vs load | rises smoothly with memory: ~70 ms at 60% memory, ~240 ms at 80% | flat ~23–25 ms, then a cliff near the CPU limit |
| Cause of high latency | memory pressure | CPU throttling → queueing |
| Errors start at | utilization > 1.0 | ~0.85 (in-flight cap) |
| Crashes | only OOM at 98% memory | liveness restarts when CPU-starved |
| New pod ready after | 2 steps (60 s) | < 10 s (image is local, Go starts instantly) |

The consequence was visible immediately: during the overload the pre-trained DQN proposed
**ScaleDown** at 99.8% CPU and 957 ms latency, a state it had never seen in simulation
(extreme latency with moderate memory). In active mode that would have been dangerous.

## Changes to the mock environment

All values are in `mock_env` in `back/python/setup_config.py`.

- **Latency** = service time (22 ms) + memory term `3 ms · e^(6 · (RAM − 0.5))` (the design's
  exponential memory curve, now a smaller component) + **queueing term**
  `1300 ms · e^(23 · (utilization − 0.88))`, capped at 1.5 s (the in-flight cap bounds the queue).
  The queueing term is smooth (no hard step) but very steep, like the measured cliff.
- **CPU** = 15% idle overhead + 85% × utilization; **memory** = 15% + 65% × utilization, with
  growth stopping at 1.1 (rejected requests don't allocate).
- **Errors** start at 85% utilization (slope 1.5); **crashes** from liveness failures with
  probability `2 · (utilization − 0.95)` per step, up to 0.5.
- **Pod lifecycle**: ready by the next control loop, one step at 70% capacity.

| Requests/s per pod | Sim CPU | Real CPU | Sim memory | Real memory | Sim latency | Real latency | Sim errors | Real errors |
|---|---|---|---|---|---|---|---|---|
| 30 | 41% | 39% | 35% | 27% | 23 ms | 23 ms | 0% | 0% |
| 60 | 66% | 61% | 54% | 61% | 27 ms | 25 ms | 0% | 0% |
| 90 | 92% | 88% | 74% | 72% | 1,500 ms | 1,336 ms | 8% | 16% |

After the change the LSTM and the DQN were retrained and the experiments re-run (README,
*Results*). The safe operating point moved: the agent must keep utilization below ~0.8, because
past ~0.85 latency crosses the 600 ms severe-failure threshold.
