# Direct computation cost of gradient proxies

Last modified: 2026-11-08

## Protocol

Measured on the same Chronos-T5 Tiny warmed checkpoint and ten M5 trajectories used by v0-gradient-proxy-test. Eight evenly spaced windows per trajectory (80 total) were selected, each method was timed three times per window at batch size one, and method order was randomized after three warmup calls per method. All methods process the same window and include CPU tokenization, model forward, method-specific computation, copying the result to CPU, and CUDA synchronization. Full BP returns the complete gradient converted to BF16 for storage; the head proxy returns all output-weight coordinates converted to BF16. Embedding and loss return their respective vectors/scalars.

Hardware: NVIDIA GeForce RTX 4080. The measurements describe this implementation, precision, device, and batch size.

## Results

| Method | Median ms/window | Mean ms/window | P90 ms/window | Paired speedup vs full BP | Per-window speedup range | Across-track median range (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Full BP | 42.80 | 42.79 | 47.29 | 1.00x | 1.00x-1.00x | 40.29-44.62 |
| Output head gradient | 14.21 | 14.75 | 16.96 | 2.93x | 2.17x-3.83x | 14.02-14.79 |
| Mean decoder embedding | 11.35 | 11.65 | 13.03 | 3.72x | 2.92x-4.86x | 11.04-11.76 |
| Forward loss | 11.28 | 11.59 | 12.55 | 3.71x | 2.84x-4.66x | 11.03-12.05 |

Each row has 240 timed calls: 80 windows times three repeats. Median speedup compares paired per-window median latency against full BP. The per-window range and across-track range are descriptive, not confidence intervals.

## Validation

The head proxy direction matched the saved head-gradient cache on one sampled window per trajectory (cosine greater than 0.999). The mean embedding matched the saved feature with maximum absolute error at most 1.9e-06; forward loss differed by at most 4.8e-07. All ten trajectory checks passed.

Raw repeat timings and exact sample starts: outputs/train/v0-gradient-proxy-cost/per_window_timings.csv. Machine-readable summary: outputs/train/v0-gradient-proxy-cost/summary.json.

## Interpretation

On this GPU and implementation, output-head gradients take about one third of full BP time. Embedding means and loss are each about 3.7 times faster than full BP, and cost almost the same as each other because both require the same model forward; extracting the embedding adds little overhead. The head proxy costs more because it also computes the loss derivative with respect to logits and forms the output-weight gradient.

These times include data preparation and result transfer. They do not include large-scale storage throughput, concurrent CRB replay, or a different batching strategy. The CRB queue remains paused under its STOP marker after the observed container memory pressure.
