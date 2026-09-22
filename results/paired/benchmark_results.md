# Measured paired MLX results

| Variant | Correct / total | Accuracy | Median latency | Peak MLX | Input tokens/s | Decisions/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 186/231 | 80.52% | 694.1 ms | 8.997 GiB | 382.50 | 0.548 |
| quantized | 180/231 | 77.92% | 566.1 ms | 3.370 GiB | 384.80 | 0.551 |

Quantized minus baseline: -2.60 pp; paired bootstrap 95% interval: [-6.926406926406926, 1.7316017316017316] pp.

Regressions: 15; improvements: 9.

![Paired MLX accuracy and resources](comparison.png)

Public development data, not a held-out test. Related items are treated as independent by the bootstrap.
Timing includes prompt construction, tokenization and synchronized inference; excludes loading and warmup.
Peak MLX includes live weights; RSS is a post-run snapshot. Do not add overlapping memory counters.
