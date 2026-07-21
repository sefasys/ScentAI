# Evaluation

## Frozen V4 Suite

The public evaluation pack contains 120 cases split across:

- recommendation: 20;
- comparison: 15;
- hard filters: 15;
- perfume profiles: 15;
- similarity and alternatives: 15;
- conversations: 10;
- entity resolution: 10;
- multilingual and noisy input: 10;
- explicitly unsupported requests: 10.

There are 61 English and 59 Turkish cases.

## Results

The evaluated pipeline passed all 120 functional cases. Language, requested-count, hard-filter, entity-resolution, conversation no-repeat, and performance-calibration checks all reached 100%.

Warm evaluation latency:

- average: 9.27 seconds;
- median: 9.43 seconds;
- p95: 14.29 seconds;
- maximum: 21.34 seconds.

The first generation passed in 90.91% of generation-bearing cases. Eight requests used validated template fallback, producing a 6.67% fallback rate. That exceeded the original 5% target, so the report correctly records `all_quality_gates_passed: false` even though every functional case passed.

## What These Numbers Do Not Prove

- They do not measure every possible perfume-name abbreviation.
- They do not establish factual ownership or licensing of the source catalog.
- They do not measure current prices, stock, reformulations, or batch variation.
- They do not replace broad human preference studies.
- Latency excludes scale-to-zero cold start and depends on hardware and cache state.
- The fixed set was developed alongside the system, so a future blind holdout is still needed.

The compact source files are available in [`evaluation/`](../evaluation/). Full generated answers and large cloud reports are intentionally not committed.

