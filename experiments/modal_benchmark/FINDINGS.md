# AgentKB local versus Modal benchmark

## Decision

Modal wins decisively for warm search and materially for the fixed indexing batch, with effectively identical ranking quality. It is viable if the product can provide or tolerate warm-container behavior. It does not win as a transparent scale-to-zero interactive backend: the measured authenticated-client cold p50 was 25.47 seconds.

| Measurement | Local | Modal T4 | Decision |
|---|---:|---:|---|
| Warm search p50 | 1,720.13 ms | 114.95 ms | Modal passes the <2.0 s threshold; 14.96x faster |
| Warm search p95 | 2,829.51 ms | 569.03 ms | Modal passes the <3.0 s threshold; 4.97x faster |
| Cold client p50 | 8,297.44 ms | 25,473.38 ms | Modal fails the <7.8 s threshold and is 3.07x slower |
| Cold client p95 | 9,902.47 ms | 35,712.36 ms | Scale-to-zero tail latency is unsuitable for interactive use |
| 1,411-document batch | 188.58 s | 56.01 s internal / 56.95 s client | Modal narrowly passes <60 s and is 3.37x faster |

The clean canonical generation contained 7,561 chunks. The canonical, SQLite, FTS, forward PLAID, and reverse PLAID counts all matched, with no duplicates or orphan mappings.

## Ranking quality

Eleven of twelve ordered top-10 lists were identical. The remaining query returned the same top-10 set and the same score at ranks 6 and 7, with those tied results reversed. All twelve top-10 sets matched, all scores matched within 0.001, and the maximum observed score delta was 0.00048828125. This is effectively identical ranking quality.

## Why cold loses

The Modal cold client p50 was 25.47 seconds, of which the container p50 was 18.32 seconds. Initialization dominated: model load p50 was 15.58 seconds, index load p50 was 1.03 seconds, and the generation copy p50 was 0.31 seconds. Once initialized, the cold query itself had a 1.36-second p50. The benchmark forced fresh single-use containers and used `min_containers=0`, so it measures a defensible scale-to-zero path rather than a warmed request.

Only three cold samples per environment were run, so the interpolated p95 is descriptive rather than a stable production tail estimate.

## Cost estimate

The estimate uses Modal's published rates captured on 2026-07-25: T4 at $0.000164/s, physical CPU at $0.0000131/core/s, and memory at $0.00000222/GiB/s. The app did not explicitly request CPU or memory, so the calculation uses Modal's documented defaults of 0.125 core and 0.125 GiB. Modal bills CPU and memory at the higher of request or actual use, so actual CPU/memory cost may be higher.

Using authenticated client elapsed time as a conservative proxy, the successful measured build and benchmark calls totaled 888.86 seconds: approximately $0.1458 GPU-only or $0.1475 including minimum CPU and memory. This is not dashboard-authoritative billing and excludes the failed initial image attempt, image-build charges, actual CPU/memory utilization, storage, credits, and exact platform billing boundaries.

A representative month of 10,000 warm searches at measured p50, 100 cold searches at client p50, and 30 full batches uses 5,405.29 proxy seconds and estimates to $0.90 at minimum resource requests. That assumes the warm requests arrive while a container already exists; it does not price a strategy for maintaining warmth. For scale, a continuously allocated T4 plus minimum CPU/memory for 730 hours would be about $436.02, but this experiment did not use an always-warm deployment.

Published rate sources: [Modal pricing](https://modal.com/pricing) and [Modal CPU/memory resource billing](https://modal.com/docs/guide/resources).

## Caveats

- The local machine and Modal T4 are different hardware classes; this is an operational A/B, not a hardware-normalized microbenchmark.
- Warm measurements contain one pass over twelve fixed queries. Modal's first warm-class query is visibly slower, and p95 reflects that small sample.
- Local cold timing includes a fresh Python subprocess; Modal cold timing includes authenticated client, scheduling, container startup, generation copy, model/index initialization, and query execution.
- The full batch ran once per environment after a 10-document smoke run. The 56.01-second Modal result clears the 60-second threshold by only 3.99 seconds.
- Cost figures are estimates from published unit rates and client timing, not billing exports.
