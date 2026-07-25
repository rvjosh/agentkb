# Modal GPU follow-up findings

Use baseline T4 for interactive SessionStart warming. GPU snapshots worked, but the measured reduction would save only a trivial amount per week for that workload and the feature remains alpha, so snapshots are not worth first-deployment complexity. Use L4 as the candidate for refresh/indexing jobs: this one staged 1,411-document build took 28.04 seconds internally versus 68.37 seconds on T4 and was cheaper on measured container time.

## Search

| Variant | Cold samples | Client p50 | Container p50 | Warm query p50 | Published GPU cost at container p50 |
|---|---:|---:|---:|---:|---:|
| T4 baseline | 3 fresh | 25.41 s | 16.42 s | 116.83 ms | $0.002692 |
| L4 baseline | 3 fresh | 27.59 s | 19.60 s | 83.64 ms | $0.004352 |
| T4 GPU snapshot | 4 verified restored | 15.97 s | 1.99 s after restore | — | $0.000326 |

T4 cold initialization p50 was 15.10 seconds: model 13.34 seconds, Volume copy 0.78 seconds, metadata 0.03 seconds, and FastPLAID 0.91 seconds. L4 initialization p50 was 18.62 seconds: model 17.22 seconds, Volume copy 0.25 seconds, metadata 0.02 seconds, and FastPLAID 1.06 seconds.

The 12-query warm passes had p50 stage splits of 24.53 ms query encoding, 45.77 ms semantic search, 5.19 ms FTS, and 32.31 ms hydration on T4; L4 measured 29.39 ms, 20.51 ms, 4.66 ms, and 29.08 ms respectively. First-call client totals, including cold initialization and all 12 queries, were 29.11 seconds for T4 and 26.84 seconds for L4.

## GPU snapshot

The snapshot boundary contained model initialization and a representative query forward pass only. Its median captured work was 15.40 seconds for model loading plus 0.46 seconds for the forward pass. Volume copy, temporary-directory creation, read-only metadata opening, and FastPLAID loading ran after restore; their combined restored p50 was 1.36 seconds.

Six forced-fresh calls were made. Exact-app logs contained two instances of Modal's documented `Snapshot created. Restoring Function from memory snapshot.` evidence, followed by four calls without another creation. All six emitted the post-restore marker. The first two calls are classified as snapshot creation; the remaining four are verified restored cold samples.

Restored client p50 improved 37.16% against the same-deployment T4 baseline and 12.83% against the prior 18.32-second T4 p50. The client p95 was still 27.44 seconds, and the result depends on an alpha feature with worker-specific snapshot creation. For SessionStart warming, the resulting weekly saving is trivial and does not justify first-deployment complexity; snapshots remain a follow-up candidate after Modal promotes or further stabilizes the feature.

## Staged 1,411-document build

| GPU | Client total | Internal total | Model load | Encode | SQLite | FastPLAID | Published GPU cost at internal total |
|---|---:|---:|---:|---:|---:|---:|---:|
| T4 | 99.73 s | 68.37 s | 25.34 s | 26.86 s | 2.34 s | 38.95 s | $0.011212 |
| L4 | 50.35 s | 28.04 s | 17.10 s | 16.24 s | 2.07 s | 9.50 s | $0.006224 |

`internal total` starts after model load, matching the predeclared build gate; model load is reported separately. L4 passed the 41.37-second build gate but its 19.60-second cold container p50 failed the 13.53-second cold gate. That makes L4 the candidate for refresh/indexing jobs, while interactive warming stays on T4. This is only one staged-build sample, not a general performance guarantee.

Published rates checked on 2026-07-25 were $0.000164/T4-second and $0.000222/L4-second. Modal's authoritative hourly billing report returned two exact-app rows totaling **$0.22216071** for the benchmark. Published-rate figures above are GPU-only estimates from the named internal timing basis; actual billing includes the platform's billable resources and lifecycle time.

No A10 or larger GPU was tested. No SessionStart hook was wired during measurement.
