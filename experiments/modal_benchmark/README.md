# AgentKB Modal benchmark

This contained experiment compares AgentKB's real local CPU search/index path with an authenticated, single-T4 Modal deployment. It creates a deterministic private corpus snapshot under an exact `/tmp/agentkb-modal-benchmark-*` path, builds one clean generation on Modal, downloads that same generation for local measurements, and records only sanitized aggregates in the repository.

Do not commit the temporary corpus, generation, raw result JSON, document paths, result IDs, credentials, or Modal configuration.

## Setup and run

Run from the repository root. The commands intentionally keep private artifacts outside the repository.

```bash
uv sync
uvx modal profile current

BENCHMARK_ROOT=/tmp/agentkb-modal-benchmark-20260725
uv run python -m experiments.modal_benchmark.snapshot --output "$BENCHMARK_ROOT"

uvx modal volume create agentkb-benchmark-20260725-data
uvx modal volume put agentkb-benchmark-20260725-data "$BENCHMARK_ROOT/corpus.jsonl" corpus/corpus.jsonl
uvx modal volume put agentkb-benchmark-20260725-data "$BENCHMARK_ROOT/batch-1411.jsonl" corpus/batch-1411.jsonl
uvx modal volume put agentkb-benchmark-20260725-data "$BENCHMARK_ROOT/manifest.json" corpus/manifest.json

uvx modal deploy -m experiments.modal_benchmark.modal_app \
  --name agentkb-benchmark-20260725 --strategy recreate
uv run --with modal==1.5.3 python -m experiments.modal_benchmark.modal_client build \
  --output "$BENCHMARK_ROOT/modal-build.json" --summary-only
uvx modal volume get agentkb-benchmark-20260725-data generation "$BENCHMARK_ROOT/generation"

uv run python -m experiments.modal_benchmark.local_runner validate \
  --index "$BENCHMARK_ROOT/generation" --expected-count 7561
uv run python -m experiments.modal_benchmark.run_local_suite \
  --index "$BENCHMARK_ROOT/generation" \
  --snapshot-root "$BENCHMARK_ROOT" \
  --queries experiments/modal_benchmark/queries.json \
  --output "$BENCHMARK_ROOT/local-results.json"
uv run --with modal==1.5.3 python -m experiments.modal_benchmark.run_modal_suite \
  --queries experiments/modal_benchmark/queries.json \
  --output "$BENCHMARK_ROOT/modal-results.json"
```

These commands rebuild the generation and rerun the expensive benchmark. They are reproduction instructions, not closeout instructions. The 2026-07-25 closeout used the already-preserved raw results and did not rerun either operation.

The raw JSON includes hashed result identities needed to compare ranking order. Treat it as private even though it contains no corpus bodies. The durable sanitized aggregate is [`results/benchmark.json`](results/benchmark.json), and the interpretation is in [`FINDINGS.md`](FINDINGS.md).

## Configuration

The Modal app uses one T4, at most one container, no warm minimum, no buffer containers, a 15-second scaledown window, and explicit call timeouts. Calls use the authenticated Modal SDK; no public endpoint is created. Search copies the immutable generation from the Volume to container-local temporary storage before FastPLAID loads it.

Local model access is forced offline with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`. The model must already exist in the local Hugging Face cache.

## Cleanup

Export and inspect the durable sanitized result before cleanup. Then remove only the exact benchmark resources:

```bash
uvx modal app stop agentkb-benchmark-20260725 --yes
uvx modal volume delete agentkb-benchmark-20260725-data --yes

uvx modal app list
uvx modal volume list

trash /tmp/agentkb-modal-benchmark-20260725
pgrep -af 'experiments.modal_benchmark|agentkb-modal-benchmark' || true
```

Modal CLI 1.5.3 has no separate `app delete` command; `app stop` permanently stops the deployed app and terminates its containers, but its stopped history row remains visible. Verify that the app has zero tasks and the exact Volume name is absent before moving the temporary local copy to Trash.

## GPU follow-up: T4, L4, and T4 GPU snapshot

The 2026-07-25 remote-only follow-up preserves the original benchmark result and writes its sanitized aggregate to [`results/gpu-followup-20260725.json`](results/gpu-followup-20260725.json). Its interpretation is in [`GPU_FOLLOWUP_FINDINGS.md`](GPU_FOLLOWUP_FINDINGS.md).

The commands below are the exact reproduction sequence. They intentionally omit every local model, search, index-validation, and `run_local_suite` path.

```bash
uv sync
uvx --from modal==1.5.3 modal profile current

BENCHMARK_ROOT=/tmp/agentkb-modal-benchmark-gpu-20260725
uv run python -m experiments.modal_benchmark.snapshot --output "$BENCHMARK_ROOT"

uvx --from modal==1.5.3 modal volume create agentkb-gpu-benchmark-20260725-data
uvx --from modal==1.5.3 modal volume put agentkb-gpu-benchmark-20260725-data \
  "$BENCHMARK_ROOT/corpus.jsonl" corpus/corpus.jsonl
uvx --from modal==1.5.3 modal volume put agentkb-gpu-benchmark-20260725-data \
  "$BENCHMARK_ROOT/batch-1411.jsonl" corpus/batch-1411.jsonl
uvx --from modal==1.5.3 modal volume put agentkb-gpu-benchmark-20260725-data \
  "$BENCHMARK_ROOT/manifest.json" corpus/manifest.json

uvx --from modal==1.5.3 modal deploy \
  -m experiments.modal_benchmark.gpu_followup_modal_app \
  --name agentkb-gpu-benchmark-20260725 --strategy recreate
uv run --with modal==1.5.3 python -m experiments.modal_benchmark.followup_runner \
  --phase build \
  --output "$BENCHMARK_ROOT/generation-build-raw.json"
uv run --with modal==1.5.3 python -m experiments.modal_benchmark.followup_runner \
  --phase search --queries experiments/modal_benchmark/queries.json \
  --output "$BENCHMARK_ROOT/search-raw.json"
uv run --with modal==1.5.3 python -m experiments.modal_benchmark.followup_runner \
  --phase snapshot --snapshot-attempts 6 \
  --queries experiments/modal_benchmark/queries.json \
  --output "$BENCHMARK_ROOT/snapshot-all-raw.json"
uv run --with modal==1.5.3 python -m experiments.modal_benchmark.followup_runner \
  --phase batch \
  --output "$BENCHMARK_ROOT/batch-raw.json"

uvx --from modal==1.5.3 modal app logs \
  agentkb-gpu-benchmark-20260725 --timestamps \
  > "$BENCHMARK_ROOT/modal-logs-final-raw.txt" 2>&1
uv run python -m experiments.modal_benchmark.aggregate_followup \
  --root "$BENCHMARK_ROOT" \
  --output experiments/modal_benchmark/results/gpu-followup-20260725.json

uvx --from modal==1.5.3 modal billing report \
  --for today --resolution h --json \
  > "$BENCHMARK_ROOT/billing-report-raw.json"
```

Snapshot creation is worker-specific. If the first six calls do not yield at least three log-proven restored samples, run at most two additional single-use calls and combine the private timing artifacts before aggregation. Never exceed eight attempts.

The follow-up raw files contain private operational detail and stay beneath the exact temporary root. Do not print or commit them.

### GPU follow-up cleanup

Remove only the exact follow-up resources:

```bash
uvx --from modal==1.5.3 modal app stop \
  agentkb-gpu-benchmark-20260725 --yes
uvx --from modal==1.5.3 modal volume delete \
  agentkb-gpu-benchmark-20260725-data --yes

uvx --from modal==1.5.3 modal app list --json
uvx --from modal==1.5.3 modal volume list --json

trash /tmp/agentkb-modal-benchmark-gpu-20260725
pgrep -af 'agentkb-gpu-benchmark-20260725|experiments.modal_benchmark' || true
```
