# AgentKB private Modal backend

This directory contains the Bun/TypeScript control plane for AgentKB's private,
SDK-only Modal service. Python remains responsible for the existing AgentKB
parsers, model, index, and Modal declarations. The deployed app is `agentkb`;
the private Volume is `agentkb-data`; there are no web endpoints.

## Production commands

Use the stable `agentkb-modal` executable for normal production control-plane
work. From a development checkout, `bun run modal/src/cli.ts ...` remains an
equivalent repository-local fallback.

Deploy the private app with the pinned Modal Python client:

```bash
uv run --with modal==1.5.3 modal deploy -m agentkb.modal_backend.app
```

Export the default `wiki`, `wiki:source`, and `chats` corpus, validate it
locally, stage it, build a new L4 generation, validate the build response, and
remove successful staging data:

```bash
agentkb-modal refresh
```

The refresh reads `wiki_path` and `chats_path` from
`~/.agentkb/config.json`. If absent, the portable fallbacks are
`~/.agentkb/wiki` and `~/.agentkb/chats`; the latter's `readable/` directory is
used for localized chat search paths. Override only the wiki root when needed:

```bash
agentkb-modal refresh --wiki-path /absolute/path/to/wiki
```

Refresh uses the pinned official CLI boundary for Volume writes:
`uvx --from modal==1.5.3 modal volume ...`. It stages `corpus.jsonl` first and
`manifest.json` last. Failed builds preserve `staged/<generation_id>` for
diagnosis; successful builds remove it. Local refresh temporary data is always
removed. Export and local validation stream canonical JSONL instead of loading
the corpus into memory.

Inspect the active generation:

```bash
agentkb-modal status
```

Warm the active T4 search container and wait until it is ready:

```bash
agentkb-modal warm
```

The warm response includes `startup_timing_ms` fields for the immutable
Volume artifact mount, build-certificate check, model load, PLAID index load,
and total initialization. Production search opens `metadata.db` directly from
the mounted Volume with SQLite `mode=ro&immutable=1`. It maps the certified
premerged FastPLAID tensors read-only in place and constructs the Rust index
without copying the generation, copying chunk payloads, or invoking
FastPLAID's mutable merge loader. It does not repeat the index tree hash,
SQLite integrity check, or full count comparison on every new container.

Search and reconstruct result paths against the configured local roots:

```bash
agentkb-modal search --query "retry logic" --k 10
```

Each result's `content` is limited to 600 JavaScript string characters by
default. Truncated content ends with `…` within that limit and reports
`content_truncated: true`; short or absent content reports
`content_truncated: false`. Explicitly request the stored content without this
limit when needed:

```bash
agentkb-modal search --query "retry logic" --k 10 --full-content
```

Generate an ID or trigger a build only for data that was staged separately:

```bash
agentkb-modal generation-id
agentkb-modal build --generation-id <generation_id>
```

Report actual hourly metered usage for the exact Modal app description
`agentkb` over a bounded one-to-seven-day range:

```bash
agentkb-modal cost
agentkb-modal cost --days 7
```

The schema-1 result includes the queried range, exact decimal-string
`metered_cost`, exact totals by resource, and matching raw hourly rows. This is
metered usage, not billed or invoiced cost: Modal reports can lag, and the
reported totals are before credits and reservations.

Preview or remove exactly the generation currently named by
`previous_generation_id`:

```bash
agentkb-modal prune-previous --generation-id <generation_id> --dry-run
agentkb-modal prune-previous --generation-id <generation_id> --force
```

Dry run validates the pointer, target directory, and matching manifest without
remote mutation. A real prune requires `--force`, always refuses the current
generation, atomically clears and commits the previous pointer before deletion,
re-checks that the target is unreferenced, removes only the exact validated
generation directory, then commits and verifies the final state. Races or
malformed pointer state fail closed. A final deletion-commit failure leaves the
already-cleared pointer safe and does not attempt risky compensation.

## SessionStart warm primitive

`modal/src/session-start.ts` reads hook JSON from stdin and submits a detached
`warm_current` call for `startup`, `resume`, and `fork` SessionStart events. It
skips delegated-agent sessions, honors `AGENTKB_SKIP_WARM=1`, emits no normal or
error output, and always exits successfully:

```bash
bun run modal/src/session-start.ts
```

The stable SessionStart hook is installed and live. Accepted top-level session
events can submit `warm_current`; warming may start billable remote GPU compute.
Set `AGENTKB_SKIP_WARM=1` for sessions that must not submit a warm request.

For local development of the control plane without the stable launcher:

```bash
bun run modal/src/cli.ts status
```

## Generation contract

Generation IDs use `g-YYYYMMDDTHHMMSSZ-<12 lowercase hex>`. Volume layout:

```text
staged/<generation_id>/corpus.jsonl
staged/<generation_id>/manifest.json
generations/<generation_id>/index/...
generations/<generation_id>/manifest.json
current.json
```

The staged schema-1 manifest records the generation ID, pinned AgentKB model,
positive corpus count, SHA-256 over the exact canonical JSONL bytes, source file
counts, and export timestamp. A generation is published only after SQLite,
FTS, and both PLAID mapping counts agree and the copied generation validates.
The L4 builder validates staged bytes in a streaming first pass, parses records
lazily in a second pass, encodes exactly 256 documents at a time, and inserts
SQLite/FTS metadata incrementally. Each batch's token embeddings are converted
to FP16 and appended to one scratch payload with an aligned descriptor ledger;
the complete embedding tensor is never materialized in RAM. After encoding,
one memmap backs per-document views. IDs and views are reordered together by
the deterministic `sha256-key-sort-v1` permutation keyed by the corpus hash,
then passed to exactly one PyLate/FastPLAID create call with an explicit
`n_samples_kmeans=16384`. This makes FastPLAID's prefix sample corpus-wide
instead of training centroids on the export-order prefix.

The L4 build requests 32 GiB of container memory. Scratch payload and metadata
are removed on both success and failure before publication. Build responses
and the immutable generation manifest record batch count, FP16 byte count,
embedding dimension, permutation algorithm, one global create, and the K-means
sample size; the control plane validates those invariants. Production search
requires this build block and the full-validation block as its build
certificate; the older disposable baseline manifest without global-create
metrics is not a search-compatible production generation. The model remains
loaded once remotely, the local Mac only exports and streams JSONL, and no
failed build exposes a generation or changes `current.json`.

The validation certificate also records the pinned FastPLAID version, chunk
count, document count, padding, and the dtype, shape, and byte size of every
small or premerged runtime tensor plus metadata and document-length files.
Production fails closed if that certificate is absent or the mounted artifacts
do not match it. Ordinary local AgentKB read-only search keeps using its
cleaned-up disposable FastPLAID workspace.

Remote search responses carry the stored relative path as authority. The
control plane keeps `relative_path` and reconstructs `file` and `path` against
the configured wiki or chats-readable root, so Modal container paths cannot
leak into hook-facing output.

## Local verification

These commands do not contact Modal or load a model:

```bash
cd modal && bun test
cd modal && bun run typecheck
uv run --with pytest pytest tests/test_modal_exporter.py tests/test_modal_runtime.py tests/test_modal_generations.py tests/test_modal_adapter.py
```
