# AgentKB private Modal backend

This directory contains the Bun/TypeScript control plane for AgentKB's private,
SDK-only Modal service. Python remains responsible for the existing AgentKB
parsers, model, index, and Modal declarations. The deployed app is `agentkb`;
the private Volume is `agentkb-data`; there are no web endpoints.

## Production commands

Run these commands from the repository root.

Deploy the private app with the pinned Modal Python client:

```bash
uv run --with modal==1.5.3 modal deploy -m agentkb.modal_backend.app
```

Export the default `wiki`, `wiki:source`, and `chats` corpus, validate it
locally, stage it, build a new L4 generation, validate the build response, and
remove successful staging data:

```bash
bun run modal/src/cli.ts refresh
```

The refresh reads `wiki_path` and `chats_path` from
`~/.agentkb/config.json`. If absent, the portable fallbacks are
`~/.agentkb/wiki` and `~/.agentkb/chats`; the latter's `readable/` directory is
used for localized chat search paths. Override only the wiki root when needed:

```bash
bun run modal/src/cli.ts refresh --wiki-path /absolute/path/to/wiki
```

Refresh uses the pinned official CLI boundary for Volume writes:
`uvx --from modal==1.5.3 modal volume ...`. It stages `corpus.jsonl` first and
`manifest.json` last. Failed builds preserve `staged/<generation_id>` for
diagnosis; successful builds remove it. Local refresh temporary data is always
removed. Export and local validation stream canonical JSONL instead of loading
the corpus into memory.

Inspect the active generation:

```bash
bun run modal/src/cli.ts status
```

Warm the active T4 search container and wait until it is ready:

```bash
bun run modal/src/cli.ts warm
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
bun run modal/src/cli.ts search --query "retry logic" --k 10
```

Generate an ID or trigger a build only for data that was staged separately:

```bash
bun run modal/src/cli.ts generation-id
bun run modal/src/cli.ts build --generation-id <generation_id>
```

## SessionStart warm primitive

`modal/src/session-start.ts` reads hook JSON from stdin and submits a detached
`warm_current` call for `startup`, `resume`, and `fork` SessionStart events. It
skips delegated-agent sessions, honors `AGENTKB_SKIP_WARM=1`, emits no normal or
error output, and always exits successfully:

```bash
bun run modal/src/session-start.ts
```

The top-level session will deploy the live app and install the actual hook
configuration later. This implementation does not deploy, mutate the Volume,
start a GPU, or change global hooks.

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
