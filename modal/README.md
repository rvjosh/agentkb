# AgentKB private Modal backend

This directory contains the Bun/TypeScript control plane for AgentKB's private
Modal service. The Modal deployment adapter is
`src/agentkb/modal_backend/app.py`; Python is limited to Modal declarations,
Volume filesystem atomicity, and calls into AgentKB's existing model,
`IndexStore`, and search implementation.

The deployed app is named `agentkb` and mounts the existing private Volume
`agentkb-data`. It defines no web endpoint. All calls use the authenticated
Modal SDK.

## Generation contract

Generation IDs are created and validated as
`g-YYYYMMDDTHHMMSSZ-<12 lowercase hex>`. A Volume contains:

```text
staged/<generation_id>/corpus.jsonl
staged/<generation_id>/manifest.json
generations/<generation_id>/index/...
generations/<generation_id>/manifest.json
current.json
```

`staged/<id>/manifest.json` has schema `1` and must include
`generation_id`, `model`, positive `corpus_count`, and lowercase SHA-256
`corpus_hash`. The hash is over the exact `corpus.jsonl` bytes. The production
image currently accepts the bundled AgentKB default model,
`lightonai/GTE-ModernColBERT-v1`.

Generation directories are immutable and never overwritten. A build copies a
validated scratch generation into a hidden sibling, validates the copy, renames
it into place, atomically replaces `current.json`, and then commits the Volume.
The pointer records both current and previous generation IDs.

## Local verification

From the repository root:

```bash
cd modal && bun install
cd modal && bun test
cd modal && bun run typecheck
uv run --with pytest pytest tests/test_modal_generations.py tests/test_modal_adapter.py
```

These checks do not contact Modal or load a model.

## Intended later operations

The following are the intended commands after review. They have **not** been
run as part of this slice.

Create an ID locally:

```bash
bun run modal/src/cli.ts generation-id
```

Later, create the private Volume once and stage an already prepared corpus and
manifest under the same ID:

```bash
uvx modal volume create agentkb-data
uvx modal volume put agentkb-data ./corpus.jsonl staged/<generation_id>/corpus.jsonl
uvx modal volume put agentkb-data ./manifest.json staged/<generation_id>/manifest.json
```

Deploy the private SDK-only app:

```bash
uv run --with modal==1.5.3 modal deploy -m agentkb.modal_backend.app
```

Invoke it through the official TypeScript SDK:

```bash
bun run modal/src/cli.ts status
bun run modal/src/cli.ts build --generation-id <generation_id>
bun run modal/src/cli.ts warm
bun run modal/src/cli.ts search --query "retry logic" --k 10
```

Deployment, persistent resource creation, corpus staging, GPU build/search,
full-corpus indexing, and SessionStart hook wiring are explicitly deferred.
