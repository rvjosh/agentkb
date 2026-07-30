# agentkb

> A personal, local-first memory system for coding agents.

Background, motivation, and design decisions: https://isaacflath.com/writing/agentkb

Agents are amnesiacs. agentkb is my attempt at giving *mine* a memory I can live with. Plain markdown I can read and edit, a  search layer that's reliable enough to trust, and separate stores for different kinds of knowledge so each one can fetch, chunk, and retrieve on its own terms.

The power is in the intersection of the stores:

- **Wiki** captures distilled, durable knowledge.
- **Chats** capture what was tried, what failed, what was learned.
- **Communications** capture dense, link-rich threads (currently X).
- **Skills** capture repeatable procedures on disk.

All source data lives in normal files (markdown, JSONL, plain directories). Indexes are ephemeral caches rebuilt from source data. Backed up to git, one repo per store.


## Is this for you?

Probably not as-is. This is a personal tool, shaped to my workflow and the specific agents I use (Claude Code and Pi).  There no extension mechanism. The code is here so you can:

- **Point your agent at the repo**, take the parts that make sense, and build your own.
- **Fork it** if it's close to what you want and extend it with stores you care about.
- Read it to steal ideas for how to structure a memory layer of your own.

The bar for "just install it and use it" is not what I'm optimizing for. If that's what you want, the README still tells you everything you need, but don't expect smooth onboarding or long-term stability.

## Architecture

Every store follows the same three-layer shape:

- **Source data** is owned by agentkb (copied in, not referenced in place) so sync has a stable snapshot.
- **Readable markdown** is what you and your agent read and what gets indexed.
- **Indexes** are rebuilt incrementally when readable files change.

### On disk

```
~/.agentkb/
  config.json              # settings
  traceability.db          # every search + intermediate rankings, for evals
  wiki/
    wiki/                  # pages you and the agent write
    sources/               # raw ingested documents
    schema.md              # writing conventions
    index.md               # page catalog
    .index/                # FTS5 + PLAID, git-ignored
  chats/
    sessions/              # JSONL copied from Claude Code / Pi
      claude/
      pi/
    readable/              # rendered markdown, one file per session
      YYYY-MM/
    .index/
  communications/
    raw/
      x/                   # tweets + handle manifest
    readable/              # thread-per-file markdown
    .index/
  wiki/sources/refs/       # refs: mirrored external prose, managed by manifest.json
  references/_cache/       # git-clone cache for git refs (bare-ish, reused across pulls)
  skills/
    .agents/skills/        # loaded by agents via --add-dir
```

## The stores

### Wiki

Plain markdown files you and your agents write. General-purpose: gotchas, taste, people, tools, domain knowledge, mental models. Complements skills. Skills are procedures, the wiki is knowledge.

```
~/.agentkb/wiki/
  wiki/            # pages
  sources/         # raw ingested documents
  schema.md        # writing conventions (read this before bulk edits)
  index.md         # catalog
```

### Chats

Coding-agent conversations, exported to readable markdown and indexed. Built-in sources: **Claude Code** (`~/.claude/projects/`), **Codex** (`~/.codex/sessions/`), and **Pi** (`~/.pi/agent/sessions/`).

Three-stage pipeline:

```
source JSONL
  -> ~/.agentkb/chats/sessions/{source}/...         # agentkb-owned copy
  -> ~/.agentkb/chats/readable/YYYY-MM/*.md         # renderable + searchable
  -> ~/.agentkb/chats/.index/                       # FTS5 + PLAID
```

The readable markdown keeps user messages, assistant text and thinking blocks, tool calls formatted for reading, tool results capped, and frontmatter with `source`, `session_id`, `project`, `date`, `messages`.

Central-history exports require the private schema-1 `current.json` generation
pointer and validate its selected content-addressed database and provenance
catalog, including compressed/catalog hashes and the exact decompressed SQLite
hash. There is no fixed-name fallback. Agent-history schema v1 snapshots remain
readable; for schemas v2, v3, and v4, exports publish only sessions in the canonical
`publication_eligible_sessions` view, so lifecycle-excluded sessions contribute
zero records to future builds. Schema v3 also supports archives where physically
erased transcript, version, and event rows are gone while lifecycle tombstones
remain independently retained. Unknown and newer schemas fail closed.

The private Modal control plane can inventory all bounded immutable and staged
generations, verify an exact central-history session file identity in each
corpus, and dry-run deletion of one exact non-current generation. Forced
deletion requires an expected-current guard, actor, and reason; it writes a
content-free receipt and never deletes the active generation. This is generation
cleanup, not a provider or physical-media guarantee. It does not build or publish
a clean current generation, remove native logs or backups, or prove erasure from
provider infrastructure; the higher-level orchestrator must publish a clean
current generation before removing older copies.

### Communications

Dense, link-rich communications. Today that's X.  A curated handle list fetched via the X API, filtered to originals + self-reply threads + quote-tweets (skips retweets and replies to other users), rendered one thread per markdown file.

Requires `X_BEARER_TOKEN` in the environment (app-only bearer, from your X developer portal).

### Refs

Automated mirrors of external prose — github-hosted docs, papers, blog posts — fetched locally and indexed into the wiki as `wiki:source` chunks. A manifest under `~/.agentkb/wiki/sources/refs/manifest.json` tracks what's mirrored; `agentkb index` pulls the latest and reindexes.

```bash
agentkb refs add https://github.com/lightonai/pylate --subpath docs
agentkb refs add https://alexzhang13.github.io/blog/2025/rlm/ --id rlm-blog
agentkb refs add https://arxiv.org/html/2512.24601v2       --id attention
agentkb refs list
agentkb refs remove <id>
```

For git refs, always pass `--subpath` pointing at the docs directory — without it the whole repo is cloned and stray READMEs/CHANGELOGs clutter search. URL refs convert HTML→markdown on fetch (via `markdownify`).

### Skills

Agent skill directories (`SKILL.md` + scripts + references) synced via git. **Not indexed, not searched.** Agents load them directly via `--add-dir`.

```
~/.agentkb/skills/
  .agents/skills/
    content-blog/
      SKILL.md
      scripts/
      references/
    ...
```

```bash
ls ~/.agentkb/skills
codex --add-dir ~/.agentkb/skills
```

## Search

```bash
agentkb search "retry logic with backoff"                    # default scope: wiki
agentkb "retry logic with backoff"                           # shorthand (default-search group)

agentkb search -s chats "how did I fix the auth bug"
agentkb search -s wiki:notes "vector search"                 # hand-authored notes only
agentkb search -s wiki:source "vector search"                # ingested sources + refs only
agentkb search -s all "authentication flow"                  # wiki + chats (NOT communications)
agentkb search -s communications "colbert plaid"             # explicit opt-in

agentkb search -e "async def" "error handling"               # regex pre-filter
agentkb search -F "TODO:" "incomplete work"                  # fixed string
agentkb search -w -e "test" "testing patterns"               # word boundary
agentkb search --include="*.md" "writing style"
agentkb search --exclude-dir=archive "config"

agentkb search -k 10 "retry"                                 # top-k (default: 3)
agentkb search -c "main entry point"                         # full content (including JSON)
agentkb search -l "authentication"                           # files only
agentkb search --json "error handling"                       # compact metadata for scripts/agents
agentkb search --semantic-only "retry"                       # skip keyword search
agentkb search --no-refresh "retry"                          # existing index only; no refresh or trace write
```

Results are tagged with their source: `[wiki]`, `[wiki:source]`, `[chats]`, `[communications]`.
JSON search results use absolute `file`/`path` values, include `filename`, title,
section, tags, and score metadata, and omit chunk content unless `-c` is passed.
`--no-refresh` opens metadata and ID mappings as immutable SQLite inputs and
runs FastPLAID from a cleaned-up temporary copy, so source index and traceability
artifacts are not written. Do not rebuild an index concurrently with a
`--no-refresh` search; immutable reads intentionally assume a stable index snapshot.

Every search is recorded in `~/.agentkb/traceability.db` — the original query, semantic-expanded query, pattern, per-stage rankings (semantic / keyword / RRF), and final results. Useful for evals and for debugging why a result did or didn't surface.

### Cached-only local indexing

Before offline maintenance, check that the ColBERT snapshot is already present:

```bash
agentkb model-cache
agentkb model-cache --json
agentkb model-cache --model lightonai/GTE-ModernColBERT-v1
```

`agentkb index --cached-only` implies `--no-fetch` and verifies the requested
`--model` (or the default model) before source discovery, rendering, index
creation/clearing, or encoding. It never downloads a model. If the cache is
missing, populate it while online with:

```bash
uv run hf download lightonai/GTE-ModernColBERT-v1
```

Then retry cached-only indexing, or use the remote AgentKB path. The diagnostic
JSON contains only `model`, `cached`, and `recovery`; it never exposes cache
paths or source content.

## Consolidation

Turning chat and communications activity into durable wiki pages. Consolidation is an **instruction generator**: agentkb exports the relevant sessions/threads, prints the paths, and prints a prompt the agent acts on. The agent does the synthesis.

```bash
agentkb consolidate chats
agentkb consolidate communications
```

## Sync

One git repo per store. The source data syncs.

```bash
agentkb settings set wiki_remote           "git@github.com:you/wiki.git"
agentkb settings set chats_remote          "git@github.com:you/chats.git"
agentkb settings set communications_remote "git@github.com:you/communications.git"
agentkb settings set skills_remote         "git@github.com:you/skills.git"

agentkb sync pull                          # clone missing, pull the rest
agentkb sync push                          # stage + commit + push per store
agentkb sync status
```

The traceability DB syncs separately to S3 (so it can stay private and grow without bloating git):

```bash
agentkb settings set traceability_s3_bucket "my-bucket"
agentkb settings set traceability_s3_key    "agentkb/traceability.db"
# `agentkb sync push/pull` handles S3 upload/download when configured
```
