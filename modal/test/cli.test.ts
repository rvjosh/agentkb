import { expect, test } from "bun:test";

import type { AgentKbClient } from "../src/client";
import {
  DEFAULT_CONTENT_LIMIT,
  DEFAULT_MIN_MEANING_SCORE,
  dedupByFile,
  defaultCliDependencies,
  runCli,
  runMain,
  shapeSearchResult,
} from "../src/cli";
import type {
  BuildResult,
  DocumentSearchResult,
  PrunePreviousResult,
  SearchHit,
  SearchResult,
  StatusResponse,
  WarmResult,
} from "../src/protocol";

const ID = "g-20260725T123456Z-001122aabbcc";
const CURRENT_ID = "g-20260725T133456Z-112233aabbcc";

function immutableCertificate(documentCount: number) {
  const tensor = { size_bytes: 1, dtype: "float16", shape: [1] };
  return {
    schema: 1 as const,
    fast_plaid_version: "1.3.0.290",
    num_chunks: 1,
    nbits: 4,
    document_count: documentCount,
    padding_rows: 0,
    artifacts: {
      "metadata.json": { size_bytes: 1 },
      "doclens.0.json": { size_bytes: 1 },
      "centroids.npy": { ...tensor, shape: [2, 128] },
      "avg_residual.npy": tensor,
      "bucket_cutoffs.npy": tensor,
      "bucket_weights.npy": tensor,
      "ivf.npy": { ...tensor, dtype: "int64" },
      "ivf_lengths.npy": { ...tensor, dtype: "int32" },
      "merged_codes.npy": { ...tensor, dtype: "int64" },
      "merged_residuals.npy": { ...tensor, dtype: "uint8", shape: [1, 16] },
    },
  };
}

class FakeClient implements AgentKbClient {
  calls: unknown[][] = [];
  closed = false;

  async status(): Promise<StatusResponse> {
    this.calls.push(["status"]);
    return {
      schema: 1,
      current_generation_id: null,
      previous_generation_id: null,
      published_at: null,
      current_manifest: null,
      previous_manifest: null,
    };
  }

  async generations(): Promise<any> {
    this.calls.push(["generations"]);
    return {
      schema: 1,
      current_generation_id: ID,
      previous_generation_id: null,
      items: [{ generation_id: ID, type: "generation", classification: "current" }],
      counts: { current: 1, previous: 0, orphan: 0, staged: 0 },
    };
  }

  async findSession(source: "claude" | "codex", sessionId: string): Promise<any> {
    this.calls.push(["findSession", source, sessionId]);
    return {
      schema: 1,
      source,
      session_id: sessionId,
      canonical_file: `agent-history-central/${source}/${sessionId}.md`,
      results: [],
      total_exact_match_count: 0,
      verification_failures: [],
      verified: true,
    };
  }

  async deleteGeneration(
    generationId: string,
    targetType: "generation" | "staged",
    expectedCurrent: string,
    force: boolean,
    actor: string,
    reason: string,
    exactSessionKey?: string,
  ): Promise<any> {
    this.calls.push([
      "deleteGeneration",
      generationId,
      targetType,
      expectedCurrent,
      force,
      actor,
      reason,
      exactSessionKey,
    ]);
    return {
      schema: 1,
      dry_run: !force,
      deleted: force,
      idempotent: false,
      target_id: generationId,
      target_type: targetType,
      classification: targetType === "staged" ? "staged" : "orphan",
      current_generation_id: expectedCurrent,
      operation_id: force ? "op" : null,
      receipt: null,
    };
  }

  async warm(): Promise<WarmResult> {
    this.calls.push(["warm"]);
    return {
      schema: 1,
      generation_id: ID,
      model: "model",
      corpus_count: 1,
      startup_timing_ms: {
        artifact_mount: 1,
        certificate: 1,
        model: 1,
        index_load: 1,
        total: 4,
      },
      ready: true,
    };
  }

  async warmDetached(): Promise<void> {
    this.calls.push(["warmDetached"]);
  }

  async search(
    query: string,
    k: number,
    transcriptSessions?: boolean,
  ): Promise<SearchResult> {
    this.calls.push(
      transcriptSessions
        ? ["search", query, k, true]
        : ["search", query, k],
    );
    return {
      schema: 1,
      generation_id: ID,
      query,
      k,
      results: [],
    };
  }

  async searchDocuments(
    query: string,
    k: number,
    includeChats: boolean,
  ): Promise<DocumentSearchResult> {
    this.calls.push(["searchDocuments", query, k, includeChats]);
    return {
      schema: 1,
      generation_id: ID,
      query,
      k,
      collections: includeChats
        ? ["wiki", "wiki:source", "chats"]
        : ["wiki", "wiki:source"],
      words_terms: [],
      by_words: [],
      by_meaning: [],
    };
  }

  async build(generationId: string): Promise<BuildResult> {
    this.calls.push(["build", generationId]);
    return {
      schema: 1,
      generation_id: generationId,
      previous_generation_id: null,
      model: "model",
      corpus_count: 1,
      corpus_hash: "a".repeat(64),
      document_batch_size: 256,
      document_batch_count: 1,
      embedding_dimension: 128,
      staged_embedding_bytes: 256,
      plaid_create_count: 1,
      plaid_kmeans_sample_size: 16_384,
      plaid_permutation_algorithm: "sha256-key-sort-v1",
      validation: {
        sqlite_count: 1,
        fts_count: 1,
        plaid_mapping_count: 1,
        plaid_reverse_mapping_count: 1,
        index_tree_hash: "b".repeat(64),
        immutable_premerged: immutableCertificate(1),
      },
      duration_ms: 1,
    };
  }

  async prunePrevious(
    generationId: string,
    dryRun: boolean,
  ): Promise<PrunePreviousResult> {
    this.calls.push(["prunePrevious", generationId, dryRun]);
    return {
      schema: 1,
      dry_run: dryRun,
      deleted: !dryRun,
      target_generation_id: generationId,
      current_generation_id: "g-20260725T130000Z-ddeeff001122",
      previous_generation_id: generationId,
      final_previous_generation_id: dryRun ? generationId : null,
    };
  }

  close(): void {
    this.closed = true;
  }
}

test("default search is documents-only, two lists, hint last and runnable", async () => {
  const client = new FakeClient();
  const output: string[] = [];
  const query = "what's Josh's retry plan";
  await runCli(["search", "--query", query], () => client, (line) => output.push(line));
  // Per-file de-dup over-fetches 3x k, then trims back to k.
  expect(client.calls).toEqual([["searchDocuments", query, 30, false]]);
  expect(client.closed).toBeTrue();
  const parsed = JSON.parse(output[0]!);
  expect(Object.keys(parsed)).toEqual([
    "schema",
    "generation_id",
    "query",
    "k",
    "searched",
    "words_terms",
    "by_words",
    "by_meaning",
    "match",
    "hint",
  ]);
  expect(parsed.searched).toEqual(["wiki", "wiki:source"]);
  const quoted = `'what'\\''s Josh'\\''s retry plan'`;
  expect(parsed.hint).toContain(`agentkb search --query ${quoted} --include-chats`);
  expect(parsed.hint).toContain(`agent-history search ${quoted}`);
  // The quoted form must round-trip through a real shell to the exact query.
  const echoed = Bun.spawnSync(["sh", "-c", `printf %s ${quoted}`]).stdout.toString();
  expect(echoed).toBe(query);

  const chats = new FakeClient();
  const chatOutput: string[] = [];
  await runCli(
    ["search", "--query", "x", "--k", "3", "--include-chats"],
    () => chats,
    (line) => chatOutput.push(line),
  );
  expect(chats.calls).toEqual([["searchDocuments", "x", 9, true]]);
  const chatParsed = JSON.parse(chatOutput[0]!);
  expect(chatParsed.searched).toEqual(["wiki", "wiki:source", "chats"]);
  expect(chatParsed.hint).toContain("nightly index built 2026-07-25 12:34 UTC");
  expect(Object.keys(chatParsed).at(-1)).toBe("hint");
});

test("routes transcript session search only when explicitly enabled", async () => {
  const client = new FakeClient();
  await runCli(
    [
      "search",
      "--query",
      "private knowledge",
      "--k",
      "4",
      "--transcript-sessions",
    ],
    () => client,
    () => {},
  );
  expect(client.calls).toEqual([
    ["search", "private knowledge", 12, true],
  ]);
});

test("bounds search content and preserves every other localized result field", () => {
  const longContent = "x".repeat(DEFAULT_CONTENT_LIMIT + 25);
  const shaped = shapeSearchResult({
    schema: 1,
    generation_id: ID,
    query: "bounded",
    k: 3,
    results: [
      {
        collection: "wiki",
        file: "/local/wiki/long.md",
        path: "/local/wiki/long.md",
        filename: "long.md",
        relative_path: "long.md",
        line: 4,
        score: 0.9,
        title: "Long result",
        tags: ["kept"],
        content: longContent,
      },
      {
        collection: "chats",
        file: "/local/chats/short.md",
        path: "/local/chats/short.md",
        filename: "short.md",
        relative_path: "short.md",
        line: 5,
        score: 0.8,
        content: "short content",
      },
      {
        collection: "wiki:source",
        file: "/local/wiki/absent.md",
        path: "/local/wiki/absent.md",
        filename: "absent.md",
        relative_path: "absent.md",
        line: 6,
        score: 0.7,
      },
    ],
  }, false);

  expect(shaped.results[0]!.content).toHaveLength(DEFAULT_CONTENT_LIMIT);
  expect(shaped.results[0]!.content).toBe(
    `${"x".repeat(DEFAULT_CONTENT_LIMIT - 1)}…`,
  );
  expect(shaped.results[0]!.content_truncated).toBeTrue();
  expect(shaped.results[0]).toMatchObject({
    file: "/local/wiki/long.md",
    path: "/local/wiki/long.md",
    filename: "long.md",
    relative_path: "long.md",
    title: "Long result",
    tags: ["kept"],
  });
  expect(shaped.results[1]!.content).toBe("short content");
  expect(shaped.results[1]!.content_truncated).toBeFalse();
  expect(shaped.results[2]).not.toHaveProperty("content");
  expect(shaped.results[2]!.content_truncated).toBeFalse();
});

test("--full-content preserves content and makes exactly one search call", async () => {
  const client = new FakeClient();
  const content = "z".repeat(DEFAULT_CONTENT_LIMIT + 1);
  client.searchDocuments = async (
    query: string,
    k: number,
    includeChats: boolean,
  ): Promise<DocumentSearchResult> => {
    client.calls.push(["searchDocuments", query, k, includeChats]);
    const hit = {
        collection: "wiki",
        file: "/local/wiki/result.md",
        path: "/local/wiki/result.md",
        filename: "result.md",
        relative_path: "result.md",
        line: 1,
        score: 1,
        content,
    };
    return {
      schema: 1,
      generation_id: ID,
      query,
      k,
      collections: ["wiki", "wiki:source"],
      words_terms: ["wide"],
      by_words: [hit],
      by_meaning: [hit],
    };
  };
  const output: string[] = [];

  await runCli(
    ["search", "--query", "wide", "--k", "1", "--full-content"],
    () => client,
    (line) => output.push(line),
  );

  expect(client.calls).toEqual([["searchDocuments", "wide", 3, false]]);
  expect(JSON.parse(output[0]!)).toMatchObject({
    by_words: [{ content, content_truncated: false }],
    by_meaning: [{ content, content_truncated: false }],
  });
});

test("--metadata-only preserves the envelope, metadata, and backend order", async () => {
  const result: SearchResult = {
    schema: 1,
    generation_id: ID,
    query: "identities",
    k: 2,
    results: [
      {
        collection: "chats",
        file: "/local/chats/second.md",
        path: "/local/chats/second.md",
        filename: "second.md",
        relative_path: "second.md",
        line: 20,
        score: 0.2,
        title: "Second",
        tags: ["stable"],
        content: "omitted payload",
      },
      {
        collection: "wiki",
        file: "/local/wiki/first.md",
        path: "/local/wiki/first.md",
        filename: "first.md",
        relative_path: "first.md",
        line: 10,
        score: 0.9,
        content: "also omitted",
      },
    ],
  };
  const client = new FakeClient();
  client.search = async (query: string, k: number): Promise<SearchResult> => {
    client.calls.push(["search", query, k]);
    return result;
  };
  const output: string[] = [];

  await runCli(
    [
      "search",
      "--query",
      "identities",
      "--k",
      "2",
      "--metadata-only",
      "--transcript-sessions",
    ],
    () => client,
    (line) => output.push(line),
  );

  expect(client.calls).toEqual([["search", "identities", 6]]);
  const parsed = JSON.parse(output[0]!);
  expect(Object.keys(parsed).at(-1)).toBe("hint");
  expect(parsed.hint).toBe(
    "Searched chats only (nightly index built 2026-07-25 12:34 UTC). Documents: agentkb search --query 'identities' · today's chats by exact words: agent-history search 'identities'",
  );
  delete parsed.hint;
  expect(parsed.match.no_good_match).toBeTrue();
  delete parsed.match;
  expect(parsed).toEqual({
    schema: 1,
    generation_id: ID,
    query: "identities",
    k: 2,
    results: [
      {
        collection: "chats",
        file: "/local/chats/second.md",
        path: "/local/chats/second.md",
        filename: "second.md",
        relative_path: "second.md",
        line: 20,
        score: 0.2,
        title: "Second",
        tags: ["stable"],
      },
      {
        collection: "wiki",
        file: "/local/wiki/first.md",
        path: "/local/wiki/first.md",
        filename: "first.md",
        relative_path: "first.md",
        line: 10,
        score: 0.9,
      },
    ],
  });
  expect(
    parsed.results.map((hit: { relative_path: string }) => hit.relative_path),
  ).toEqual(["second.md", "first.md"]);
  for (const hit of parsed.results) {
    expect(hit).not.toHaveProperty("content");
    expect(hit).not.toHaveProperty("content_truncated");
  }
});

test("search content flags conflict with usage status and flags stay strict", async () => {
  for (const args of [
    [
      "search",
      "--query",
      "x",
      "--metadata-only",
      "--full-content",
    ],
    ["search", "--query", "x", "--metadata-only", "--metadata-only"],
    ["search", "--query", "x", "--unknown"],
    ["search", "--query", "x", "--include-chats", "--transcript-sessions"],
  ]) {
    const client = new FakeClient();
    const errors: string[] = [];
    expect(
      await runMain({
        args,
        stdout: () => {},
        stderr: (line) => errors.push(line),
        clientFactory: () => client,
        dependencies: defaultCliDependencies,
      }),
    ).toBe(2);
    expect(client.calls).toEqual([]);
    expect(errors.join("")).toContain("agentkb --help");
  }
});

test("routes an already-staged build", async () => {
  const client = new FakeClient();
  await runCli(["build", "--generation-id", ID], () => client, () => {});
  expect(client.calls).toEqual([["build", ID]]);
  expect(client.closed).toBeTrue();
});

test("rejects invalid command input before a remote call", async () => {
  const client = new FakeClient();
  expect(
    runCli(["search", "--query", "x", "--k", "0"], () => client, () => {}),
  ).rejects.toThrow(/between 1 and 100/);
  expect(client.calls).toEqual([]);
  expect(client.closed).toBeFalse();
});

test("routes prune dry runs without force and real prunes only with force", async () => {
  const dryClient = new FakeClient();
  await runCli(
    ["prune-previous", "--generation-id", ID, "--dry-run"],
    () => dryClient,
    () => {},
  );
  expect(dryClient.calls).toEqual([["prunePrevious", ID, true]]);

  const realClient = new FakeClient();
  await runCli(
    ["prune-previous", "--force", "--generation-id", ID],
    () => realClient,
    () => {},
  );
  expect(realClient.calls).toEqual([["prunePrevious", ID, false]]);
});

test("prune rejects missing force and invalid IDs before a remote call", async () => {
  for (const args of [
    ["prune-previous", "--generation-id", ID],
    ["prune-previous", "--generation-id", "../escape", "--force"],
  ]) {
    const client = new FakeClient();
    expect(runCli(args, () => client, () => {})).rejects.toThrow();
    expect(client.calls).toEqual([]);
    expect(client.closed).toBeFalse();
  }
});

test("routes bounded generation inventory and exact-session verification", async () => {
  const client = new FakeClient();
  await runCli(["generations", "--json"], () => client, () => {});
  await runCli(
    [
      "find-session",
      "--source",
      "codex",
      "--session-id",
      "session-1",
      "--json",
    ],
    () => client,
    () => {},
  );
  expect(client.calls).toEqual([
    ["generations"],
    ["findSession", "codex", "session-1"],
  ]);
});

test("generation deletion is dry-run first and force is explicit", async () => {
  const dryClient = new FakeClient();
  await runCli(
    [
      "delete-generation",
      "--generation-id",
      ID,
      "--expected-current",
      CURRENT_ID,
      "--actor",
      "test",
      "--reason",
      "privacy",
      "--json",
    ],
    () => dryClient,
    () => {},
  );
  expect(dryClient.calls).toEqual([[
    "deleteGeneration",
    ID,
    "generation",
    CURRENT_ID,
    false,
    "test",
    "privacy",
    undefined,
  ]]);

  const forceClient = new FakeClient();
  await runCli(
    [
      "delete-staged",
      "--generation-id",
      ID,
      "--expected-current",
      CURRENT_ID,
      "--actor",
      "test",
      "--reason",
      "privacy",
      "--exact-session-key",
      "codex/session-1",
      "--force",
    ],
    () => forceClient,
    () => {},
  );
  expect(forceClient.calls[0]).toEqual([
    "deleteGeneration",
    ID,
    "staged",
    CURRENT_ID,
    true,
    "test",
    "privacy",
    "codex/session-1",
  ]);
});

test("generation erasure rejects incomplete or unsafe CLI input locally", async () => {
  for (const args of [
    ["find-session", "--source", "unsupported", "--session-id", "session-1"],
    [
      "delete-generation",
      "--generation-id",
      ID,
      "--expected-current",
      "../escape",
      "--actor",
      "test",
      "--reason",
      "privacy",
    ],
    [
      "delete-generation",
      "--generation-id",
      ID,
      "--expected-current",
      CURRENT_ID,
      "--reason",
      "privacy",
    ],
  ]) {
    const client = new FakeClient();
    expect(runCli(args, () => client, () => {})).rejects.toThrow();
    expect(client.calls).toEqual([]);
  }
});

test("cost routes the exact pinned command and filters to AgentKB", async () => {
  const calls: string[][] = [];
  const output: string[] = [];
  await runCli(
    ["cost", "--days", "2"],
    () => new FakeClient(),
    (line) => output.push(line),
    {
      ...defaultCliDependencies,
      billingSpawn: (command) => {
        calls.push(command);
        return {
          exitCode: 0,
          stderr: "",
          stdout: JSON.stringify([
            {
              object_id: "ap-1",
              description: "agentkb",
              environment: "main",
              interval_start: "2026-07-25T12:00:00+00:00",
              resource: "T4",
              cost: "0.125",
            },
            {
              object_id: "ap-2",
              description: "other",
              environment: "main",
              interval_start: "2026-07-25T12:00:00+00:00",
              resource: "CPU",
              cost: "99",
            },
          ]),
        };
      },
    },
  );
  expect(calls).toEqual([[
    "uvx",
    "--from",
    "modal==1.5.3",
    "modal",
    "billing",
    "report",
    "--start",
    "2 days ago",
    "--resolution",
    "h",
    "--show-resources",
    "--json",
  ]]);
  expect(JSON.parse(output[0]!)).toMatchObject({
    days: 2,
    app_name: "agentkb",
    metered_cost: "0.125",
  });
});

test("invalid cost input makes no subprocess call", async () => {
  let spawned = false;
  expect(
    runCli(
      ["cost", "--days", "8"],
      () => new FakeClient(),
      () => {},
      {
        ...defaultCliDependencies,
        billingSpawn: () => {
          spawned = true;
          throw new Error("should not spawn");
        },
      },
    ),
  ).rejects.toThrow(/between 1 and 7/);
  expect(spawned).toBeFalse();
});

test("help wins anywhere and executable boundary classifies failures", async () => {
  const output: string[] = [];
  await runCli(
    ["prune-previous", "--generation-id", "../escape", "--help"],
    () => new FakeClient(),
    (line) => output.push(line),
  );
  expect(output.join("")).toContain("agentkb-modal");
  expect(output.join("")).toContain("--full-content");

  const errors: string[] = [];
  expect(
    await runMain({
      args: ["cost", "--days", "0"],
      stdout: () => {},
      stderr: (line) => errors.push(line),
      clientFactory: () => new FakeClient(),
      dependencies: defaultCliDependencies,
    }),
  ).toBe(2);
  expect(errors.join("")).toContain("agentkb --help");

  expect(
    await runMain({
      args: ["status"],
      stdout: () => {},
      stderr: () => {},
      clientFactory: () => {
        throw new Error("runtime unavailable");
      },
      dependencies: defaultCliDependencies,
    }),
  ).toBe(1);
});

function hit(relativePath: string, score: number, line = 1): SearchHit {
  return {
    collection: "wiki",
    file: relativePath,
    path: relativePath,
    filename: relativePath,
    line,
    score,
    relative_path: relativePath,
  };
}

function documentClient(byWords: SearchHit[], byMeaning: SearchHit[]): FakeClient {
  const client = new FakeClient();
  client.searchDocuments = async (query, k, includeChats) => {
    client.calls.push(["searchDocuments", query, k, includeChats]);
    return {
      schema: 1,
      generation_id: ID,
      query,
      k,
      collections: ["wiki", "wiki:source"],
      words_terms: [],
      by_words: byWords,
      by_meaning: byMeaning,
    };
  };
  return client;
}

async function searchJson(client: FakeClient, extra: string[]) {
  const output: string[] = [];
  await runCli(
    ["search", "--query", "q", ...extra],
    () => client,
    (line) => output.push(line),
  );
  return JSON.parse(output[0]!);
}

test("dedupByFile keeps the best-ranked hit per file, counts the rest, trims to k", () => {
  const kept = dedupByFile(
    [hit("a.md", 30, 1), hit("a.md", 29, 9), hit("b.md", 28), hit("a.md", 27, 20), hit("c.md", 26)],
    2,
  );
  expect(kept.map((h) => [h.relative_path, h.line, h.more_in_file])).toEqual([
    ["a.md", 1, 2],
    ["b.md", 1, undefined],
  ]);
});

test("search collapses repeated files in both lists by default; --no-dedup keeps them", async () => {
  const words = [hit("a.md", 9, 1), hit("a.md", 8, 5), hit("a.md", 7, 9), hit("b.md", 6)];
  const meaning = [hit("c.md", 25, 1), hit("c.md", 24, 3), hit("d.md", 23)];

  const deduped = await searchJson(documentClient(words, meaning), ["--k", "2"]);
  expect(deduped.k).toBe(2);
  expect(deduped.by_words.map((h: SearchHit) => h.relative_path)).toEqual(["a.md", "b.md"]);
  expect(deduped.by_words[0].more_in_file).toBe(2);
  expect(deduped.by_meaning.map((h: SearchHit) => h.relative_path)).toEqual(["c.md", "d.md"]);

  const client = documentClient(words, meaning);
  const raw = await searchJson(client, ["--k", "2", "--no-dedup"]);
  expect(client.calls).toEqual([["searchDocuments", "q", 2, false]]);
  expect(raw.by_words.map((h: SearchHit) => h.relative_path)).toEqual(["a.md", "a.md"]);
  expect(raw.by_words[0]).not.toHaveProperty("more_in_file");
});

test("over-fetch for de-dup is capped at the server's k limit", async () => {
  const client = documentClient([], []);
  await searchJson(client, ["--k", "50"]);
  expect(client.calls).toEqual([["searchDocuments", "q", 100, false]]);
});

test("match.no_good_match flags a weak or empty by-meaning list", async () => {
  const strong = await searchJson(
    documentClient([], [hit("a.md", DEFAULT_MIN_MEANING_SCORE + 1)]),
    [],
  );
  expect(strong.match).toEqual({
    no_good_match: false,
    top_meaning_score: DEFAULT_MIN_MEANING_SCORE + 1,
    min_score: DEFAULT_MIN_MEANING_SCORE,
  });

  const weak = await searchJson(
    documentClient([hit("a.md", 50)], [hit("b.md", DEFAULT_MIN_MEANING_SCORE - 1)]),
    [],
  );
  // Words hits do not count: the any-term fill can match noise.
  expect(weak.match.no_good_match).toBeTrue();

  const empty = await searchJson(documentClient([], []), []);
  expect(empty.match).toEqual({
    no_good_match: true,
    top_meaning_score: null,
    min_score: DEFAULT_MIN_MEANING_SCORE,
  });

  const overridden = await searchJson(
    documentClient([], [hit("b.md", 5)]),
    ["--min-score", "4.5"],
  );
  expect(overridden.match).toEqual({
    no_good_match: false,
    top_meaning_score: 5,
    min_score: 4.5,
  });
  expect(Object.keys(overridden).at(-1)).toBe("hint");
});

test("unsupported search filters fail clearly before any remote call", async () => {
  for (const [flag, value] of [
    ["--author", "josh"],
    ["--since", "2026-01-01"],
    ["--until", "2026-02-01"],
    ["--tag", "retry"],
    ["--type", "section"],
    ["--collection", "wiki"],
  ]) {
    const client = new FakeClient();
    const errors: string[] = [];
    expect(
      await runMain({
        args: ["search", "--query", "x", flag!, value!],
        stdout: () => {},
        stderr: (line) => errors.push(line),
        clientFactory: () => client,
        dependencies: defaultCliDependencies,
      }),
    ).toBe(2);
    expect(client.calls).toEqual([]);
    expect(errors.join("")).toContain(
      `search filter not supported by the AgentKB server yet: ${flag}`,
    );
  }
});

test("search filter and score flags are validated before the support check", async () => {
  for (const [args, message] of [
    [["--since", "2026-13-01"], "--since must be a date as YYYY-MM-DD"],
    [["--until", "yesterday"], "--until must be a date as YYYY-MM-DD"],
    [["--since", "2026-03-01", "--until", "2026-02-01"], "--since must not be after --until"],
    [["--collection", "notes"], "--collection must be one of wiki, wiki:source, chats"],
    [["--min-score", "high"], "--min-score must be a number"],
  ] as const) {
    const client = new FakeClient();
    const errors: string[] = [];
    expect(
      await runMain({
        args: ["search", "--query", "x", ...args],
        stdout: () => {},
        stderr: (line) => errors.push(line),
        clientFactory: () => client,
        dependencies: defaultCliDependencies,
      }),
    ).toBe(2);
    expect(client.calls).toEqual([]);
    expect(errors.join("")).toContain(message);
  }
});

test("several unsupported filters are all named in one error", async () => {
  expect(
    runCli(
      ["search", "--query", "x", "--tag", "a", "--author", "b"],
      () => new FakeClient(),
      () => {},
    ),
  ).rejects.toThrow("--author, --tag");
});
