import { expect, test } from "bun:test";

import type { AgentKbClient } from "../src/client";
import { defaultCliDependencies, runMain } from "../src/cli";
import {
  assertNoLargeCollapse,
  makeCurrent,
  resolveSourceRegistry,
  runBoundedCommand,
  type MakeCurrentDependencies,
  type MakeCurrentReceipt,
  type SourcePlan,
} from "../src/make-current";
import type { RefreshResult } from "../src/refresh";
import type { GenerationManifest } from "../src/protocol";

const ID = "g-20260725T123456Z-001122aabbcc";
const client = { close() {} } as AgentKbClient;

function manifest(plan: SourcePlan): GenerationManifest {
  return {
    schema: 1,
    generation_id: ID,
    model: "lightonai/GTE-ModernColBERT-v1",
    corpus_count: 30,
    corpus_hash: "a".repeat(64),
    source_file_counts: { wiki: 2, "wiki:source": 4, chats: 2 },
    collection_counts: {
      wiki: { documents: 10, files: 2 },
      "wiki:source": { documents: 10, files: 4 },
      chats: { documents: 10, files: 2 },
    },
    sources: {
      schema: 1,
      items: plan.sources.map(({ export_roots: _, ...source }) => ({
        ...source,
        exported_document_count: Math.max(1, source.source_file_count),
      })),
    },
    exported_at: "2026-07-26T12:00:00Z",
  };
}

function dependencies(options: {
  readwiseExit?: number;
  readwiseCount?: number;
  historySyncExit?: number;
  lock?: boolean;
  archiveLock?: boolean;
} = {}) {
  const commands: string[][] = [];
  const writes = new Map<string, unknown>();
  let now = Date.parse("2026-07-26T12:00:00Z");
  const deps: MakeCurrentDependencies = {
    home: "/home/tester",
    now: () => new Date((now += 10)),
    randomId: () => "001122aabbcc",
    resolveRoots: async (wikiPath) => ({
      wikiRoot: wikiPath ?? "/wiki-projection",
      chatsReadableRoot: "/unused/readable",
    }),
    readConfig: async () => JSON.stringify({}),
    runCommand: async (args) => {
      commands.push(args);
      const readwise = args.some((arg) => arg.includes("readwise_tweets.py"));
      const historySync =
        args[0] === "agent-history-sync" && args[1] === "run";
      const exitCode = readwise
        ? options.readwiseExit ?? 0
        : historySync
        ? options.historySyncExit ?? 0
        : 0;
      return {
        exitCode,
        stdout: "{}",
        stderr: exitCode ? "network failed" : "",
      };
    },
    scan: async (roots) => ({
      count: roots.some((root) => root.includes("readwise-tweets"))
        ? options.readwiseCount ?? 3
        : roots.some((root) => root.includes("current.json"))
        ? 4
        : 3,
      newest: "2026-07-26T11:00:00Z",
    }),
    acquireLock: async () =>
      options.lock === false ? null : { release: async () => {} },
    acquireArchiveLock: async () =>
      options.archiveLock === false ? null : { release: async () => {} },
    validateHistoryGeneration: async () => ({
      databaseSha256: "a".repeat(64),
      catalogSha256: "b".repeat(64),
      databaseFilename: `history-index-${"a".repeat(64)}.sqlite3.zst`,
      catalogFilename: `provenance-catalog-${"b".repeat(64)}.jsonl`,
    }),
    readJson: async (path) => writes.get(path) ?? null,
    writeJsonAtomic: async (path, value) => {
      writes.set(path, value);
    },
    makeTempDirectory: async () => "/tmp/fake-make-current",
    removeLocal: async () => {},
    refresh: async (_client, refreshOptions) => {
      const generated = manifest(refreshOptions.sourcePlan);
      refreshOptions.validateBeforeStaging(generated);
      return {
        schema: 1,
        generation_id: ID,
        previous_generation_id: null,
        model: generated.model,
        corpus_count: generated.corpus_count,
        corpus_hash: generated.corpus_hash,
        document_batch_size: 256,
        document_batch_count: 1,
        embedding_dimension: 128,
        staged_embedding_bytes: 256,
        plaid_create_count: 1,
        plaid_kmeans_sample_size: 16_384,
        plaid_permutation_algorithm: "sha256-key-sort-v1",
        validation: {
          sqlite_count: 30,
          fts_count: 30,
          plaid_mapping_count: 30,
          plaid_reverse_mapping_count: 30,
          index_tree_hash: "b".repeat(64),
        },
        duration_ms: 1,
        staged_removed: true,
        manifest: generated,
      } as RefreshResult;
    },
    refreshDependencies: {} as MakeCurrentDependencies["refreshDependencies"],
  };
  return { deps, commands, writes };
}

test("source registry has deterministic defaults and path overrides", async () => {
  const registry = await resolveSourceRegistry(undefined, {
    home: "/home/tester",
    resolveRoots: async () => ({
      wikiRoot: "/wiki",
      chatsReadableRoot: "/unused",
    }),
    readConfig: async () =>
      JSON.stringify({
        source_paths: {
          "readwise-tweets": "~/custom/readwise",
          "agent-history-central": "/backup",
        },
        make_current: { collapse_ratio: 0.6, outer_timeout_minutes: 360 },
      }),
  });
  expect(
    registry.sources.find((source) => source.sourceId === "readwise-tweets")?.root,
  ).toBe("/home/tester/custom/readwise");
  expect(registry.backupRoot).toBe("/backup");
  expect(registry.collapseRatio).toBe(0.6);
  expect(
    registry.sources.find((source) => source.sourceId === "youtube-saved")
      ?.exportRoots[1]?.include,
  ).toEqual(["watch-history-latest.jsonl", "watch-later-latest.jsonl"]);
});

test("upstream success publishes healthy after backup run and status", async () => {
  const state = dependencies();
  const result = await makeCurrent(client, undefined, state.deps);
  expect(result.exitCode).toBe(0);
  expect(result.receipt.health).toBe("healthy");
  expect(state.commands.slice(0, 2).map((args) => args.slice(0, 2))).toEqual([
    ["agent-history-sync", "run"],
    ["agent-history-sync", "status"],
  ]);
});

test("upstream failure publishes degraded only with a valid fallback", async () => {
  const state = dependencies({ readwiseExit: 1, readwiseCount: 3 });
  const result = await makeCurrent(client, undefined, state.deps);
  expect(result.exitCode).toBe(0);
  expect(result.receipt.health).toBe("degraded");
  expect(
    result.receipt.sources.find((source) => source.source_id === "readwise-tweets")
      ?.state,
  ).toBe("fallback");
});

test("history sync failure uses the verified nonempty backup as degraded fallback", async () => {
  const state = dependencies({ historySyncExit: 1 });
  const result = await makeCurrent(client, undefined, state.deps);
  expect(result.exitCode).toBe(0);
  expect(result.receipt.health).toBe("degraded");
  expect(
    result.receipt.sources.find(
      (source) => source.source_id === "agent-history-central",
    )?.state,
  ).toBe("fallback");
});

test("archive-lock overlap refuses to read or publish a moving generation", async () => {
  const state = dependencies({ archiveLock: false });
  const result = await makeCurrent(client, undefined, state.deps);
  expect(result.exitCode).toBe(1);
  expect(result.receipt.published).toBeFalse();
  expect(result.receipt.error).toContain("remained locked");
});

test("upstream failure with an empty fallback refuses publication", async () => {
  const state = dependencies({ readwiseExit: 1, readwiseCount: 0 });
  const result = await makeCurrent(client, undefined, state.deps);
  expect(result.exitCode).toBe(1);
  expect(result.receipt.published).toBeFalse();
  expect(result.receipt.error).toContain("readwise-tweets durable projection is empty");
});

test("collapse guard permits first run and rejects zero or large collapse", () => {
  const plan = dependencies();
  const sourcePlan = {
    schema: 1 as const,
    include_local_chats: false as const,
    sources: [],
  };
  const current = manifest(sourcePlan);
  expect(() => assertNoLargeCollapse(current, null, 0.5)).not.toThrow();
  const previous = {
    sources: [{
      source_id: "wiki-pages",
      source_file_count: 100,
    }],
    collection_counts: {
      wiki: { documents: 100, files: 1 },
      "wiki:source": { documents: 10, files: 1 },
      chats: { documents: 10, files: 1 },
    },
  } as MakeCurrentReceipt;
  current.sources!.items = [{
    source_id: "wiki-pages",
    mode: "projection",
    state: "fresh",
    operation: "validate",
    started_at: "x",
    finished_at: "x",
    duration_ms: 0,
    root: "/wiki",
    source_file_count: 40,
    exported_document_count: 1,
    newest_source_timestamp: null,
    freshness_threshold_minutes: null,
    age_minutes: null,
    warning: null,
    error: null,
  }];
  expect(() => assertNoLargeCollapse(current, previous, 0.5)).toThrow(
    /source collapse/,
  );
  current.collection_counts!.chats.documents = 0;
  expect(() => assertNoLargeCollapse(current, null, 0.5)).toThrow(
    /zero collection/,
  );
  expect(plan).toBeDefined();
});

test("live overlap writes a receipt and exits 75 without source work", async () => {
  const state = dependencies({ lock: false });
  const result = await makeCurrent(client, undefined, state.deps);
  expect(result.exitCode).toBe(75);
  expect(result.receipt.state).toBe("overlap");
  expect(state.commands).toHaveLength(0);
});

test("make-current JSON output preserves the command exit code", async () => {
  const state = dependencies({ lock: false });
  const stdout: string[] = [];
  const stderr: string[] = [];
  const code = await runMain({
    args: ["make-current", "--json"],
    stdout: (value) => stdout.push(value),
    stderr: (value) => stderr.push(value),
    clientFactory: () => client,
    dependencies: {
      ...defaultCliDependencies,
      makeCurrent: state.deps,
    },
  });
  expect(code).toBe(75);
  expect(JSON.parse(stdout.join(""))).toMatchObject({
    state: "overlap",
    published: false,
  });
  expect(stderr).toEqual([]);
});

test("bounded subprocess terminates after its timeout", async () => {
  const result = await runBoundedCommand(
    ["/bin/sh", "-c", "sleep 1"],
    { timeoutMs: 10 },
  );
  expect(result.exitCode).toBe(124);
  expect(result.stderr).toContain("timed out");
});
