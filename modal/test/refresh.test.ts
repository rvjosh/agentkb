import { createHash } from "node:crypto";
import { expect, test } from "bun:test";

import type { AgentKbClient } from "../src/client";
import {
  type CommandResult,
  type RefreshDependencies,
  localPath,
  refreshProduction,
  resolvePathRoots,
  validateCorpusStream,
} from "../src/refresh";
import type {
  BuildResult,
  PrunePreviousResult,
  SearchResult,
  StatusResponse,
  WarmResult,
} from "../src/protocol";
import { DEFAULT_MODEL } from "../src/protocol";

const ID = "g-20260725T123456Z-001122aabbcc";

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

class RefreshClient implements AgentKbClient {
  buildCalls: string[] = [];
  failBuild = false;

  async status(): Promise<StatusResponse> {
    throw new Error("not used");
  }
  async warm(): Promise<WarmResult> {
    throw new Error("not used");
  }
  async warmDetached(): Promise<void> {
    throw new Error("not used");
  }
  async search(): Promise<SearchResult> {
    throw new Error("not used");
  }
  async build(generationId: string): Promise<BuildResult> {
    this.buildCalls.push(generationId);
    if (this.failBuild) throw new Error("build failed");
    return buildResult(generationId);
  }
  async prunePrevious(): Promise<PrunePreviousResult> {
    throw new Error("not used");
  }
  close(): void {}
}

function buildResult(generationId = ID): BuildResult {
  return {
    schema: 1,
    generation_id: generationId,
    previous_generation_id: null,
    model: DEFAULT_MODEL,
    corpus_count: 1,
    corpus_hash: corpusHash(),
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

const corpus = new TextEncoder().encode(
  `${JSON.stringify({
    canonical_id: "a".repeat(64),
    collection: "wiki",
    content: "content",
    file: "wiki/page.md",
  })}\n`,
);

function corpusHash(): string {
  return createHash("sha256").update(corpus).digest("hex");
}

function dependencies(
  commands: string[][],
  removed: string[],
  manifestOverrides: Record<string, unknown> = {},
): RefreshDependencies {
  const manifest = {
    schema: 1,
    generation_id: ID,
    model: DEFAULT_MODEL,
    corpus_count: 1,
    corpus_hash: corpusHash(),
    source_file_counts: { chats: 0, wiki: 1, "wiki:source": 0 },
    exported_at: "2026-07-25T12:34:56Z",
    ...manifestOverrides,
  };
  return {
    createGenerationId: () => ID,
    makeTempDirectory: async (prefix) => {
      expect(prefix).toBe("agentkb-modal-refresh-");
      return "/tmp/agentkb-modal-refresh-test";
    },
    removeLocal: async (path) => {
      removed.push(path);
    },
    readText: async () => JSON.stringify(manifest),
    streamBytes: async function* () {
      yield corpus;
    },
    runCommand: async (args): Promise<CommandResult> => {
      commands.push(args);
      return { exitCode: 0, stdout: "{}", stderr: "" };
    },
    resolveRoots: async () => ({
      wikiRoot: "/local/wiki root",
      chatsReadableRoot: "/local/chats/readable",
      externalRoots: {},
    }),
  };
}

test("refresh validates locally, stages corpus then manifest, builds, and removes staging", async () => {
  const commands: string[][] = [];
  const removed: string[] = [];
  const client = new RefreshClient();
  const result = await refreshProduction(client, {}, dependencies(commands, removed));

  expect(commands[0]).toEqual([
    "uv",
    "run",
    "python",
    "-m",
    "agentkb.modal_backend.exporter",
    "--generation-id",
    ID,
    "--wiki-root",
    "/local/wiki root",
    "--chats-root",
    "/local/chats",
    "--output-dir",
    "/tmp/agentkb-modal-refresh-test",
  ]);
  expect(commands.slice(1).map((args) => args.slice(0, 7))).toEqual([
    ["uvx", "--from", "modal==1.5.3", "modal", "volume", "put", "agentkb-data"],
    ["uvx", "--from", "modal==1.5.3", "modal", "volume", "put", "agentkb-data"],
    ["uvx", "--from", "modal==1.5.3", "modal", "volume", "rm", "agentkb-data"],
  ]);
  expect(commands[1]!.at(-1)).toBe(`staged/${ID}/corpus.jsonl`);
  expect(commands[2]!.at(-1)).toBe(`staged/${ID}/manifest.json`);
  expect(commands[3]!.slice(-2)).toEqual([`staged/${ID}`, "--recursive"]);
  expect(client.buildCalls).toEqual([ID]);
  expect(removed).toEqual(["/tmp/agentkb-modal-refresh-test"]);
  expect(result.staged_removed).toBeTrue();
});

test("refresh preserves remote staging on build failure and removes local temp", async () => {
  const commands: string[][] = [];
  const removed: string[] = [];
  const client = new RefreshClient();
  client.failBuild = true;
  expect(
    refreshProduction(client, {}, dependencies(commands, removed)),
  ).rejects.toThrow("build failed");
  expect(commands.some((args) => args.includes("rm"))).toBeFalse();
  expect(removed).toEqual(["/tmp/agentkb-modal-refresh-test"]);
});

test("refresh rejects corpus hash before staging", async () => {
  const commands: string[][] = [];
  const removed: string[] = [];
  expect(
    refreshProduction(
      new RefreshClient(),
      {},
      dependencies(commands, removed, { corpus_hash: "0".repeat(64) }),
    ),
  ).rejects.toThrow("corpus_hash mismatch");
  expect(commands).toHaveLength(1);
  expect(removed).toHaveLength(1);
});

test("refresh validates every remote index count and preserves staging on mismatch", async () => {
  const commands: string[][] = [];
  const removed: string[] = [];
  const client = new RefreshClient();
  client.build = async () => ({
    ...buildResult(),
    validation: {
      ...buildResult().validation,
      plaid_reverse_mapping_count: 0,
    },
  });
  expect(
    refreshProduction(client, {}, dependencies(commands, removed)),
  ).rejects.toThrow("plaid_reverse_mapping_count");
  expect(commands.some((args) => args.includes("rm"))).toBeFalse();
  expect(removed).toHaveLength(1);
});

test("refresh forwards the wiki override to injected root resolution", async () => {
  const commands: string[][] = [];
  const removed: string[] = [];
  const deps = dependencies(commands, removed);
  let override: string | undefined;
  deps.resolveRoots = async (wikiPath) => {
    override = wikiPath;
    return {
      wikiRoot: "/override/wiki",
      chatsReadableRoot: "/local/chats/readable",
      externalRoots: {},
    };
  };
  await refreshProduction(
    new RefreshClient(),
    { wikiPath: "/override/wiki" },
    deps,
  );
  expect(override).toBe("/override/wiki");
});

test("path roots use override, config, and portable fallbacks", async () => {
  const configured = await resolvePathRoots(
    "~/override wiki",
    "/home/tester",
    async () => JSON.stringify({ wiki_path: "/ignored", chats_path: "~/chat-data" }),
  );
  expect(configured).toEqual({
    wikiRoot: "/home/tester/override wiki",
    chatsReadableRoot: "/home/tester/chat-data/readable",
    externalRoots: {
      "historical-chat-exports/":
        "/home/tester/home/llm-wiki-generated/chat-exports/qmd-docs",
      "readwise-tweets/":
        "/home/tester/home/llm-wiki-generated/readwise-tweets/qmd-docs",
      "youtube-saved/":
        "/home/tester/home/llm-wiki-generated/youtube-playlists",
    },
  });
  const fallback = await resolvePathRoots(
    undefined,
    "/home/tester",
    async () => {
      const error = new Error("missing") as Error & { code: string };
      error.code = "ENOENT";
      throw error;
    },
  );
  expect(fallback).toEqual({
    wikiRoot: "/home/tester/.agentkb/wiki",
    chatsReadableRoot: "/home/tester/.agentkb/chats/readable",
    externalRoots: {
      "historical-chat-exports/":
        "/home/tester/home/llm-wiki-generated/chat-exports/qmd-docs",
      "readwise-tweets/":
        "/home/tester/home/llm-wiki-generated/readwise-tweets/qmd-docs",
      "youtube-saved/":
        "/home/tester/home/llm-wiki-generated/youtube-playlists",
    },
  });
});

test("localPath rejects absolute and escaping stored paths", () => {
  expect(localPath("/local/wiki", "wiki/page.md")).toBe(
    "/local/wiki/wiki/page.md",
  );
  expect(() => localPath("/local/wiki", "/root/index/page.md")).toThrow(
    /must be non-empty and relative/,
  );
  expect(() => localPath("/local/wiki", "../escape")).toThrow(/escapes/);
});

test("stream validation handles JSONL split across arbitrary byte boundaries", async () => {
  const encoder = new TextEncoder();
  const lines = [
    {
      canonical_id: "1".repeat(64),
      collection: "wiki",
      content: "café",
      file: "wiki/one.md",
    },
    {
      canonical_id: "2".repeat(64),
      collection: "chats",
      content: "second",
      file: "2026-07/two.md",
    },
  ];
  const bytes = encoder.encode(`${lines.map((line) => JSON.stringify(line)).join("\n")}\n`);
  async function* chunks(): AsyncGenerator<Uint8Array> {
    const widths = [1, 2, 7, 3, 11, 1, 5];
    let offset = 0;
    let index = 0;
    while (offset < bytes.length) {
      const end = Math.min(bytes.length, offset + widths[index % widths.length]!);
      yield bytes.slice(offset, end);
      offset = end;
      index += 1;
    }
  }

  expect(await validateCorpusStream(chunks())).toEqual({
    hash: createHash("sha256").update(bytes).digest("hex"),
    count: 2,
  });
});

test("production corpus validation has no whole-file read boundary", async () => {
  const source = await Bun.file(new URL("../src/refresh.ts", import.meta.url)).text();
  expect(source).toContain(
    "validateCorpusStream(dependencies.streamBytes(corpusPath))",
  );
  expect(source).not.toContain("readBytes(corpusPath)");
  expect(source).not.toContain("readFile(corpusPath");
});
