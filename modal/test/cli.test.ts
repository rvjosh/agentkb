import { expect, test } from "bun:test";

import type { AgentKbClient } from "../src/client";
import { runCli } from "../src/cli";
import type {
  BuildResult,
  SearchResult,
  StatusResponse,
  WarmResult,
} from "../src/protocol";

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

  async search(query: string, k: number): Promise<SearchResult> {
    this.calls.push(["search", query, k]);
    return {
      schema: 1,
      generation_id: ID,
      query,
      k,
      results: [],
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

  close(): void {
    this.closed = true;
  }
}

test("routes search with parsed and default k", async () => {
  const client = new FakeClient();
  const output: string[] = [];
  await runCli(
    ["search", "--query", "private knowledge"],
    () => client,
    (line) => output.push(line),
  );
  expect(client.calls).toEqual([["search", "private knowledge", 10]]);
  expect(client.closed).toBeTrue();
  expect(JSON.parse(output[0]!)).toMatchObject({ query: "private knowledge", k: 10 });
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
