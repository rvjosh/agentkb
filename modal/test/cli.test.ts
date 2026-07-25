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
      ready: true,
    };
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
      validation: {
        sqlite_count: 1,
        fts_count: 1,
        plaid_mapping_count: 1,
        plaid_reverse_mapping_count: 1,
        index_tree_hash: "b".repeat(64),
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
  expect(client.closed).toBeTrue();
});
