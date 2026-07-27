import { expect, test } from "bun:test";
import type { ModalClient } from "modal";

import { ModalAgentKbClient } from "../src/client";

const ID = "g-20260725T123456Z-001122aabbcc";
const CURRENT_ID = "g-20260725T133456Z-112233aabbcc";

test("warmDetached awaits SDK spawn without awaiting GPU completion", async () => {
  const calls: unknown[][] = [];
  const modal = {
    functions: {
      fromName: async (...args: unknown[]) => {
        calls.push(["fromName", ...args]);
        return {
          spawn: async (args: unknown[]) => {
            calls.push(["spawn", args]);
            return { objectId: "fc-1" };
          },
        };
      },
    },
    close: () => calls.push(["close"]),
  } as unknown as ModalClient;
  const client = new ModalAgentKbClient(modal);
  await client.warmDetached();
  client.close();
  expect(calls).toEqual([
    ["fromName", "agentkb", "warm_current"],
    ["spawn", []],
    ["close"],
  ]);
});

test("search reconstructs local paths solely from relative_path", async () => {
  const modal = {
    functions: {
      fromName: async () => ({
        remote: async () => ({
          schema: 1,
          generation_id: ID,
          query: "test",
          k: 2,
          results: [
            {
              collection: "wiki",
              file: "/root/leak.md",
              path: "/root/leak.md",
              filename: "leak.md",
              relative_path: "wiki/local.md",
              line: 1,
              score: 1,
            },
            {
              collection: "chats",
              file: "/container/cwd/chat.md",
              path: "/container/cwd/chat.md",
              filename: "chat.md",
              relative_path: "2026-07/chat.md",
              line: 2,
              score: 0.5,
            },
          ],
        }),
      }),
    },
    close: () => {},
  } as unknown as ModalClient;
  const client = new ModalAgentKbClient(modal, {
    wikiRoot: "/Users/local/wiki",
    chatsReadableRoot: "/Users/local/chats/readable",
    externalRoots: {},
  });
  const result = await client.search("test", 2);
  expect(result.results.map((hit) => hit.path)).toEqual([
    "/Users/local/wiki/wiki/local.md",
    "/Users/local/chats/readable/2026-07/chat.md",
  ]);
  expect(result.results.map((hit) => hit.relative_path)).toEqual([
    "wiki/local.md",
    "2026-07/chat.md",
  ]);
  expect(JSON.stringify(result)).not.toContain("/root/");
  expect(JSON.stringify(result)).not.toContain("/container/cwd");
});

test("prune validates input and routes to the private CPU endpoint", async () => {
  const calls: unknown[][] = [];
  const modal = {
    functions: {
      fromName: async (...args: unknown[]) => {
        calls.push(["fromName", ...args]);
        return {
          remote: async (args: unknown[]) => {
            calls.push(["remote", args]);
            return {
              schema: 1,
              dry_run: true,
              deleted: false,
              target_generation_id: ID,
              current_generation_id: "g-20260725T130000Z-ddeeff001122",
              previous_generation_id: ID,
              final_previous_generation_id: ID,
            };
          },
        };
      },
    },
    close: () => {},
  } as unknown as ModalClient;
  const client = new ModalAgentKbClient(modal);
  expect((await client.prunePrevious(ID, true)).dry_run).toBeTrue();
  expect(calls).toEqual([
    ["fromName", "agentkb", "prune_previous"],
    ["remote", [ID, true]],
  ]);
  expect(client.prunePrevious("../escape", false)).rejects.toThrow();
  expect(calls).toHaveLength(2);
});

test("generation erasure methods validate and route only private SDK calls", async () => {
  const calls: unknown[][] = [];
  const modal = {
    functions: {
      fromName: async (_app: string, name: string) => ({
        remote: async (args: unknown[]) => {
          calls.push([name, args]);
          if (name === "generations") {
            return {
              schema: 1,
              current_generation_id: ID,
              previous_generation_id: null,
              items: [{
                generation_id: ID,
                type: "generation",
                classification: "current",
              }],
              counts: { current: 1, previous: 0, orphan: 0, staged: 0 },
            };
          }
          if (name === "find_session") {
            return {
              schema: 1,
              source: "codex",
              session_id: "session-1",
              canonical_file: "agent-history-central/codex/session-1.md",
              results: [],
              total_exact_match_count: 0,
              verification_failures: [],
              verified: true,
            };
          }
          return {
            schema: 1,
            dry_run: true,
            deleted: false,
            idempotent: false,
            target_id: ID,
            target_type: "staged",
            classification: "staged",
            current_generation_id: CURRENT_ID,
            operation_id: null,
            receipt: null,
          };
        },
      }),
    },
    close: () => {},
  } as unknown as ModalClient;
  const client = new ModalAgentKbClient(modal);
  expect((await client.generations()).counts.current).toBe(1);
  expect((await client.findSession("codex", "session-1")).verified).toBeTrue();
  expect(
    (await client.deleteGeneration(
      ID,
      "staged",
      CURRENT_ID,
      false,
      "test",
      "privacy",
    )).dry_run,
  ).toBeTrue();
  expect(calls).toEqual([
    ["generations", []],
    ["find_session", ["codex", "session-1"]],
    [
      "delete_generation_exact",
      [ID, "staged", CURRENT_ID, false, "test", "privacy", null],
    ],
  ]);
  expect(client.findSession("pi" as "codex", "session-1")).rejects.toThrow();
});
