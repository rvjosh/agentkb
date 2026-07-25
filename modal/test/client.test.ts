import { expect, test } from "bun:test";
import type { ModalClient } from "modal";

import { ModalAgentKbClient } from "../src/client";

const ID = "g-20260725T123456Z-001122aabbcc";

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
