import { expect, test } from "bun:test";

import type { AgentKbClient } from "../src/client";
import { handleSessionStart } from "../src/session-start";
import type {
  BuildResult,
  PrunePreviousResult,
  SearchResult,
  StatusResponse,
  WarmResult,
} from "../src/protocol";

class HookClient implements AgentKbClient {
  warmed = 0;
  closed = 0;
  async status(): Promise<StatusResponse> { throw new Error("unused"); }
  async warm(): Promise<WarmResult> { throw new Error("unused"); }
  async warmDetached(): Promise<void> { this.warmed += 1; }
  async search(): Promise<SearchResult> { throw new Error("unused"); }
  async build(): Promise<BuildResult> { throw new Error("unused"); }
  async prunePrevious(): Promise<PrunePreviousResult> { throw new Error("unused"); }
  close(): void { this.closed += 1; }
}

async function run(input: string, env: Record<string, string | undefined> = {}) {
  const client = new HookClient();
  let factories = 0;
  await handleSessionStart({
    readInput: async () => input,
    env,
    clientFactory: () => {
      factories += 1;
      return client;
    },
  });
  return { client, factories };
}

test.each(["startup", "resume", "fork"])("warms accepted %s sessions", async (source) => {
  const { client, factories } = await run(
    JSON.stringify({ hook_event_name: "SessionStart", source }),
  );
  expect(factories).toBe(1);
  expect(client.warmed).toBe(1);
  expect(client.closed).toBe(1);
});

test("filters invalid, agent, skipped, and unrelated hook input", async () => {
  for (const [input, env] of [
    ["not-json", {}],
    [JSON.stringify({ hook_event_name: "Other", source: "startup" }), {}],
    [JSON.stringify({ hook_event_name: "SessionStart", source: "compact" }), {}],
    [
      JSON.stringify({
        hook_event_name: "SessionStart",
        source: "startup",
        agent_id: "agent-1",
      }),
      {},
    ],
    [
      JSON.stringify({ hook_event_name: "SessionStart", source: "startup" }),
      { AGENTKB_SKIP_WARM: "1" },
    ],
  ] as const) {
    const { factories } = await run(input, env);
    expect(factories).toBe(0);
  }
});

test("swallows detached warm failures and closes the client", async () => {
  const client = new HookClient();
  client.warmDetached = async () => {
    throw new Error("offline");
  };
  await handleSessionStart({
    readInput: async () =>
      JSON.stringify({ hook_event_name: "SessionStart", source: "startup" }),
    env: {},
    clientFactory: () => client,
  });
  expect(client.closed).toBe(1);
});
