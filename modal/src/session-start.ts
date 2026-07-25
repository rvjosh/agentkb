#!/usr/bin/env bun

import {
  type AgentKbClient,
  ModalAgentKbClient,
} from "./client";

export interface SessionStartDependencies {
  readInput(): Promise<string>;
  env: Record<string, string | undefined>;
  clientFactory(): AgentKbClient;
}

const defaultDependencies: SessionStartDependencies = {
  readInput: () => Bun.stdin.text(),
  env: process.env,
  clientFactory: () => new ModalAgentKbClient(),
};

export async function handleSessionStart(
  dependencies: SessionStartDependencies = defaultDependencies,
): Promise<void> {
  let client: AgentKbClient | undefined;
  try {
    if (dependencies.env.AGENTKB_SKIP_WARM === "1") return;
    const value: unknown = JSON.parse(await dependencies.readInput());
    if (typeof value !== "object" || value === null || Array.isArray(value)) return;
    const input = value as Record<string, unknown>;
    if (input.hook_event_name !== "SessionStart") return;
    if (!["startup", "resume", "fork"].includes(String(input.source))) return;
    if ("agent_id" in input) return;
    client = dependencies.clientFactory();
    await client.warmDetached();
  } catch {
    // Session hooks must never delay or break session startup.
  } finally {
    client?.close();
  }
}

if (import.meta.main) {
  await handleSessionStart();
}
