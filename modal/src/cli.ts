#!/usr/bin/env bun

import {
  type AgentKbClient,
  createModalAgentKbClient,
} from "./client";
import {
  type PathRoots,
  type RefreshDependencies,
  defaultRefreshDependencies,
  refreshProduction,
  resolvePathRoots,
} from "./refresh";
import {
  assertJsonSerializable,
  createGenerationId,
  validateGenerationId,
  validateSearchRequest,
} from "./protocol";

export type ClientFactory = (roots?: PathRoots) => AgentKbClient;
export type Output = (line: string) => void;

export interface CliDependencies {
  refresh: RefreshDependencies;
  resolveRoots(wikiPath?: string): Promise<PathRoots>;
}

const defaultCliDependencies: CliDependencies = {
  refresh: defaultRefreshDependencies,
  resolveRoots: resolvePathRoots,
};

function usage(): string {
  return [
    "Usage:",
    "  bun run modal/src/cli.ts status",
    "  bun run modal/src/cli.ts warm",
    "  bun run modal/src/cli.ts search --query <text> [--k <1-100>]",
    "  bun run modal/src/cli.ts refresh [--wiki-path <path>]",
    "  bun run modal/src/cli.ts build --generation-id <id>",
    "  bun run modal/src/cli.ts generation-id",
  ].join("\n");
}

function option(args: string[], name: string): string | undefined {
  const index = args.indexOf(name);
  if (index === -1) return undefined;
  const value = args[index + 1];
  if (value === undefined || value.startsWith("--")) {
    throw new TypeError(`${name} requires a value`);
  }
  return value;
}

function ensureKnownOptions(args: string[], names: string[]): void {
  for (let index = 0; index < args.length; index += 1) {
    const value = args[index];
    if (!value?.startsWith("--")) {
      throw new TypeError(`unexpected argument: ${value ?? ""}`);
    }
    if (!names.includes(value)) throw new TypeError(`unknown option: ${value}`);
    index += 1;
    if (args[index] === undefined) throw new TypeError(`${value} requires a value`);
  }
}

function writeJson(output: Output, value: unknown): void {
  output(JSON.stringify(assertJsonSerializable(value), null, 2));
}

export async function runCli(
  argv: string[],
  clientFactory: ClientFactory = createModalAgentKbClient,
  output: Output = console.log,
  dependencies: CliDependencies = defaultCliDependencies,
): Promise<void> {
  const [command, ...args] = argv;
  if (!command || command === "--help" || command === "help") {
    output(usage());
    return;
  }
  if (command === "generation-id") {
    if (args.length) throw new TypeError("generation-id takes no arguments");
    output(createGenerationId());
    return;
  }

  let client: AgentKbClient | undefined;
  try {
    switch (command) {
      case "status":
        if (args.length) throw new TypeError("status takes no arguments");
        client = clientFactory();
        writeJson(output, await client.status());
        return;
      case "warm":
        if (args.length) throw new TypeError("warm takes no arguments");
        client = clientFactory();
        writeJson(output, await client.warm());
        return;
      case "search": {
        ensureKnownOptions(args, ["--query", "--k"]);
        const request = validateSearchRequest({
          query: option(args, "--query"),
          k: option(args, "--k") === undefined ? 10 : Number(option(args, "--k")),
        });
        const roots = await dependencies.resolveRoots();
        client = clientFactory(roots);
        writeJson(output, await client.search(request.query, request.k));
        return;
      }
      case "refresh": {
        ensureKnownOptions(args, ["--wiki-path"]);
        const wikiPath = option(args, "--wiki-path");
        client = clientFactory();
        writeJson(
          output,
          await refreshProduction(
            client,
            wikiPath === undefined ? {} : { wikiPath },
            dependencies.refresh,
          ),
        );
        return;
      }
      case "build": {
        ensureKnownOptions(args, ["--generation-id"]);
        const generationId = validateGenerationId(
          option(args, "--generation-id"),
        );
        client = clientFactory();
        writeJson(output, await client.build(generationId));
        return;
      }
      default:
        throw new TypeError(`unknown command: ${command}\n${usage()}`);
    }
  } finally {
    client?.close();
  }
}

if (import.meta.main) {
  try {
    await runCli(Bun.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
