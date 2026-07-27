#!/usr/bin/env bun

import {
  type AgentKbClient,
  createModalAgentKbClient,
} from "./client";
import {
  type BillingSpawn,
  defaultBillingSpawn,
  reportCost,
} from "./cost";
import {
  type MakeCurrentDependencies,
  defaultMakeCurrentDependencies,
  makeCurrent,
} from "./make-current";
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
  type SearchHit,
  type SearchResult,
  validateGenerationId,
  validateSearchRequest,
} from "./protocol";

export const VERSION = "0.1.0";
export const DEFAULT_CONTENT_LIMIT = 1_200;
export type ClientFactory = (roots?: PathRoots) => AgentKbClient;
export type Output = (line: string) => void;

export class UsageError extends TypeError {}
export class CommandExitError extends Error {
  constructor(readonly exitCode: 1 | 75) {
    super(`command exited ${exitCode}`);
  }
}

export interface CliSearchHit extends SearchHit {
  content_truncated: boolean;
}

export interface CliSearchResult extends Omit<SearchResult, "results"> {
  results: CliSearchHit[];
}

export interface CliDependencies {
  refresh: RefreshDependencies;
  makeCurrent: MakeCurrentDependencies;
  resolveRoots(wikiPath?: string): Promise<PathRoots>;
  billingSpawn: BillingSpawn;
}

export interface MainEnv {
  args: string[];
  stdout: Output;
  stderr: Output;
  clientFactory: ClientFactory;
  dependencies: CliDependencies;
}

export const defaultCliDependencies: CliDependencies = {
  refresh: defaultRefreshDependencies,
  makeCurrent: defaultMakeCurrentDependencies,
  resolveRoots: resolvePathRoots,
  billingSpawn: defaultBillingSpawn,
};

export function usage(): string {
  return [
    "agentkb-modal - private AgentKB Modal control plane",
    "",
    "Usage:",
    "  agentkb-modal status",
    "  agentkb-modal warm",
    "  agentkb-modal search --query <text> [--k <1-100>] [--full-content]",
    "  agentkb-modal refresh [--wiki-path <path>]",
    "  agentkb-modal make-current [--wiki-path <path>] [--json]",
    "  agentkb-modal build --generation-id <id>",
    "  agentkb-modal prune-previous --generation-id <id> [--dry-run] [--force]",
    "  agentkb-modal cost [--days <1-7>]",
    "  agentkb-modal generation-id",
    "  agentkb-modal -h | --help",
    "  agentkb-modal --version",
    "",
    "Search content is limited to 1,200 characters by default; --full-content widens it.",
    "A real prune requires --force. --dry-run validates and plans without mutation.",
    "Cost reports are hourly metered usage for app description \"agentkb\".",
  ].join("\n");
}

function usageError(message: string): UsageError {
  return new UsageError(`${message}\nrun 'agentkb-modal --help' for usage.`);
}

function option(args: string[], name: string): string | undefined {
  const index = args.indexOf(name);
  if (index === -1) return undefined;
  const value = args[index + 1];
  if (value === undefined || value.startsWith("--")) {
    throw usageError(`${name} requires a value`);
  }
  return value;
}

function ensureKnownOptions(
  args: string[],
  specs: Record<string, "value" | "boolean">,
): void {
  const seen = new Set<string>();
  for (let index = 0; index < args.length; index += 1) {
    const value = args[index];
    if (!value?.startsWith("--")) {
      throw usageError(`unexpected argument: ${value ?? ""}`);
    }
    const kind = specs[value];
    if (!kind) throw usageError(`unknown option: ${value}`);
    if (seen.has(value)) throw usageError(`option may be provided only once: ${value}`);
    seen.add(value);
    if (kind === "value") {
      index += 1;
      const next = args[index];
      if (next === undefined || next.startsWith("--")) {
        throw usageError(`${value} requires a value`);
      }
    }
  }
}

function validateUsage<T>(operation: () => T): T {
  try {
    return operation();
  } catch (error) {
    if (error instanceof TypeError) throw usageError(error.message);
    throw error;
  }
}

function writeJson(output: Output, value: unknown): void {
  output(JSON.stringify(assertJsonSerializable(value), null, 2));
}

export function shapeSearchResult(
  result: SearchResult,
  fullContent: boolean,
): CliSearchResult {
  return {
    ...result,
    results: result.results.map((hit) => {
      if (
        fullContent ||
        hit.content === undefined ||
        hit.content.length <= DEFAULT_CONTENT_LIMIT
      ) {
        return { ...hit, content_truncated: false };
      }
      return {
        ...hit,
        content: `${hit.content.slice(0, DEFAULT_CONTENT_LIMIT - 1)}…`,
        content_truncated: true,
      };
    }),
  };
}

export async function runCli(
  argv: string[],
  clientFactory: ClientFactory = createModalAgentKbClient,
  output: Output = console.log,
  dependencies: CliDependencies = defaultCliDependencies,
): Promise<void> {
  if (argv.some((arg) => arg === "-h" || arg === "--help")) {
    output(usage());
    return;
  }
  if (argv.includes("--version")) {
    output(`${VERSION}\n`);
    return;
  }

  const [command, ...args] = argv;
  if (!command || command === "help") {
    output(usage());
    return;
  }
  if (command === "generation-id") {
    if (args.length) throw usageError("generation-id takes no arguments");
    output(createGenerationId());
    return;
  }
  if (command === "cost") {
    ensureKnownOptions(args, { "--days": "value" });
    const rawDays = option(args, "--days");
    const days = rawDays === undefined ? 1 : Number(rawDays);
    if (!Number.isInteger(days) || days < 1 || days > 7) {
      throw usageError("--days must be an integer between 1 and 7");
    }
    writeJson(output, reportCost(days, dependencies.billingSpawn));
    return;
  }

  let client: AgentKbClient | undefined;
  try {
    switch (command) {
      case "status":
        if (args.length) throw usageError("status takes no arguments");
        client = clientFactory();
        writeJson(output, await client.status());
        return;
      case "warm":
        if (args.length) throw usageError("warm takes no arguments");
        client = clientFactory();
        writeJson(output, await client.warm());
        return;
      case "search": {
        ensureKnownOptions(args, {
          "--query": "value",
          "--k": "value",
          "--full-content": "boolean",
        });
        const request = validateUsage(() =>
          validateSearchRequest({
            query: option(args, "--query"),
            k: option(args, "--k") === undefined ? 10 : Number(option(args, "--k")),
          })
        );
        const roots = await dependencies.resolveRoots();
        client = clientFactory(roots);
        writeJson(
          output,
          shapeSearchResult(
            await client.search(request.query, request.k),
            args.includes("--full-content"),
          ),
        );
        return;
      }
      case "refresh": {
        ensureKnownOptions(args, { "--wiki-path": "value" });
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
      case "make-current": {
        ensureKnownOptions(args, {
          "--wiki-path": "value",
          "--json": "boolean",
        });
        const wikiPath = option(args, "--wiki-path");
        client = clientFactory();
        const execution = await makeCurrent(
          client,
          wikiPath,
          dependencies.makeCurrent,
        );
        writeJson(output, execution.receipt);
        if (execution.exitCode !== 0) {
          throw new CommandExitError(execution.exitCode);
        }
        return;
      }
      case "build": {
        ensureKnownOptions(args, { "--generation-id": "value" });
        const generationId = validateUsage(() =>
          validateGenerationId(option(args, "--generation-id"))
        );
        client = clientFactory();
        writeJson(output, await client.build(generationId));
        return;
      }
      case "prune-previous": {
        ensureKnownOptions(args, {
          "--generation-id": "value",
          "--dry-run": "boolean",
          "--force": "boolean",
        });
        const generationId = validateUsage(() =>
          validateGenerationId(option(args, "--generation-id"))
        );
        const dryRun = args.includes("--dry-run");
        if (!dryRun && !args.includes("--force")) {
          throw usageError("prune-previous requires --force unless --dry-run is used");
        }
        client = clientFactory();
        writeJson(output, await client.prunePrevious(generationId, dryRun));
        return;
      }
      default:
        throw usageError(`unknown command: ${command}`);
    }
  } finally {
    client?.close();
  }
}

export async function runMain(e: MainEnv): Promise<number> {
  try {
    await runCli(e.args, e.clientFactory, e.stdout, e.dependencies);
    return 0;
  } catch (error) {
    if (error instanceof CommandExitError) return error.exitCode;
    e.stderr(`${error instanceof Error ? error.message : String(error)}\n`);
    return error instanceof UsageError ? 2 : 1;
  }
}

async function main(): Promise<never> {
  const code = await runMain({
    args: Bun.argv.slice(2),
    stdout: (value) => process.stdout.write(value.endsWith("\n") ? value : `${value}\n`),
    stderr: (value) => process.stderr.write(value),
    clientFactory: createModalAgentKbClient,
    dependencies: defaultCliDependencies,
  });
  process.exit(code);
}

if (import.meta.main) await main();
