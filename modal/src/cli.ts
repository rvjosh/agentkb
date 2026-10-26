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
  type DocumentSearchResult,
  type SearchHit,
  type SearchResult,
  validateGenerationId,
  validateSessionId,
  validateSessionSource,
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

export type CliMetadataSearchHit = Omit<SearchHit, "content">;

export interface CliMetadataSearchResult extends Omit<SearchResult, "results"> {
  results: CliMetadataSearchHit[];
}

export interface CliDocumentSearchResult {
  schema: 1;
  generation_id: string;
  query: string;
  k: number;
  searched: string[];
  words_terms: string[];
  by_words: CliSearchHit[] | CliMetadataSearchHit[];
  by_meaning: CliSearchHit[] | CliMetadataSearchHit[];
  hint: string;
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
    "agentkb - private AgentKB Modal control plane (agentkb-modal is an alias)",
    "",
    "Usage:",
    "  agentkb status",
    "  agentkb generations --json",
    "  agentkb find-session --source <claude|codex|cursor|pi|opencode|openclaw|heyclicky|grok-bot> --session-id <id> --json",
    "  agentkb delete-generation --generation-id <id> --expected-current <id> --actor <actor> --reason <reason> [--exact-session-key <source/id>] [--force] [--json]",
    "  agentkb delete-staged --generation-id <id> --expected-current <id> --actor <actor> --reason <reason> [--exact-session-key <source/id>] [--force] [--json]",
    "  agentkb warm",
    "  agentkb search --query <text> [--k <1-100>] [--include-chats | --transcript-sessions] [--full-content | --metadata-only]",
    "  agentkb refresh [--wiki-path <path>]",
    "  agentkb make-current [--wiki-path <path>] [--json]",
    "  agentkb build --generation-id <id>",
    "  agentkb prune-previous --generation-id <id> [--dry-run] [--force]",
    "  agentkb cost [--days <1-7>]",
    "  agentkb generation-id",
    "  agentkb -h | --help",
    "  agentkb --version",
    "",
    "Search returns two separate lists over wiki pages and sources: by_words (exact words, keyword rank) and by_meaning (semantic rank). --include-chats adds agent chats to both lists.",
    "Search content is limited to 1,200 characters by default; --full-content widens it and --metadata-only omits it.",
    "Generation deletion is dry-run by default and requires --force to mutate.",
    "Cost reports are hourly metered usage for app description \"agentkb\".",
  ].join("\n");
}

function usageError(message: string): UsageError {
  return new UsageError(`${message}\nrun 'agentkb --help' for usage.`);
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

function shapeHit(hit: SearchHit, fullContent: boolean): CliSearchHit {
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
}

function metadataOnlyHit({
  content: _content,
  ...metadata
}: SearchHit): CliMetadataSearchHit {
  return metadata;
}

export function shapeSearchResult(
  result: SearchResult,
  fullContent: boolean,
): CliSearchResult {
  return {
    ...result,
    results: result.results.map((hit) => shapeHit(hit, fullContent)),
  };
}

export function metadataOnlySearchResult(
  result: SearchResult,
): CliMetadataSearchResult {
  return { ...result, results: result.results.map(metadataOnlyHit) };
}

/** Quote for a POSIX shell: single quotes, embedded ' as '\''. */
export function shellQuote(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

/** g-YYYYMMDDTHHMMSSZ-... -> "YYYY-MM-DD HH:MM UTC". */
export function indexBuildTime(generationId: string): string {
  const match = /^g-(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})\d{2}Z-/.exec(generationId);
  if (!match) return "at an unknown time";
  const [, year, month, day, hour, minute] = match;
  return `${year}-${month}-${day} ${hour}:${minute} UTC`;
}

export function searchHint(
  mode: "documents" | "include-chats" | "transcript-sessions",
  query: string,
  generationId: string,
): string {
  const q = shellQuote(query);
  const built = indexBuildTime(generationId);
  switch (mode) {
    case "documents":
      return `Not searched: chats (agent sessions). Include them: agentkb search --query ${q} --include-chats · today's chats by exact words: agent-history search ${q}`;
    case "include-chats":
      return `Chats come from the nightly index built ${built}; for chats since then: agent-history search ${q}`;
    case "transcript-sessions":
      return `Searched chats only (nightly index built ${built}). Documents: agentkb search --query ${q} · today's chats by exact words: agent-history search ${q}`;
  }
}

export function shapeDocumentSearchResult(
  result: DocumentSearchResult,
  content: "bounded" | "full" | "metadata",
  includeChats: boolean,
): CliDocumentSearchResult {
  const shape = (hits: SearchHit[]) =>
    content === "metadata"
      ? hits.map(metadataOnlyHit)
      : hits.map((hit) => shapeHit(hit, content === "full"));
  return {
    schema: result.schema,
    generation_id: result.generation_id,
    query: result.query,
    k: result.k,
    searched: result.collections,
    words_terms: result.words_terms,
    by_words: shape(result.by_words),
    by_meaning: shape(result.by_meaning),
    hint: searchHint(
      includeChats ? "include-chats" : "documents",
      result.query,
      result.generation_id,
    ),
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
      case "generations":
        ensureKnownOptions(args, { "--json": "boolean" });
        client = clientFactory();
        writeJson(output, await client.generations());
        return;
      case "find-session": {
        ensureKnownOptions(args, {
          "--source": "value",
          "--session-id": "value",
          "--json": "boolean",
        });
        const source = validateUsage(() =>
          validateSessionSource(option(args, "--source"))
        );
        const sessionId = validateUsage(() =>
          validateSessionId(option(args, "--session-id"))
        );
        client = clientFactory();
        writeJson(output, await client.findSession(source, sessionId));
        return;
      }
      case "delete-generation":
      case "delete-staged": {
        ensureKnownOptions(args, {
          "--generation-id": "value",
          "--expected-current": "value",
          "--actor": "value",
          "--reason": "value",
          "--exact-session-key": "value",
          "--force": "boolean",
          "--json": "boolean",
        });
        const generationId = validateUsage(() =>
          validateGenerationId(option(args, "--generation-id"))
        );
        const expectedCurrent = validateUsage(() =>
          validateGenerationId(option(args, "--expected-current"))
        );
        const actor = option(args, "--actor");
        const reason = option(args, "--reason");
        if (!actor?.trim()) throw usageError("--actor is required");
        if (!reason?.trim()) throw usageError("--reason is required");
        client = clientFactory();
        writeJson(
          output,
          await client.deleteGeneration(
            generationId,
            command === "delete-staged" ? "staged" : "generation",
            expectedCurrent,
            args.includes("--force"),
            actor,
            reason,
            option(args, "--exact-session-key"),
          ),
        );
        return;
      }
      case "warm":
        if (args.length) throw usageError("warm takes no arguments");
        client = clientFactory();
        writeJson(output, await client.warm());
        return;
      case "search": {
        ensureKnownOptions(args, {
          "--query": "value",
          "--k": "value",
          "--transcript-sessions": "boolean",
          "--include-chats": "boolean",
          "--full-content": "boolean",
          "--metadata-only": "boolean",
        });
        const fullContent = args.includes("--full-content");
        const metadataOnly = args.includes("--metadata-only");
        if (fullContent && metadataOnly) {
          throw usageError(
            "--metadata-only cannot be combined with --full-content",
          );
        }
        const transcriptSessions = args.includes("--transcript-sessions");
        const includeChats = args.includes("--include-chats");
        if (transcriptSessions && includeChats) {
          throw usageError(
            "--include-chats cannot be combined with --transcript-sessions",
          );
        }
        const request = validateUsage(() =>
          validateSearchRequest({
            query: option(args, "--query"),
            k: option(args, "--k") === undefined ? 10 : Number(option(args, "--k")),
            ...(transcriptSessions ? { transcript_sessions: true } : {}),
          })
        );
        const roots = await dependencies.resolveRoots();
        client = clientFactory(roots);
        if (!transcriptSessions) {
          const result = await client.searchDocuments(
            request.query,
            request.k,
            includeChats,
          );
          writeJson(
            output,
            shapeDocumentSearchResult(
              result,
              metadataOnly ? "metadata" : fullContent ? "full" : "bounded",
              includeChats,
            ),
          );
          return;
        }
        const result = await client.search(request.query, request.k, true);
        writeJson(output, {
          ...(metadataOnly
            ? metadataOnlySearchResult(result)
            : shapeSearchResult(result, fullContent)),
          hint: searchHint(
            "transcript-sessions",
            result.query,
            result.generation_id,
          ),
        });
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
