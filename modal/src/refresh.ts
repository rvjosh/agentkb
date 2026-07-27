import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { isAbsolute, join, normalize, relative, resolve, sep } from "node:path";
import { mkdtemp, readFile, rm } from "node:fs/promises";

import type { AgentKbClient } from "./client";
import {
  DEFAULT_MODEL,
  BUILD_DOCUMENT_BATCH_SIZE,
  BUILD_PLAID_KMEANS_SAMPLE_SIZE,
  BUILD_PLAID_PERMUTATION_ALGORITHM,
  type BuildResult,
  type GenerationManifest,
  SHA256_PATTERN,
  createGenerationId,
  generationPaths,
  validateManifest,
} from "./protocol";

export const VOLUME_NAME = "agentkb-data";
const MODAL_CLI_PREFIX = ["uvx", "--from", "modal==1.5.3", "modal"];
const COLLECTIONS = new Set(["wiki", "wiki:source", "chats"]);

export interface PathRoots {
  wikiRoot: string;
  chatsReadableRoot: string;
  externalRoots: Record<string, string>;
}

export interface RefreshOptions {
  wikiPath?: string;
  chatsRoot?: string;
  sourcePlan?: unknown;
  validateBeforeStaging?: (manifest: GenerationManifest) => void;
}

export interface CommandResult {
  exitCode: number;
  stdout: string;
  stderr: string;
}

export interface RefreshDependencies {
  createGenerationId(): string;
  makeTempDirectory(prefix: string): Promise<string>;
  removeLocal(path: string): Promise<void>;
  readText(path: string): Promise<string>;
  streamBytes(path: string): AsyncIterable<Uint8Array>;
  runCommand(args: string[], timeoutMs?: number): Promise<CommandResult>;
  resolveRoots(wikiPath?: string): Promise<PathRoots>;
}

export interface RefreshResult extends BuildResult {
  staged_removed: true;
  manifest: GenerationManifest;
}

function expandHome(path: string, home: string): string {
  if (path === "~") return home;
  if (path.startsWith("~/")) return join(home, path.slice(2));
  return path;
}

export async function resolvePathRoots(
  wikiOverride?: string,
  home = homedir(),
  configReader: (path: string) => Promise<string> = async (path) =>
    readFile(path, "utf8"),
): Promise<PathRoots> {
  let config: Record<string, unknown> = {};
  try {
    const parsed: unknown = JSON.parse(
      await configReader(join(home, ".agentkb", "config.json")),
    );
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      throw new TypeError("~/.agentkb/config.json must contain a JSON object");
    }
    config = parsed as Record<string, unknown>;
  } catch (error) {
    if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) {
      throw error;
    }
  }
  const configuredWiki = config.wiki_path;
  const configuredChats = config.chats_path;
  const sourcePaths = config.source_paths;
  if (configuredWiki !== undefined && typeof configuredWiki !== "string") {
    throw new TypeError("wiki_path must be a string");
  }
  if (configuredChats !== undefined && typeof configuredChats !== "string") {
    throw new TypeError("chats_path must be a string");
  }
  if (
    sourcePaths !== undefined &&
    (typeof sourcePaths !== "object" ||
      sourcePaths === null ||
      Array.isArray(sourcePaths))
  ) {
    throw new TypeError("source_paths must be an object");
  }
  const configuredSources = (sourcePaths ?? {}) as Record<string, unknown>;
  const externalDefaults: Record<string, string> = {
    "readwise-tweets/": join(
      home,
      "home",
      "llm-wiki-generated",
      "readwise-tweets",
      "qmd-docs",
    ),
    "youtube-saved/": join(
      home,
      "home",
      "llm-wiki-generated",
      "youtube-playlists",
    ),
    "historical-chat-exports/": join(
      home,
      "home",
      "llm-wiki-generated",
      "chat-exports",
      "qmd-docs",
    ),
  };
  const externalRoots = Object.fromEntries(
    Object.entries(externalDefaults).map(([prefix, fallback]) => {
      const sourceId = prefix.slice(0, -1);
      const configured = configuredSources[sourceId];
      if (
        configured !== undefined &&
        (typeof configured !== "string" || !configured)
      ) {
        throw new TypeError(`source_paths.${sourceId} must be a non-empty string`);
      }
      return [
        prefix,
        resolve(expandHome((configured as string | undefined) ?? fallback, home)),
      ];
    }),
  );
  const wikiRoot = wikiOverride || configuredWiki || join(home, ".agentkb", "wiki");
  const chatsRoot = configuredChats || join(home, ".agentkb", "chats");
  return {
    wikiRoot: resolve(expandHome(wikiRoot, home)),
    chatsReadableRoot: resolve(expandHome(chatsRoot, home), "readable"),
    externalRoots,
  };
}

export interface CorpusValidation {
  hash: string;
  count: number;
}

function validateCorpusLine(
  line: string,
  lineNumber: number,
  canonicalIds: Set<string>,
): void {
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch (error) {
      throw new TypeError(`corpus line ${lineNumber} is invalid JSON`, {
        cause: error,
      });
    }
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      throw new TypeError(`corpus line ${lineNumber} must be an object`);
    }
    const record = value as Record<string, unknown>;
    if (!COLLECTIONS.has(String(record.collection))) {
      throw new TypeError(`corpus line ${lineNumber} has an unsupported collection`);
    }
    if (
      typeof record.file !== "string" ||
      !record.file ||
      isAbsolute(record.file) ||
      normalize(record.file).split(sep).includes("..") ||
      typeof record.content !== "string" ||
      typeof record.canonical_id !== "string" ||
      !SHA256_PATTERN.test(record.canonical_id)
    ) {
      throw new TypeError(`corpus line ${lineNumber} is invalid`);
    }
    if (canonicalIds.has(record.canonical_id)) {
      throw new TypeError(`corpus line ${lineNumber} has a duplicate canonical_id`);
    }
    canonicalIds.add(record.canonical_id);
}

export async function validateCorpusStream(
  chunks: AsyncIterable<Uint8Array>,
): Promise<CorpusValidation> {
  const hash = createHash("sha256");
  const decoder = new TextDecoder("utf-8", { fatal: true });
  const canonicalIds = new Set<string>();
  let pending = "";
  let count = 0;
  for await (const chunk of chunks) {
    hash.update(chunk);
    pending += decoder.decode(chunk, { stream: true });
    let newline = pending.indexOf("\n");
    while (newline !== -1) {
      const line = pending.slice(0, newline);
      pending = pending.slice(newline + 1);
      if (!line || line.endsWith("\r")) {
        throw new TypeError("corpus must contain non-empty LF-terminated lines");
      }
      count += 1;
      validateCorpusLine(line, count, canonicalIds);
      newline = pending.indexOf("\n");
    }
  }
  pending += decoder.decode();
  if (pending || count === 0) {
    throw new TypeError("corpus must have exactly one terminal newline");
  }
  return { hash: hash.digest("hex"), count };
}

function validateExportManifest(
  value: unknown,
  generationId: string,
  corpus: CorpusValidation,
  requireSourceMetadata = false,
): GenerationManifest {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError("export manifest must be an object");
  }
  const record = value as Record<string, unknown>;
  const sourceCounts = record.source_file_counts;
  if (
    typeof sourceCounts !== "object" ||
    sourceCounts === null ||
    Array.isArray(sourceCounts)
  ) {
    throw new TypeError("export manifest source_file_counts must be an object");
  }
  const counts = sourceCounts as Record<string, unknown>;
  if (
    Object.keys(counts).sort().join(",") !== "chats,wiki,wiki:source" ||
    Object.values(counts).some(
      (count) =>
        typeof count !== "number" || !Number.isInteger(count) || count < 0,
    )
  ) {
    throw new TypeError("export manifest source_file_counts is invalid");
  }
  if (
    typeof record.exported_at !== "string" ||
    !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$/.test(
      record.exported_at,
    ) ||
    Number.isNaN(Date.parse(record.exported_at))
  ) {
    throw new TypeError("export manifest exported_at must be an ISO date-time");
  }
  const manifest = validateManifest(value, "export manifest");
  if (manifest.generation_id !== generationId) {
    throw new TypeError("export manifest generation_id mismatch");
  }
  if (manifest.model !== DEFAULT_MODEL) {
    throw new TypeError(`export manifest model must be ${DEFAULT_MODEL}`);
  }
  if (manifest.corpus_hash !== corpus.hash) {
    throw new TypeError("export manifest corpus_hash mismatch");
  }
  if (manifest.corpus_count !== corpus.count) {
    throw new TypeError("export manifest corpus_count mismatch");
  }
  if (requireSourceMetadata) {
    if (!manifest.sources || !manifest.collection_counts) {
      throw new TypeError(
        "export manifest must include source receipts and collection counts",
      );
    }
    for (const collection of ["wiki", "wiki:source", "chats"] as const) {
      if (manifest.collection_counts[collection].documents < 1) {
        throw new TypeError(`export collection ${collection} must be nonempty`);
      }
    }
    const collectionTotal = Object.values(manifest.collection_counts).reduce(
      (total, count) => total + count.documents,
      0,
    );
    if (collectionTotal !== manifest.corpus_count) {
      throw new TypeError(
        "export collection document counts must sum to corpus_count",
      );
    }
  }
  return manifest;
}

function validateBuild(manifest: GenerationManifest, build: BuildResult): void {
  if (
    build.generation_id !== manifest.generation_id ||
    build.model !== manifest.model ||
    build.corpus_count !== manifest.corpus_count ||
    build.corpus_hash !== manifest.corpus_hash
  ) {
    throw new TypeError("Modal build result does not match the export manifest");
  }
  if (
    build.document_batch_size !== BUILD_DOCUMENT_BATCH_SIZE ||
    build.document_batch_count !==
      Math.ceil(manifest.corpus_count / build.document_batch_size) ||
    build.plaid_create_count !== 1 ||
    build.plaid_kmeans_sample_size !== BUILD_PLAID_KMEANS_SAMPLE_SIZE ||
    build.plaid_permutation_algorithm !== BUILD_PLAID_PERMUTATION_ALGORITHM
  ) {
    throw new TypeError("Modal build metrics do not match the corpus contract");
  }
  for (const name of [
    "sqlite_count",
    "fts_count",
    "plaid_mapping_count",
    "plaid_reverse_mapping_count",
  ] as const) {
    if (build.validation[name] !== manifest.corpus_count) {
      throw new TypeError(`Modal build validation ${name} does not match corpus_count`);
    }
  }
  if (
    build.validation.immutable_premerged?.document_count !== manifest.corpus_count
  ) {
    throw new TypeError(
      "Modal immutable premerged certificate does not match corpus_count",
    );
  }
}

async function defaultRunCommand(
  args: string[],
  timeoutMs = 30 * 60_000,
): Promise<CommandResult> {
  const process = Bun.spawn(args, { stdout: "pipe", stderr: "pipe" });
  let timedOut = false;
  let forceTimer: ReturnType<typeof setTimeout> | undefined;
  const timer = setTimeout(() => {
    timedOut = true;
    process.kill("SIGTERM");
    forceTimer = setTimeout(() => process.kill("SIGKILL"), 1_000);
  }, timeoutMs);
  const [exitCode, stdout, stderr] = await Promise.all([
    process.exited,
    new Response(process.stdout).text(),
    new Response(process.stderr).text(),
  ]).finally(() => {
    clearTimeout(timer);
    if (forceTimer !== undefined) clearTimeout(forceTimer);
  });
  if (timedOut) {
    return {
      exitCode: 124,
      stdout,
      stderr: `${stderr}${stderr && !stderr.endsWith("\n") ? "\n" : ""}timed out after ${timeoutMs}ms`,
    };
  }
  return { exitCode, stdout, stderr };
}

export const defaultRefreshDependencies: RefreshDependencies = {
  createGenerationId,
  makeTempDirectory: (prefix) => mkdtemp(join(tmpdir(), prefix)),
  removeLocal: (path) => rm(path, { recursive: true, force: true }),
  readText: (path) => readFile(path, "utf8"),
  streamBytes: (path) => createReadStream(path),
  runCommand: defaultRunCommand,
  resolveRoots: resolvePathRoots,
};

async function checkedCommand(
  dependencies: RefreshDependencies,
  args: string[],
  timeoutMs: number,
): Promise<CommandResult> {
  const result = await dependencies.runCommand(args, timeoutMs);
  if (result.exitCode !== 0) {
    throw new Error(
      `command failed (${args.join(" ")}): ${result.stderr.trim() || result.stdout.trim()}`,
    );
  }
  return result;
}

export async function refreshProduction(
  client: AgentKbClient,
  options: RefreshOptions = {},
  dependencies: RefreshDependencies = defaultRefreshDependencies,
): Promise<RefreshResult> {
  const generationId = dependencies.createGenerationId();
  const roots = await dependencies.resolveRoots(options.wikiPath);
  const localDirectory = await dependencies.makeTempDirectory(
    "agentkb-modal-refresh-",
  );
  const chatsRoot = options.chatsRoot ?? resolve(roots.chatsReadableRoot, "..");
  const paths = generationPaths(generationId);
  try {
    const exporterArgs = [
      "uv",
      "run",
      "python",
      "-m",
      "agentkb.modal_backend.exporter",
      "--generation-id",
      generationId,
      "--wiki-root",
      roots.wikiRoot,
      "--chats-root",
      chatsRoot,
      "--output-dir",
      localDirectory,
    ];
    if (options.sourcePlan !== undefined) {
      exporterArgs.push(
        "--source-plan-json",
        JSON.stringify(options.sourcePlan),
      );
    }
    await checkedCommand(dependencies, exporterArgs, 60 * 60_000);
    const corpusPath = join(localDirectory, "corpus.jsonl");
    const manifestPath = join(localDirectory, "manifest.json");
    const [corpus, manifestText] = await Promise.all([
      validateCorpusStream(dependencies.streamBytes(corpusPath)),
      dependencies.readText(manifestPath),
    ]);
    const manifestValue: unknown = JSON.parse(manifestText);
    const manifest = validateExportManifest(
      manifestValue,
      generationId,
      corpus,
      options.sourcePlan !== undefined,
    );
    options.validateBeforeStaging?.(manifest);

    await checkedCommand(dependencies, [
      ...MODAL_CLI_PREFIX,
      "volume",
      "put",
      VOLUME_NAME,
      corpusPath,
      paths.stagedCorpus,
    ], 30 * 60_000);
    await checkedCommand(dependencies, [
      ...MODAL_CLI_PREFIX,
      "volume",
      "put",
      VOLUME_NAME,
      manifestPath,
      paths.stagedManifest,
    ], 10 * 60_000);
    const build = await client.build(generationId);
    validateBuild(manifest, build);
    await checkedCommand(dependencies, [
      ...MODAL_CLI_PREFIX,
      "volume",
      "rm",
      VOLUME_NAME,
      `staged/${generationId}`,
      "--recursive",
    ], 5 * 60_000);
    return { ...build, staged_removed: true, manifest };
  } finally {
    await dependencies.removeLocal(localDirectory);
  }
}

export function localPath(root: string, storedRelativePath: string): string {
  if (!storedRelativePath || isAbsolute(storedRelativePath)) {
    throw new TypeError("stored relative_path must be non-empty and relative");
  }
  const absoluteRoot = resolve(root);
  const localized = resolve(absoluteRoot, storedRelativePath);
  const fromRoot = relative(absoluteRoot, localized);
  if (fromRoot === ".." || fromRoot.startsWith(`..${sep}`)) {
    throw new TypeError("stored relative_path escapes its collection root");
  }
  return localized;
}
