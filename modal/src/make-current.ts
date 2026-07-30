import { createHash, randomBytes } from "node:crypto";
import { createReadStream } from "node:fs";
import {
  chmod,
  lstat,
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";

import type { AgentKbClient } from "./client";
import {
  type CommandResult,
  type RefreshDependencies,
  type RefreshResult,
  defaultRefreshDependencies,
  refreshProduction,
  resolvePathRoots,
} from "./refresh";
import type {
  CorpusCollection,
  GenerationManifest,
  SourceMode,
  SourceReceipt,
  SourceState,
} from "./protocol";

export const RECEIPT_SCHEMA = 1;
export const SOURCE_PLAN_SCHEMA = 1;
export const DEFAULT_COLLAPSE_RATIO = 0.5;
export const DEFAULT_OUTER_TIMEOUT_MS = 5 * 60 * 60_000;
export const DEFAULT_COMMAND_TIMEOUT_MS = 30 * 60_000;
export const BACKUP_FRESHNESS_MINUTES = 90;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const ARCHIVE_POINTER_SCHEMA = 1;
const ARCHIVE_SCHEMA = 5;

export interface SourceExportRoot {
  path: string;
  collection: CorpusCollection;
  prefix: string;
  kind: "markdown" | "jsonl";
  include?: string[];
}

export interface PlannedSource extends SourceReceipt {
  export_roots?: SourceExportRoot[];
}

export interface SourcePlan {
  schema: 1;
  include_local_chats: false;
  sources: PlannedSource[];
}

export interface SourceRegistryEntry {
  sourceId: string;
  mode: SourceMode;
  root: string;
  required: boolean;
  exportRoots: SourceExportRoot[];
  representedPaths?: string[];
}

export interface SourceRegistry {
  wikiRoot: string;
  wikiCwd: string;
  backupRoot: string;
  collapseRatio: number;
  outerTimeoutMs: number;
  sources: SourceRegistryEntry[];
}

export interface MakeCurrentReceipt {
  schema: 1;
  run_id: string;
  state: "published" | "failed" | "overlap";
  health: "healthy" | "degraded" | "failed";
  published: boolean;
  generation_id: string | null;
  started_at: string;
  finished_at: string;
  duration_ms: number;
  sources: SourceReceipt[];
  collection_counts: GenerationManifest["collection_counts"] | null;
  collapse_ratio: number;
  error: string | null;
  warning: string | null;
}

export interface MakeCurrentExecution {
  exitCode: 0 | 1 | 75;
  receipt: MakeCurrentReceipt;
}

export interface LockHandle {
  release(): Promise<void>;
}

export interface HistoryGenerationIdentity {
  databaseSha256: string;
  catalogSha256: string;
  databaseFilename: string;
  catalogFilename: string;
}

export interface MakeCurrentDependencies {
  home: string;
  now(): Date;
  randomId(): string;
  resolveRoots(wikiPath?: string): Promise<{ wikiRoot: string; chatsReadableRoot: string }>;
  readConfig(path: string): Promise<string>;
  runCommand(
    args: string[],
    options: { cwd?: string; timeoutMs: number },
  ): Promise<CommandResult>;
  scan(
    roots: string[],
    options?: { include?: (path: string) => boolean },
  ): Promise<{ count: number; newest: string | null }>;
  acquireLock(root: string, runId: string, startedAt: string): Promise<LockHandle | null>;
  acquireArchiveLock(home: string): Promise<LockHandle | null>;
  validateHistoryGeneration(root: string): Promise<HistoryGenerationIdentity>;
  readJson(path: string): Promise<unknown | null>;
  writeJsonAtomic(path: string, value: unknown): Promise<void>;
  makeTempDirectory(prefix: string): Promise<string>;
  removeLocal(path: string): Promise<void>;
  refresh(
    client: AgentKbClient,
    options: {
      wikiPath: string;
      chatsRoot: string;
      sourcePlan: SourcePlan;
      validateBeforeStaging: (manifest: GenerationManifest) => void;
    },
    dependencies: RefreshDependencies,
  ): Promise<RefreshResult>;
  refreshDependencies: RefreshDependencies;
}

function iso(date: Date): string {
  return date.toISOString();
}

function expandHome(path: string, home: string): string {
  if (path === "~") return home;
  return path.startsWith("~/") ? join(home, path.slice(2)) : path;
}

function sourceOverride(
  config: Record<string, unknown>,
  sourceId: string,
  fallback: string,
  home: string,
): string {
  const sourcePaths = config.source_paths;
  if (
    sourcePaths !== undefined &&
    (typeof sourcePaths !== "object" ||
      sourcePaths === null ||
      Array.isArray(sourcePaths))
  ) {
    throw new TypeError("source_paths must be an object");
  }
  const value = (sourcePaths as Record<string, unknown> | undefined)?.[sourceId];
  if (value !== undefined && (typeof value !== "string" || !value)) {
    throw new TypeError(`source_paths.${sourceId} must be a non-empty string`);
  }
  return resolve(expandHome((value as string | undefined) ?? fallback, home));
}

export async function resolveSourceRegistry(
  wikiOverride: string | undefined,
  dependencies: Pick<
    MakeCurrentDependencies,
    "home" | "readConfig" | "resolveRoots"
  >,
): Promise<SourceRegistry> {
  const configPath = join(dependencies.home, ".agentkb", "config.json");
  let config: Record<string, unknown> = {};
  try {
    const value: unknown = JSON.parse(await dependencies.readConfig(configPath));
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      throw new TypeError("~/.agentkb/config.json must contain a JSON object");
    }
    config = value as Record<string, unknown>;
  } catch (error) {
    if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) {
      throw error;
    }
  }
  const roots = await dependencies.resolveRoots(wikiOverride);
  const wikiRoot = roots.wikiRoot;
  const configuredWikiCwd = config.wiki_cwd;
  if (
    configuredWikiCwd !== undefined &&
    (typeof configuredWikiCwd !== "string" || !configuredWikiCwd)
  ) {
    throw new TypeError("wiki_cwd must be a non-empty string");
  }
  const wikiCwd = resolve(
    expandHome(
      (configuredWikiCwd as string | undefined) ??
        join(dependencies.home, "home", "llm-wiki-v1"),
      dependencies.home,
    ),
  );
  const policy = config.make_current;
  if (
    policy !== undefined &&
    (typeof policy !== "object" || policy === null || Array.isArray(policy))
  ) {
    throw new TypeError("make_current must be an object");
  }
  const policyRecord = (policy ?? {}) as Record<string, unknown>;
  const collapseRatio =
    policyRecord.collapse_ratio === undefined
      ? DEFAULT_COLLAPSE_RATIO
      : Number(policyRecord.collapse_ratio);
  if (!Number.isFinite(collapseRatio) || collapseRatio <= 0 || collapseRatio >= 1) {
    throw new TypeError("make_current.collapse_ratio must be between 0 and 1");
  }
  const outerTimeoutMinutes =
    policyRecord.outer_timeout_minutes === undefined
      ? DEFAULT_OUTER_TIMEOUT_MS / 60_000
      : Number(policyRecord.outer_timeout_minutes);
  if (!Number.isFinite(outerTimeoutMinutes) || outerTimeoutMinutes < 300) {
    throw new TypeError("make_current.outer_timeout_minutes must be at least 300");
  }

  const generated = join(dependencies.home, "home", "llm-wiki-generated");
  const readwise = sourceOverride(
    config,
    "readwise-tweets",
    join(generated, "readwise-tweets", "qmd-docs"),
    dependencies.home,
  );
  const youtube = sourceOverride(
    config,
    "youtube-saved",
    join(generated, "youtube-playlists"),
    dependencies.home,
  );
  const historical = sourceOverride(
    config,
    "historical-chat-exports",
    join(generated, "chat-exports", "qmd-docs"),
    dependencies.home,
  );
  const backup = sourceOverride(
    config,
    "agent-history-central",
    join(
      dependencies.home,
      "Library",
      "Application Support",
      "agent-history-backup",
      "mini-admin",
    ),
    dependencies.home,
  );
  return {
    wikiRoot,
    wikiCwd,
    backupRoot: backup,
    collapseRatio,
    outerTimeoutMs: outerTimeoutMinutes * 60_000,
    sources: [
      {
        sourceId: "wiki-pages",
        mode: "projection",
        root: join(wikiRoot, "wiki"),
        required: true,
        exportRoots: [],
      },
      {
        sourceId: "wiki-raw",
        mode: "projection",
        root: join(wikiRoot, "sources"),
        required: true,
        exportRoots: [],
      },
      {
        sourceId: "readwise-tweets",
        mode: "upstream",
        root: readwise,
        required: true,
        exportRoots: [{
          path: readwise,
          collection: "wiki:source",
          prefix: "readwise-tweets/",
          kind: "markdown",
        }],
      },
      {
        sourceId: "github-stars",
        mode: "upstream",
        root: wikiRoot,
        required: true,
        exportRoots: [],
        representedPaths: [
          join(wikiRoot, "wiki", "note-github-starred-repositories.md"),
          join(wikiRoot, "sources", "github-stars", "starred-repos.json"),
        ],
      },
      {
        sourceId: "youtube-saved",
        mode: "human-dependent",
        root: youtube,
        required: true,
        exportRoots: [
          {
            path: join(youtube, "summaries"),
            collection: "wiki:source",
            prefix: "youtube-saved/summaries/",
            kind: "markdown",
          },
          {
            path: youtube,
            collection: "wiki:source",
            prefix: "youtube-saved/",
            kind: "jsonl",
            include: ["watch-history-latest.jsonl", "watch-later-latest.jsonl"],
          },
        ],
      },
      {
        sourceId: "historical-chat-exports",
        mode: "human-dependent",
        root: historical,
        required: true,
        exportRoots: [{
          path: historical,
          collection: "chats",
          prefix: "historical-chat-exports/",
          kind: "markdown",
        }],
      },
      {
        sourceId: "reddit-saved",
        mode: "human-dependent",
        root: wikiRoot,
        required: true,
        exportRoots: [],
        representedPaths: [
          join(wikiRoot, "wiki", "note-reddit-saved-posts.md"),
          join(wikiRoot, "wiki", "playbook-reddit-saved-posts-source-sync.md"),
          join(wikiRoot, "sources", "reddit-saved"),
        ],
      },
      {
        sourceId: "agent-history-central",
        mode: "upstream",
        root: backup,
        required: true,
        exportRoots: [],
      },
    ],
  };
}

export async function runBoundedCommand(
  args: string[],
  options: { cwd?: string; timeoutMs: number },
): Promise<CommandResult> {
  const child = Bun.spawn(args, {
    ...(options.cwd === undefined ? {} : { cwd: options.cwd }),
    stdout: "pipe",
    stderr: "pipe",
  });
  let timedOut = false;
  let forceTimer: ReturnType<typeof setTimeout> | undefined;
  const timer = setTimeout(() => {
    timedOut = true;
    child.kill("SIGTERM");
    forceTimer = setTimeout(() => child.kill("SIGKILL"), 1_000);
  }, options.timeoutMs);
  const [exitCode, stdout, stderr] = await Promise.all([
    child.exited,
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
  ]).finally(() => {
    clearTimeout(timer);
    if (forceTimer !== undefined) clearTimeout(forceTimer);
  });
  return timedOut
    ? {
        exitCode: 124,
        stdout,
        stderr: `${stderr}${stderr && !stderr.endsWith("\n") ? "\n" : ""}timed out after ${options.timeoutMs}ms`,
      }
    : { exitCode, stdout, stderr };
}

interface ArchivePointer {
  schemaVersion: 1;
  archiveSchema: 5;
  catalogSchema: 1;
  database: {
    filename: string;
    sha256: string;
    compressedSha256: string;
    bytes: number;
    compressedBytes: number;
    logicalFingerprint: string;
  };
  catalog: {
    filename: string;
    sha256: string;
    bytes: number;
    recordCount: number;
    fingerprint: string;
  };
  sqliteRuntimeVersion: string;
  referencedBlobCount: number;
  verifiedBlobCount: number;
  verifiedBytes: number;
  knownParserProvenanceCount: number;
  legacyParserProvenanceCount: number;
  integrityCheck: "ok";
  foreignKeyCheck: "ok";
}

function exactKeys(value: Record<string, unknown>, keys: string[]): boolean {
  const expected = [...keys].sort();
  return Object.keys(value).sort().every((key, index, actual) =>
    actual.length === expected.length && key === expected[index]
  );
}

function nonnegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function parseArchivePointer(value: unknown): ArchivePointer {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError("central history current.json must contain an object");
  }
  const item = value as Record<string, unknown>;
  if (!exactKeys(item, [
    "schemaVersion", "archiveSchema", "catalogSchema", "database", "catalog",
    "sqliteRuntimeVersion", "referencedBlobCount", "verifiedBlobCount", "verifiedBytes",
    "knownParserProvenanceCount", "legacyParserProvenanceCount", "integrityCheck",
    "foreignKeyCheck",
  ])) throw new TypeError("central history current.json has unknown or missing fields");
  const database = item.database;
  const catalog = item.catalog;
  if (
    item.schemaVersion !== ARCHIVE_POINTER_SCHEMA ||
    item.archiveSchema !== ARCHIVE_SCHEMA ||
    item.catalogSchema !== 1 ||
    typeof item.sqliteRuntimeVersion !== "string" ||
    !/^\d+\.\d+\.\d+$/.test(item.sqliteRuntimeVersion) ||
    !nonnegativeInteger(item.referencedBlobCount) ||
    !nonnegativeInteger(item.verifiedBlobCount) ||
    item.verifiedBlobCount !== item.referencedBlobCount ||
    !nonnegativeInteger(item.verifiedBytes) ||
    !nonnegativeInteger(item.knownParserProvenanceCount) ||
    !nonnegativeInteger(item.legacyParserProvenanceCount) ||
    item.integrityCheck !== "ok" ||
    item.foreignKeyCheck !== "ok" ||
    typeof database !== "object" || database === null || Array.isArray(database) ||
    typeof catalog !== "object" || catalog === null || Array.isArray(catalog)
  ) throw new TypeError("central history current.json schema is invalid");
  const db = database as Record<string, unknown>;
  const cat = catalog as Record<string, unknown>;
  if (
    !exactKeys(db, ["filename", "sha256", "compressedSha256", "bytes", "compressedBytes", "logicalFingerprint"]) ||
    typeof db.sha256 !== "string" || !SHA256_PATTERN.test(db.sha256) ||
    db.filename !== `history-index-${db.sha256}.sqlite3.zst` ||
    typeof db.compressedSha256 !== "string" || !SHA256_PATTERN.test(db.compressedSha256) ||
    !nonnegativeInteger(db.bytes) || db.bytes === 0 ||
    !nonnegativeInteger(db.compressedBytes) || db.compressedBytes === 0 ||
    typeof db.logicalFingerprint !== "string" || !SHA256_PATTERN.test(db.logicalFingerprint)
  ) throw new TypeError("central history database generation is invalid");
  if (
    !exactKeys(cat, ["filename", "sha256", "bytes", "recordCount", "fingerprint"]) ||
    typeof cat.sha256 !== "string" || !SHA256_PATTERN.test(cat.sha256) ||
    cat.filename !== `provenance-catalog-${cat.sha256}.jsonl` ||
    !nonnegativeInteger(cat.bytes) ||
    !nonnegativeInteger(cat.recordCount) ||
    typeof cat.fingerprint !== "string" || !SHA256_PATTERN.test(cat.fingerprint)
  ) throw new TypeError("central history catalog generation is invalid");
  return item as unknown as ArchivePointer;
}

async function hashFile(path: string): Promise<string> {
  const digest = createHash("sha256");
  await new Promise<void>((resolvePromise, reject) => {
    const stream = createReadStream(path);
    stream.on("data", (chunk) => digest.update(chunk));
    stream.on("error", reject);
    stream.on("end", resolvePromise);
  });
  return digest.digest("hex");
}

async function privateFile(path: string, size?: number): Promise<void> {
  const metadata = await lstat(path);
  if (metadata.isSymbolicLink() || !metadata.isFile() || (metadata.mode & 0o777) !== 0o600) {
    throw new TypeError(`central history private file is unsafe: ${basename(path)}`);
  }
  if (size !== undefined && metadata.size !== size) {
    throw new TypeError(`central history file size mismatch: ${basename(path)}`);
  }
}

export async function validateHistoryGeneration(
  root: string,
): Promise<HistoryGenerationIdentity> {
  const requestedRootMetadata = await lstat(root);
  if (requestedRootMetadata.isSymbolicLink()) {
    throw new TypeError("central history mirror root must not be a symlink");
  }
  const resolvedRoot = resolve(root);
  const rootMetadata = await lstat(resolvedRoot);
  if (
    rootMetadata.isSymbolicLink() ||
    !rootMetadata.isDirectory() ||
    (rootMetadata.mode & 0o777) !== 0o700
  ) throw new TypeError("central history mirror root must be a private 0700 directory");
  const currentPath = join(resolvedRoot, "current.json");
  await privateFile(currentPath);
  const currentMetadata = await lstat(currentPath);
  if (currentMetadata.size <= 0 || currentMetadata.size > 64 * 1024) {
    throw new TypeError("central history current.json size is invalid");
  }
  const pointer = parseArchivePointer(JSON.parse(await readFile(currentPath, "utf-8")));
  const databasePath = resolve(resolvedRoot, pointer.database.filename);
  const catalogPath = resolve(resolvedRoot, pointer.catalog.filename);
  if (
    resolve(databasePath, "..") !== resolvedRoot ||
    resolve(catalogPath, "..") !== resolvedRoot
  ) throw new TypeError("central history generation filename escaped its root");
  await privateFile(databasePath, pointer.database.compressedBytes);
  await privateFile(catalogPath, pointer.catalog.bytes);
  if (await hashFile(databasePath) !== pointer.database.compressedSha256) {
    throw new TypeError("central history compressed database hash mismatch");
  }
  if (await hashFile(catalogPath) !== pointer.catalog.sha256) {
    throw new TypeError("central history catalog hash mismatch");
  }
  const temporary = join(
    tmpdir(),
    `agentkb-history-validate-${process.pid}-${randomBytes(6).toString("hex")}.sqlite3`,
  );
  try {
    const result = await runBoundedCommand(
      ["/opt/homebrew/bin/zstd", "-d", "-q", "-f", databasePath, "-o", temporary],
      { timeoutMs: 10 * 60_000 },
    );
    if (result.exitCode !== 0) {
      throw new Error(`cannot decompress central history database: ${result.stderr.trim()}`);
    }
    await chmod(temporary, 0o600);
    await privateFile(temporary, pointer.database.bytes);
    if (await hashFile(temporary) !== pointer.database.sha256) {
      throw new TypeError("central history uncompressed database hash mismatch");
    }
  } finally {
    await rm(temporary, { force: true });
  }
  return {
    databaseSha256: pointer.database.sha256,
    catalogSha256: pointer.catalog.sha256,
    databaseFilename: pointer.database.filename,
    catalogFilename: pointer.catalog.filename,
  };
}

async function defaultAcquireArchiveLock(home: string): Promise<LockHandle | null> {
  const path = join(
    home,
    "Library",
    "Application Support",
    "agent-history-archive",
    "generation.lock",
  );
  const parent = resolve(path, "..");
  await mkdir(parent, { recursive: true, mode: 0o700 });
  await chmod(parent, 0o700);
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const result = await runBoundedCommand(
      ["/usr/bin/shlock", "-f", path, "-p", String(process.pid)],
      { timeoutMs: 5_000 },
    );
    if (result.exitCode === 0) {
      return {
        release: async () => {
          try {
            if ((await readFile(path, "utf-8")).trim() === String(process.pid)) {
              await rm(path, { force: true });
            }
          } catch (error) {
            if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) {
              throw error;
            }
          }
        },
      };
    }
    if (attempt < 119) await new Promise((resolvePromise) => setTimeout(resolvePromise, 1_000));
  }
  return null;
}

async function defaultScan(
  roots: string[],
  options: { include?: (path: string) => boolean } = {},
): Promise<{ count: number; newest: string | null }> {
  let count = 0;
  let newest = 0;
  const visit = async (path: string): Promise<void> => {
    const metadata = await stat(path);
    if (metadata.isDirectory()) {
      for (const entry of await readdir(path)) {
        if (entry === ".DS_Store") continue;
        await visit(join(path, entry));
      }
      return;
    }
    if (!metadata.isFile() || (options.include && !options.include(path))) return;
    count += 1;
    newest = Math.max(newest, metadata.mtimeMs);
  };
  for (const root of roots) {
    try {
      await visit(root);
    } catch (error) {
      if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) {
        throw error;
      }
    }
  }
  return {
    count,
    newest: newest > 0 ? new Date(newest).toISOString() : null,
  };
}

async function defaultWriteJsonAtomic(path: string, value: unknown): Promise<void> {
  await mkdir(resolve(path, ".."), { recursive: true });
  const temporary = `${path}.${process.pid}.${randomBytes(6).toString("hex")}.tmp`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, {
    encoding: "utf-8",
    mode: 0o600,
  });
  await rename(temporary, path);
}

async function defaultReadJson(path: string): Promise<unknown | null> {
  try {
    return JSON.parse(await readFile(path, "utf-8"));
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") {
      return null;
    }
    throw error;
  }
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return !(
      error instanceof Error &&
      "code" in error &&
      error.code === "ESRCH"
    );
  }
}

async function defaultAcquireLock(
  root: string,
  runId: string,
  startedAt: string,
): Promise<LockHandle | null> {
  const lockPath = join(root, "lock");
  await mkdir(root, { recursive: true });
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      await mkdir(lockPath);
      await defaultWriteJsonAtomic(join(lockPath, "owner.json"), {
        pid: process.pid,
        run_id: runId,
        started_at: startedAt,
      });
      return {
        release: () => rm(lockPath, { recursive: true, force: true }),
      };
    } catch (error) {
      if (!(error instanceof Error && "code" in error && error.code === "EEXIST")) {
        throw error;
      }
      const owner = await defaultReadJson(join(lockPath, "owner.json"));
      if (owner === null) return null;
      const pid =
        typeof owner === "object" &&
        owner !== null &&
        typeof (owner as Record<string, unknown>).pid === "number"
          ? Number((owner as Record<string, unknown>).pid)
          : 0;
      if (pid > 0 && processIsAlive(pid)) return null;
      await rm(lockPath, { recursive: true, force: true });
    }
  }
  return null;
}

export const defaultMakeCurrentDependencies: MakeCurrentDependencies = {
  home: homedir(),
  now: () => new Date(),
  randomId: () => randomBytes(6).toString("hex"),
  resolveRoots: resolvePathRoots,
  readConfig: (path) => readFile(path, "utf-8"),
  runCommand: runBoundedCommand,
  scan: defaultScan,
  acquireLock: defaultAcquireLock,
  acquireArchiveLock: defaultAcquireArchiveLock,
  validateHistoryGeneration,
  readJson: defaultReadJson,
  writeJsonAtomic: defaultWriteJsonAtomic,
  makeTempDirectory: async (prefix) => {
    const directory = join(
      tmpdir(),
      `${prefix}${process.pid}-${randomBytes(6).toString("hex")}`,
    );
    await mkdir(directory);
    return directory;
  },
  removeLocal: (path) => rm(path, { recursive: true, force: true }),
  refresh: refreshProduction,
  refreshDependencies: defaultRefreshDependencies,
};

function newReceipt(
  dependencies: MakeCurrentDependencies,
  runId: string,
  started: Date,
  state: MakeCurrentReceipt["state"],
  error: string | null,
): MakeCurrentReceipt {
  const finished = dependencies.now();
  return {
    schema: RECEIPT_SCHEMA,
    run_id: runId,
    state,
    health: state === "published" ? "healthy" : "failed",
    published: state === "published",
    generation_id: null,
    started_at: iso(started),
    finished_at: iso(finished),
    duration_ms: Math.max(0, finished.getTime() - started.getTime()),
    sources: [],
    collection_counts: null,
    collapse_ratio: DEFAULT_COLLAPSE_RATIO,
    error,
    warning: null,
  };
}

function commandError(args: string[], result: CommandResult): string {
  const detail = result.stderr.trim() || result.stdout.trim() || `exit ${result.exitCode}`;
  return `${args.join(" ")}: ${detail}`;
}

function receiptFor(
  entry: SourceRegistryEntry,
  state: SourceState,
  operation: string,
  started: Date,
  finished: Date,
  scan: { count: number; newest: string | null },
  warning: string | null = null,
  error: string | null = null,
): PlannedSource {
  return {
    source_id: entry.sourceId,
    mode: entry.mode,
    state,
    operation,
    started_at: iso(started),
    finished_at: iso(finished),
    duration_ms: Math.max(0, finished.getTime() - started.getTime()),
    root: entry.root,
    source_file_count: scan.count,
    exported_document_count: 0,
    newest_source_timestamp: scan.newest,
    freshness_threshold_minutes:
      entry.sourceId === "agent-history-central"
        ? BACKUP_FRESHNESS_MINUTES
        : null,
    age_minutes:
      scan.newest === null
        ? null
        : Math.max(0, (finished.getTime() - Date.parse(scan.newest)) / 60_000),
    warning,
    error,
    export_roots: entry.exportRoots,
  };
}

function previousReceipt(value: unknown): MakeCurrentReceipt | null {
  if (
    typeof value !== "object" ||
    value === null ||
    (value as Record<string, unknown>).state !== "published"
  ) {
    return null;
  }
  return value as MakeCurrentReceipt;
}

export function assertNoLargeCollapse(
  manifest: GenerationManifest,
  previous: MakeCurrentReceipt | null,
  collapseRatio: number,
): void {
  if (!manifest.collection_counts || !manifest.sources) {
    throw new TypeError("manifest lacks collection or source counts");
  }
  for (const collection of ["wiki", "wiki:source", "chats"] as const) {
    if (manifest.collection_counts[collection].documents < 1) {
      throw new Error(`refusing zero collection: ${collection}`);
    }
  }
  if (!previous) return;
  const previousSources = new Map(
    previous.sources.map((source) => [source.source_id, source]),
  );
  for (const source of manifest.sources.items) {
    const old = previousSources.get(source.source_id);
    if (!old || old.source_file_count === 0) continue;
    if (source.source_file_count === 0) {
      throw new Error(`previously nonempty source became zero: ${source.source_id}`);
    }
    if (source.source_file_count < old.source_file_count * collapseRatio) {
      throw new Error(
        `source collapse exceeds threshold: ${source.source_id} ` +
          `${source.source_file_count}/${old.source_file_count}`,
      );
    }
  }
  if (!previous.collection_counts) return;
  for (const collection of ["wiki", "wiki:source", "chats"] as const) {
    const old = previous.collection_counts[collection].documents;
    const current = manifest.collection_counts[collection].documents;
    if (old > 0 && current < old * collapseRatio) {
      throw new Error(
        `collection collapse exceeds threshold: ${collection} ${current}/${old}`,
      );
    }
  }
}

export async function makeCurrent(
  client: AgentKbClient,
  wikiPath: string | undefined,
  dependencies: MakeCurrentDependencies = defaultMakeCurrentDependencies,
): Promise<MakeCurrentExecution> {
  const started = dependencies.now();
  const runId = `r-${iso(started).replace(/[-:.]/g, "")}-${dependencies.randomId()}`;
  const supportRoot = join(
    dependencies.home,
    "Library",
    "Application Support",
    "agentkb-refresh",
  );
  const latestPath = join(supportRoot, "latest.json");
  const successPath = join(supportRoot, "last-success.json");
  const lock = await dependencies.acquireLock(supportRoot, runId, iso(started));
  if (!lock) {
    const receipt = newReceipt(
      dependencies,
      runId,
      started,
      "overlap",
      "another make-current run owns the singleton lock",
    );
    await dependencies.writeJsonAtomic(latestPath, receipt);
    return { exitCode: 75, receipt };
  }

  const sourceReceipts: PlannedSource[] = [];
  let registry: SourceRegistry | null = null;
  let tempRoot: string | null = null;
  let archiveLock: LockHandle | null = null;
  try {
    registry = await resolveSourceRegistry(wikiPath, dependencies);
    const deadline = started.getTime() + registry.outerTimeoutMs;
    const checkDeadline = (): void => {
      if (dependencies.now().getTime() >= deadline) {
        throw new Error(`make-current exceeded ${registry!.outerTimeoutMs}ms`);
      }
    };

    const syncArgs = [
      "agent-history-sync",
      "run",
      "mini-admin-to-air-backup",
      "--json",
    ];
    const statusArgs = [
      "agent-history-sync",
      "status",
      "mini-admin-to-air-backup",
      "--max-age-minutes",
      String(BACKUP_FRESHNESS_MINUTES),
      "--json",
    ];
    const historyEntry = registry.sources.find(
      (source) => source.sourceId === "agent-history-central",
    )!;
    const historyStarted = dependencies.now();
    const syncResult = await dependencies.runCommand(syncArgs, {
      timeoutMs: 20 * 60_000,
    });
    archiveLock = await dependencies.acquireArchiveLock(dependencies.home);
    if (!archiveLock) {
      throw new Error("central history generation remained locked beyond the bounded wait");
    }
    const statusResult = await dependencies.runCommand(statusArgs, {
      timeoutMs: 2 * 60_000,
    });
    const historyGeneration = await dependencies.validateHistoryGeneration(
      registry.backupRoot,
    );
    const historyScan = await dependencies.scan([
      join(registry.backupRoot, "current.json"),
      join(registry.backupRoot, historyGeneration.databaseFilename),
      join(registry.backupRoot, historyGeneration.catalogFilename),
      join(registry.backupRoot, "raw"),
    ]);
    const historyWarning =
      syncResult.exitCode !== 0
        ? commandError(syncArgs, syncResult)
        : statusResult.exitCode !== 0
        ? commandError(statusArgs, statusResult)
        : null;
    const historyError =
      historyScan.count < 4 ? "verified central history generation is empty" : null;
    sourceReceipts.push(
      receiptFor(
        historyEntry,
        historyError ? "failed" : historyWarning ? "fallback" : "fresh",
        `${syncArgs.join(" ")}; ${statusArgs.join(" ")}`,
        historyStarted,
        dependencies.now(),
        historyScan,
        historyWarning,
        historyError,
      ),
    );
    if (historyError) throw new Error(historyError);
    checkDeadline();

    for (const entry of registry.sources) {
      if (entry.sourceId === "agent-history-central") continue;
      const sourceStarted = dependencies.now();
      let operation = `validate ${entry.root}`;
      let commandResult: CommandResult | null = null;
      if (entry.sourceId === "readwise-tweets") {
        const args = [
          "uv",
          "run",
          "python",
          "scripts/readwise-tweets/readwise_tweets.py",
          "fetch",
          "--json",
          "--no-qmd",
        ];
        operation = args.join(" ");
        commandResult = await dependencies.runCommand(args, {
          cwd: registry.wikiCwd,
          timeoutMs: DEFAULT_COMMAND_TIMEOUT_MS,
        });
      } else if (entry.sourceId === "github-stars") {
        const args = [
          "uv",
          "run",
          "python",
          "scripts/update_github_starred_repos.py",
          "--skip-log",
        ];
        operation = args.join(" ");
        commandResult = await dependencies.runCommand(args, {
          cwd: registry.wikiCwd,
          timeoutMs: DEFAULT_COMMAND_TIMEOUT_MS,
        });
      }
      const scanRoots = entry.representedPaths ?? [entry.root];
      const scanned = await dependencies.scan(scanRoots, {
        include: (path) =>
          entry.sourceId !== "youtube-saved" ||
          path.endsWith(".md") ||
          basename(path) === "watch-history-latest.jsonl" ||
          basename(path) === "watch-later-latest.jsonl",
      });
      const commandFailed = commandResult !== null && commandResult.exitCode !== 0;
      const empty = scanned.count === 0;
      const error =
        empty && entry.required
          ? `${entry.sourceId} durable projection is empty`
          : null;
      const warning = commandFailed
        ? commandError(operation.split(" "), commandResult!)
        : null;
      const state: SourceState = error
        ? "failed"
        : commandFailed
        ? "fallback"
        : entry.mode === "human-dependent"
        ? "stale"
        : "fresh";
      sourceReceipts.push(
        receiptFor(
          entry,
          state,
          operation,
          sourceStarted,
          dependencies.now(),
          scanned,
          warning,
          error,
        ),
      );
      if (error) throw new Error(error);
      checkDeadline();
    }

    tempRoot = await dependencies.makeTempDirectory("agentkb-make-current-");
    await mkdir(join(tempRoot, "chats", "readable"), { recursive: true });
    const sourcePlan: SourcePlan = {
      schema: SOURCE_PLAN_SCHEMA,
      include_local_chats: false,
      sources: sourceReceipts,
    };
    const previous = previousReceipt(await dependencies.readJson(successPath));
    const result = await dependencies.refresh(
      client,
      {
        wikiPath: registry.wikiRoot,
        chatsRoot: join(tempRoot, "chats"),
        sourcePlan,
        validateBeforeStaging: (manifest) =>
          assertNoLargeCollapse(manifest, previous, registry!.collapseRatio),
      },
      dependencies.refreshDependencies,
    );
    const finished = dependencies.now();
    const publishedSources = result.manifest.sources?.items ?? [];
    const degraded = publishedSources.some((source) => source.state === "fallback");
    const receipt: MakeCurrentReceipt = {
      schema: RECEIPT_SCHEMA,
      run_id: runId,
      state: "published",
      health: degraded ? "degraded" : "healthy",
      published: true,
      generation_id: result.generation_id,
      started_at: iso(started),
      finished_at: iso(finished),
      duration_ms: Math.max(0, finished.getTime() - started.getTime()),
      sources: publishedSources,
      collection_counts: result.manifest.collection_counts ?? null,
      collapse_ratio: registry.collapseRatio,
      error: null,
      warning: degraded ? "one or more upstream sources used a valid fallback" : null,
    };
    await dependencies.writeJsonAtomic(latestPath, receipt);
    await dependencies.writeJsonAtomic(successPath, receipt);
    return { exitCode: 0, receipt };
  } catch (error) {
    const receipt = newReceipt(
      dependencies,
      runId,
      started,
      "failed",
      error instanceof Error ? error.message : String(error),
    );
    receipt.sources = sourceReceipts;
    receipt.collapse_ratio = registry?.collapseRatio ?? DEFAULT_COLLAPSE_RATIO;
    await dependencies.writeJsonAtomic(latestPath, receipt);
    return { exitCode: 1, receipt };
  } finally {
    if (tempRoot) await dependencies.removeLocal(tempRoot);
    if (archiveLock) await archiveLock.release();
    await lock.release();
  }
}
