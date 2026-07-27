export const APP_NAME = "agentkb";
export const DEFAULT_MODEL = "lightonai/GTE-ModernColBERT-v1";
export const BUILD_DOCUMENT_BATCH_SIZE = 256;
export const BUILD_PLAID_KMEANS_SAMPLE_SIZE = 16_384;
export const BUILD_PLAID_PERMUTATION_ALGORITHM = "sha256-key-sort-v1";
export const IMMUTABLE_FAST_PLAID_VERSION = "1.3.0.290";
export const GENERATION_ID_PATTERN =
  /^g-\d{8}T\d{6}Z-[0-9a-f]{12}$/;
export const SHA256_PATTERN = /^[0-9a-f]{64}$/;

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue =
  | JsonPrimitive
  | JsonValue[]
  | { [key: string]: JsonValue };

export interface GenerationManifest {
  schema: 1;
  generation_id: string;
  model: string;
  corpus_count: number;
  corpus_hash: string;
  source_file_counts?: Record<CorpusCollection, number>;
  collection_counts?: Record<CorpusCollection, CollectionCount>;
  sources?: SourcesManifest;
  exported_at?: string;
  build?: BuildMetrics;
  validation?: IndexValidation;
}

export type CorpusCollection = "wiki" | "wiki:source" | "chats";

export interface CollectionCount {
  documents: number;
  files: number;
}

export type SourceMode =
  | "upstream"
  | "projection"
  | "human-dependent"
  | "disabled-costly";

export type SourceState = "fresh" | "fallback" | "stale" | "failed";

export interface SourceReceipt {
  source_id: string;
  mode: SourceMode;
  state: SourceState;
  operation: string;
  started_at: string;
  finished_at: string;
  duration_ms: number;
  root: string;
  source_file_count: number;
  exported_document_count: number;
  newest_source_timestamp: string | null;
  freshness_threshold_minutes: number | null;
  age_minutes: number | null;
  warning: string | null;
  error: string | null;
}

export interface SourcesManifest {
  schema: 1;
  items: SourceReceipt[];
}

export interface IndexValidation {
  sqlite_count: number;
  fts_count: number;
  plaid_mapping_count: number;
  plaid_reverse_mapping_count: number;
  index_tree_hash: string;
  immutable_premerged?: ImmutablePremergedCertificate;
}

export interface ImmutableArtifact {
  size_bytes: number;
  dtype?: string;
  shape?: number[];
}

export interface ImmutablePremergedCertificate {
  schema: 1;
  fast_plaid_version: string;
  num_chunks: number;
  nbits: number;
  document_count: number;
  padding_rows: number;
  artifacts: Record<string, ImmutableArtifact>;
}

export interface StatusResponse {
  schema: 1;
  current_generation_id: string | null;
  previous_generation_id: string | null;
  published_at: string | null;
  current_manifest: GenerationManifest | null;
  previous_manifest: GenerationManifest | null;
}

export interface WarmResult {
  schema: 1;
  generation_id: string;
  model: string;
  corpus_count: number;
  startup_timing_ms: StartupTiming;
  ready: true;
}

export interface StartupTiming {
  artifact_mount: number;
  certificate: number;
  model: number;
  index_load: number;
  total: number;
}

export interface SearchRequest {
  query: string;
  k: number;
}

export interface SearchHit {
  collection: string;
  file: string;
  path: string;
  filename: string;
  line: number;
  score: number;
  relative_path: string;
  name?: string;
  unit_type?: string;
  title?: string;
  section?: string;
  tags?: string[];
  content?: string;
}

export interface SearchResult {
  schema: 1;
  generation_id: string;
  query: string;
  k: number;
  results: SearchHit[];
}

export interface BuildResult {
  schema: 1;
  generation_id: string;
  previous_generation_id: string | null;
  model: string;
  corpus_count: number;
  corpus_hash: string;
  embedding_dimension: number;
  staged_embedding_bytes: number;
  document_batch_size: number;
  document_batch_count: number;
  plaid_create_count: 1;
  plaid_kmeans_sample_size: number;
  plaid_permutation_algorithm: string;
  validation: IndexValidation;
  duration_ms: number;
}

export interface PrunePreviousResult {
  schema: 1;
  dry_run: boolean;
  deleted: boolean;
  target_generation_id: string;
  current_generation_id: string;
  previous_generation_id: string;
  final_previous_generation_id: string | null;
}

export type GenerationClassification = "current" | "previous" | "orphan" | "staged";
export type GenerationTargetType = "generation" | "staged";

export interface GenerationInventoryItem {
  generation_id: string;
  type: GenerationTargetType;
  classification: GenerationClassification;
}

export interface GenerationInventory {
  schema: 1;
  current_generation_id: string | null;
  previous_generation_id: string | null;
  items: GenerationInventoryItem[];
  counts: Record<GenerationClassification, number>;
}

export interface SessionPresenceResult {
  schema: 1;
  source: "claude" | "codex";
  session_id: string;
  canonical_file: string;
  results: Array<GenerationInventoryItem & {
    exact_match_count: number;
    verified: boolean;
    scanned_record_count?: number;
  }>;
  total_exact_match_count: number;
  verification_failures: Array<{
    generation_id: string;
    type: GenerationTargetType;
    classification: GenerationClassification;
    error: string;
  }>;
  verified: boolean;
}

export interface DeleteGenerationResult {
  schema: 1;
  dry_run: boolean;
  deleted: boolean;
  idempotent: boolean;
  target_id: string;
  target_type: GenerationTargetType;
  classification: Exclude<GenerationClassification, "current">;
  current_generation_id: string;
  operation_id: string | null;
  receipt: Record<string, JsonValue> | null;
}

export interface BuildMetrics {
  document_batch_size: number;
  document_batch_count: number;
  embedding_dimension: number;
  staged_embedding_bytes: number;
  plaid_create_count: 1;
  plaid_kmeans_sample_size: number;
  plaid_permutation_algorithm: string;
}

function fail(path: string, expectation: string): never {
  throw new TypeError(`${path} ${expectation}`);
}

function objectAt(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail(path, "must be an object");
  }
  return value as Record<string, unknown>;
}

function stringAt(value: unknown, path: string): string {
  if (typeof value !== "string") fail(path, "must be a string");
  return value;
}

function nullableStringAt(value: unknown, path: string): string | null {
  return value === null ? null : stringAt(value, path);
}

function finiteNumberAt(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    fail(path, "must be a finite number");
  }
  return value;
}

function integerAt(
  value: unknown,
  path: string,
  minimum = 0,
  maximum = Number.MAX_SAFE_INTEGER,
): number {
  const parsed = finiteNumberAt(value, path);
  if (!Number.isInteger(parsed) || parsed < minimum || parsed > maximum) {
    fail(path, `must be an integer between ${minimum} and ${maximum}`);
  }
  return parsed;
}

function schemaAt(value: unknown, path: string): 1 {
  if (value !== 1) fail(path, "must equal 1");
  return 1;
}

function booleanAt(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") fail(path, "must be a boolean");
  return value;
}

export function validateGenerationId(value: unknown): string {
  const generationId = stringAt(value, "generation_id");
  if (!GENERATION_ID_PATTERN.test(generationId)) {
    fail(
      "generation_id",
      "must match g-YYYYMMDDTHHMMSSZ-<12 lowercase hex>",
    );
  }
  return generationId;
}

export function validateSessionSource(value: unknown): "claude" | "codex" {
  if (value !== "claude" && value !== "codex") {
    fail("source", "must be exactly claude or codex");
  }
  return value;
}

export function validateSessionId(value: unknown): string {
  const sessionId = stringAt(value, "session_id");
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$/.test(sessionId)) {
    fail("session_id", "must be a safe canonical identifier");
  }
  return sessionId;
}

export function generationPaths(generationId: unknown): {
  stagedCorpus: string;
  stagedManifest: string;
  generationIndex: string;
  generationManifest: string;
} {
  const id = validateGenerationId(generationId);
  return {
    stagedCorpus: `staged/${id}/corpus.jsonl`,
    stagedManifest: `staged/${id}/manifest.json`,
    generationIndex: `generations/${id}/index`,
    generationManifest: `generations/${id}/manifest.json`,
  };
}

export function createGenerationId(
  now = new Date(),
  randomBytes: Uint8Array = crypto.getRandomValues(new Uint8Array(6)),
): string {
  if (Number.isNaN(now.getTime())) throw new TypeError("now must be a valid Date");
  if (randomBytes.byteLength !== 6) {
    throw new TypeError("randomBytes must contain exactly 6 bytes");
  }
  const timestamp = now
    .toISOString()
    .replace(/[-:]/g, "")
    .replace(/\.\d{3}Z$/, "Z");
  const suffix = [...randomBytes]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
  return validateGenerationId(`g-${timestamp}-${suffix}`);
}

function validateIndexValidation(
  value: unknown,
  expectedCorpusCount: number,
  path = "validation",
  requireImmutablePremerged = true,
): IndexValidation {
  const record = objectAt(value, path);
  const indexTreeHash = stringAt(
    record.index_tree_hash,
    `${path}.index_tree_hash`,
  );
  if (!SHA256_PATTERN.test(indexTreeHash)) {
    fail(`${path}.index_tree_hash`, "must be lowercase SHA-256");
  }
  const result: IndexValidation = {
    sqlite_count: integerAt(record.sqlite_count, `${path}.sqlite_count`),
    fts_count: integerAt(record.fts_count, `${path}.fts_count`),
    plaid_mapping_count: integerAt(
      record.plaid_mapping_count,
      `${path}.plaid_mapping_count`,
    ),
    plaid_reverse_mapping_count: integerAt(
      record.plaid_reverse_mapping_count,
      `${path}.plaid_reverse_mapping_count`,
    ),
    index_tree_hash: indexTreeHash,
  };
  if (
    record.immutable_premerged === undefined &&
    !requireImmutablePremerged
  ) {
    return result;
  }
  const immutableRecord = objectAt(
    record.immutable_premerged,
    `${path}.immutable_premerged`,
  );
  const numChunks = integerAt(
    immutableRecord.num_chunks,
    `${path}.immutable_premerged.num_chunks`,
    1,
  );
  const documentCount = integerAt(
    immutableRecord.document_count,
    `${path}.immutable_premerged.document_count`,
    1,
  );
  if (documentCount !== expectedCorpusCount) {
    fail(
      `${path}.immutable_premerged.document_count`,
      "must match corpus_count",
    );
  }
  const artifactRecord = objectAt(
    immutableRecord.artifacts,
    `${path}.immutable_premerged.artifacts`,
  );
  const requiredArtifacts = [
    "metadata.json",
    "centroids.npy",
    "avg_residual.npy",
    "bucket_cutoffs.npy",
    "bucket_weights.npy",
    "ivf.npy",
    "ivf_lengths.npy",
    "merged_codes.npy",
    "merged_residuals.npy",
    ...Array.from({ length: numChunks }, (_, index) => `doclens.${index}.json`),
  ];
  const artifacts = Object.fromEntries(
    requiredArtifacts.map((name) => {
      const artifactPath = `${path}.immutable_premerged.artifacts.${name}`;
      const artifact = objectAt(artifactRecord[name], artifactPath);
      const parsed: ImmutableArtifact = {
        size_bytes: integerAt(artifact.size_bytes, `${artifactPath}.size_bytes`, 1),
      };
      if (name.endsWith(".npy")) {
        parsed.dtype = stringAt(artifact.dtype, `${artifactPath}.dtype`);
        if (!Array.isArray(artifact.shape) || artifact.shape.length < 1) {
          fail(`${artifactPath}.shape`, "must be a non-empty array");
        }
        parsed.shape = artifact.shape.map((dimension, index) =>
          integerAt(dimension, `${artifactPath}.shape[${index}]`, 1)
        );
      }
      return [name, parsed];
    }),
  );
  const fastPlaidVersion = stringAt(
    immutableRecord.fast_plaid_version,
    `${path}.immutable_premerged.fast_plaid_version`,
  );
  if (fastPlaidVersion !== IMMUTABLE_FAST_PLAID_VERSION) {
    fail(
      `${path}.immutable_premerged.fast_plaid_version`,
      `must equal ${IMMUTABLE_FAST_PLAID_VERSION}`,
    );
  }
  result.immutable_premerged = {
    schema: schemaAt(
      immutableRecord.schema,
      `${path}.immutable_premerged.schema`,
    ),
    fast_plaid_version: fastPlaidVersion,
    num_chunks: numChunks,
    nbits: integerAt(
      immutableRecord.nbits,
      `${path}.immutable_premerged.nbits`,
      1,
    ),
    document_count: documentCount,
    padding_rows: integerAt(
      immutableRecord.padding_rows,
      `${path}.immutable_premerged.padding_rows`,
    ),
    artifacts,
  };
  return result;
}

function validateBuildMetrics(
  value: unknown,
  corpusCount: number,
  path: string,
): BuildMetrics {
  const record = objectAt(value, path);
  const documentBatchSize = integerAt(
    record.document_batch_size,
    `${path}.document_batch_size`,
    1,
  );
  if (documentBatchSize !== BUILD_DOCUMENT_BATCH_SIZE) {
    fail(
      `${path}.document_batch_size`,
      `must equal ${BUILD_DOCUMENT_BATCH_SIZE}`,
    );
  }
  const documentBatchCount = integerAt(
    record.document_batch_count,
    `${path}.document_batch_count`,
    1,
  );
  if (documentBatchCount !== Math.ceil(corpusCount / documentBatchSize)) {
    fail(`${path}.document_batch_count`, "does not cover corpus_count exactly");
  }
  const plaidCreateCount = integerAt(
    record.plaid_create_count,
    `${path}.plaid_create_count`,
    1,
  );
  if (plaidCreateCount !== 1) {
    fail(`${path}.plaid_create_count`, "must equal 1");
  }
  const plaidKmeansSampleSize = integerAt(
    record.plaid_kmeans_sample_size,
    `${path}.plaid_kmeans_sample_size`,
    16_384,
    32_768,
  );
  if (plaidKmeansSampleSize !== BUILD_PLAID_KMEANS_SAMPLE_SIZE) {
    fail(
      `${path}.plaid_kmeans_sample_size`,
      `must equal ${BUILD_PLAID_KMEANS_SAMPLE_SIZE}`,
    );
  }
  const permutation = stringAt(
    record.plaid_permutation_algorithm,
    `${path}.plaid_permutation_algorithm`,
  );
  if (permutation !== BUILD_PLAID_PERMUTATION_ALGORITHM) {
    fail(
      `${path}.plaid_permutation_algorithm`,
      `must equal ${BUILD_PLAID_PERMUTATION_ALGORITHM}`,
    );
  }
  return {
    document_batch_size: documentBatchSize,
    document_batch_count: documentBatchCount,
    embedding_dimension: integerAt(
      record.embedding_dimension,
      `${path}.embedding_dimension`,
      1,
    ),
    staged_embedding_bytes: integerAt(
      record.staged_embedding_bytes,
      `${path}.staged_embedding_bytes`,
      1,
    ),
    plaid_create_count: 1,
    plaid_kmeans_sample_size: plaidKmeansSampleSize,
    plaid_permutation_algorithm: permutation,
  };
}

function nullableNumberAt(value: unknown, path: string): number | null {
  if (value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    fail(path, "must be null or a non-negative finite number");
  }
  return value;
}

export function validateSourcesManifest(
  value: unknown,
  path = "sources",
): SourcesManifest {
  const record = objectAt(value, path);
  if (!Array.isArray(record.items)) fail(`${path}.items`, "must be an array");
  const ids = new Set<string>();
  const modes = new Set<SourceMode>([
    "upstream",
    "projection",
    "human-dependent",
    "disabled-costly",
  ]);
  const states = new Set<SourceState>([
    "fresh",
    "fallback",
    "stale",
    "failed",
  ]);
  const items = record.items.map((item, index): SourceReceipt => {
    const itemPath = `${path}.items[${index}]`;
    const source = objectAt(item, itemPath);
    const sourceId = stringAt(source.source_id, `${itemPath}.source_id`);
    if (!sourceId || ids.has(sourceId)) {
      fail(`${itemPath}.source_id`, "must be non-empty and unique");
    }
    ids.add(sourceId);
    const mode = stringAt(source.mode, `${itemPath}.mode`) as SourceMode;
    const state = stringAt(source.state, `${itemPath}.state`) as SourceState;
    if (!modes.has(mode)) fail(`${itemPath}.mode`, "is unsupported");
    if (!states.has(state)) fail(`${itemPath}.state`, "is unsupported");
    return {
      source_id: sourceId,
      mode,
      state,
      operation: stringAt(source.operation, `${itemPath}.operation`),
      started_at: stringAt(source.started_at, `${itemPath}.started_at`),
      finished_at: stringAt(source.finished_at, `${itemPath}.finished_at`),
      duration_ms: integerAt(source.duration_ms, `${itemPath}.duration_ms`),
      root: stringAt(source.root, `${itemPath}.root`),
      source_file_count: integerAt(
        source.source_file_count,
        `${itemPath}.source_file_count`,
      ),
      exported_document_count: integerAt(
        source.exported_document_count,
        `${itemPath}.exported_document_count`,
      ),
      newest_source_timestamp: nullableStringAt(
        source.newest_source_timestamp,
        `${itemPath}.newest_source_timestamp`,
      ),
      freshness_threshold_minutes: nullableNumberAt(
        source.freshness_threshold_minutes,
        `${itemPath}.freshness_threshold_minutes`,
      ),
      age_minutes: nullableNumberAt(
        source.age_minutes,
        `${itemPath}.age_minutes`,
      ),
      warning: nullableStringAt(source.warning, `${itemPath}.warning`),
      error: nullableStringAt(source.error, `${itemPath}.error`),
    };
  });
  return {
    schema: schemaAt(record.schema, `${path}.schema`),
    items,
  };
}

export function validateManifest(
  value: unknown,
  path = "manifest",
  requireImmutablePremerged = true,
): GenerationManifest {
  const record = objectAt(value, path);
  const corpusHash = stringAt(record.corpus_hash, `${path}.corpus_hash`);
  if (!SHA256_PATTERN.test(corpusHash)) {
    fail(`${path}.corpus_hash`, "must be lowercase SHA-256");
  }
  const result: GenerationManifest = {
    schema: schemaAt(record.schema, `${path}.schema`),
    generation_id: validateGenerationId(record.generation_id),
    model: stringAt(record.model, `${path}.model`),
    corpus_count: integerAt(record.corpus_count, `${path}.corpus_count`, 1),
    corpus_hash: corpusHash,
  };
  if (record.source_file_counts !== undefined) {
    const sourceCounts = objectAt(
      record.source_file_counts,
      `${path}.source_file_counts`,
    );
    result.source_file_counts = {
      wiki: integerAt(sourceCounts.wiki, `${path}.source_file_counts.wiki`),
      "wiki:source": integerAt(
        sourceCounts["wiki:source"],
        `${path}.source_file_counts.wiki:source`,
      ),
      chats: integerAt(sourceCounts.chats, `${path}.source_file_counts.chats`),
    };
  }
  if (record.collection_counts !== undefined) {
    const counts = objectAt(record.collection_counts, `${path}.collection_counts`);
    result.collection_counts = Object.fromEntries(
      (["wiki", "wiki:source", "chats"] as CorpusCollection[]).map(
        (collection) => {
          const count = objectAt(
            counts[collection],
            `${path}.collection_counts.${collection}`,
          );
          return [
            collection,
            {
              documents: integerAt(
                count.documents,
                `${path}.collection_counts.${collection}.documents`,
              ),
              files: integerAt(
                count.files,
                `${path}.collection_counts.${collection}.files`,
              ),
            },
          ];
        },
      ),
    ) as Record<CorpusCollection, CollectionCount>;
  }
  if (record.sources !== undefined) {
    result.sources = validateSourcesManifest(record.sources, `${path}.sources`);
  }
  if (record.exported_at !== undefined) {
    result.exported_at = stringAt(record.exported_at, `${path}.exported_at`);
  }
  if (record.validation !== undefined) {
    result.validation = validateIndexValidation(
      record.validation,
      result.corpus_count,
      `${path}.validation`,
      requireImmutablePremerged,
    );
  }
  if (record.build !== undefined) {
    result.build = validateBuildMetrics(
      record.build,
      result.corpus_count,
      `${path}.build`,
    );
  }
  return result;
}

export function validateStatus(value: unknown): StatusResponse {
  const record = objectAt(value, "status");
  const current =
    record.current_generation_id === null
      ? null
      : validateGenerationId(record.current_generation_id);
  const previous =
    record.previous_generation_id === null
      ? null
      : validateGenerationId(record.previous_generation_id);
  const currentManifest =
    record.current_manifest === null
      ? null
      : validateManifest(record.current_manifest, "status.current_manifest");
  const previousManifest =
    record.previous_manifest === null
      ? null
      : validateManifest(
          record.previous_manifest,
          "status.previous_manifest",
          false,
        );
  if ((current === null) !== (currentManifest === null)) {
    fail("status.current_manifest", "must agree with current_generation_id");
  }
  if ((previous === null) !== (previousManifest === null)) {
    fail("status.previous_manifest", "must agree with previous_generation_id");
  }
  if (currentManifest && currentManifest.generation_id !== current) {
    fail("status.current_manifest.generation_id", "must match current_generation_id");
  }
  if (previousManifest && previousManifest.generation_id !== previous) {
    fail(
      "status.previous_manifest.generation_id",
      "must match previous_generation_id",
    );
  }
  const publishedAt = nullableStringAt(
    record.published_at,
    "status.published_at",
  );
  if ((current === null) !== (publishedAt === null)) {
    fail("status.published_at", "must be null exactly when current is null");
  }
  if (publishedAt !== null && Number.isNaN(Date.parse(publishedAt))) {
    fail("status.published_at", "must be an ISO date-time");
  }
  return {
    schema: schemaAt(record.schema, "status.schema"),
    current_generation_id: current,
    previous_generation_id: previous,
    published_at: publishedAt,
    current_manifest: currentManifest,
    previous_manifest: previousManifest,
  };
}

export function validateSearchRequest(value: unknown): SearchRequest {
  const record = objectAt(value, "search request");
  const query = stringAt(record.query, "search request.query");
  if (!query.trim()) fail("search request.query", "must not be empty");
  return {
    query,
    k: integerAt(record.k, "search request.k", 1, 100),
  };
}

function optionalString(
  record: Record<string, unknown>,
  key: string,
  path: string,
): string | undefined {
  return record[key] === undefined
    ? undefined
    : stringAt(record[key], `${path}.${key}`);
}

function validateSearchHit(value: unknown, index: number): SearchHit {
  const path = `search.results[${index}]`;
  const record = objectAt(value, path);
  const hit: SearchHit = {
    collection: stringAt(record.collection, `${path}.collection`),
    file: stringAt(record.file, `${path}.file`),
    path: stringAt(record.path, `${path}.path`),
    filename: stringAt(record.filename, `${path}.filename`),
    line: integerAt(record.line, `${path}.line`, 1),
    score: finiteNumberAt(record.score, `${path}.score`),
    relative_path: stringAt(record.relative_path, `${path}.relative_path`),
  };
  for (const key of [
    "name",
    "unit_type",
    "title",
    "section",
    "content",
  ] as const) {
    const parsed = optionalString(record, key, path);
    if (parsed !== undefined) hit[key] = parsed;
  }
  if (record.tags !== undefined) {
    if (!Array.isArray(record.tags)) fail(`${path}.tags`, "must be an array");
    hit.tags = record.tags.map((tag, tagIndex) =>
      stringAt(tag, `${path}.tags[${tagIndex}]`),
    );
  }
  return hit;
}

export function validateSearchResult(value: unknown): SearchResult {
  const record = objectAt(value, "search");
  if (!Array.isArray(record.results)) fail("search.results", "must be an array");
  return {
    schema: schemaAt(record.schema, "search.schema"),
    generation_id: validateGenerationId(record.generation_id),
    query: stringAt(record.query, "search.query"),
    k: integerAt(record.k, "search.k", 1, 100),
    results: record.results.map(validateSearchHit),
  };
}

export function validateWarmResult(value: unknown): WarmResult {
  const record = objectAt(value, "warm");
  if (record.ready !== true) fail("warm.ready", "must equal true");
  const timingRecord = objectAt(
    record.startup_timing_ms,
    "warm.startup_timing_ms",
  );
  const timing = Object.fromEntries(
    ["artifact_mount", "certificate", "model", "index_load", "total"].map((field) => {
      const value = finiteNumberAt(
        timingRecord[field],
        `warm.startup_timing_ms.${field}`,
      );
      if (value < 0) {
        fail(`warm.startup_timing_ms.${field}`, "must be non-negative");
      }
      return [field, value];
    }),
  ) as unknown as StartupTiming;
  return {
    schema: schemaAt(record.schema, "warm.schema"),
    generation_id: validateGenerationId(record.generation_id),
    model: stringAt(record.model, "warm.model"),
    corpus_count: integerAt(record.corpus_count, "warm.corpus_count", 1),
    startup_timing_ms: timing,
    ready: true,
  };
}

export function validateBuildResult(value: unknown): BuildResult {
  const record = objectAt(value, "build");
  const corpusHash = stringAt(record.corpus_hash, "build.corpus_hash");
  if (!SHA256_PATTERN.test(corpusHash)) {
    fail("build.corpus_hash", "must be lowercase SHA-256");
  }
  const corpusCount = integerAt(record.corpus_count, "build.corpus_count", 1);
  const buildMetrics = validateBuildMetrics(record, corpusCount, "build");
  return {
    schema: schemaAt(record.schema, "build.schema"),
    generation_id: validateGenerationId(record.generation_id),
    previous_generation_id:
      record.previous_generation_id === null
        ? null
        : validateGenerationId(record.previous_generation_id),
    model: stringAt(record.model, "build.model"),
    corpus_count: corpusCount,
    corpus_hash: corpusHash,
    ...buildMetrics,
    validation: validateIndexValidation(record.validation, corpusCount),
    duration_ms: finiteNumberAt(record.duration_ms, "build.duration_ms"),
  };
}

export function validatePrunePreviousResult(
  value: unknown,
): PrunePreviousResult {
  const record = objectAt(value, "prune_previous");
  const dryRun = booleanAt(record.dry_run, "prune_previous.dry_run");
  const deleted = booleanAt(record.deleted, "prune_previous.deleted");
  const target = validateGenerationId(record.target_generation_id);
  const current = validateGenerationId(record.current_generation_id);
  const previous = validateGenerationId(record.previous_generation_id);
  const finalPrevious =
    record.final_previous_generation_id === null
      ? null
      : validateGenerationId(record.final_previous_generation_id);
  if (target === current) {
    fail("prune_previous.target_generation_id", "must not equal current_generation_id");
  }
  if (target !== previous) {
    fail("prune_previous.target_generation_id", "must equal previous_generation_id");
  }
  if (dryRun === deleted) {
    fail(
      "prune_previous.deleted",
      "must be false for dry runs and true for real prunes",
    );
  }
  if (dryRun && finalPrevious !== previous) {
    fail("prune_previous.final_previous_generation_id", "must remain previous on dry run");
  }
  if (deleted && finalPrevious === target) {
    fail("prune_previous.final_previous_generation_id", "must not reference the deleted target");
  }
  return {
    schema: schemaAt(record.schema, "prune_previous.schema"),
    dry_run: dryRun,
    deleted,
    target_generation_id: target,
    current_generation_id: current,
    previous_generation_id: previous,
    final_previous_generation_id: finalPrevious,
  };
}

function generationTargetTypeAt(
  value: unknown,
  path: string,
): GenerationTargetType {
  if (value !== "generation" && value !== "staged") {
    fail(path, "must be generation or staged");
  }
  return value;
}

function generationClassificationAt(
  value: unknown,
  path: string,
): GenerationClassification {
  if (
    value !== "current" &&
    value !== "previous" &&
    value !== "orphan" &&
    value !== "staged"
  ) {
    fail(path, "has an unsupported classification");
  }
  return value;
}

function validateInventoryItem(
  value: unknown,
  path: string,
): GenerationInventoryItem {
  const record = objectAt(value, path);
  const type = generationTargetTypeAt(record.type, `${path}.type`);
  const classification = generationClassificationAt(
    record.classification,
    `${path}.classification`,
  );
  if ((type === "staged") !== (classification === "staged")) {
    fail(`${path}.classification`, "must agree with target type");
  }
  return {
    generation_id: validateGenerationId(record.generation_id),
    type,
    classification,
  };
}

export function validateGenerationInventory(value: unknown): GenerationInventory {
  const record = objectAt(value, "generations");
  if (!Array.isArray(record.items)) fail("generations.items", "must be an array");
  if (record.items.length > 1_000) {
    fail("generations.items", "must contain at most 1000 entries");
  }
  const items = record.items.map((item, index) =>
    validateInventoryItem(item, `generations.items[${index}]`)
  );
  const ids = new Set(items.map((item) => item.generation_id));
  if (ids.size !== items.length) fail("generations.items", "must have unique IDs");
  const countsRecord = objectAt(record.counts, "generations.counts");
  const counts = Object.fromEntries(
    (["current", "previous", "orphan", "staged"] as GenerationClassification[])
      .map((classification) => [
        classification,
        integerAt(countsRecord[classification], `generations.counts.${classification}`),
      ]),
  ) as Record<GenerationClassification, number>;
  for (const classification of Object.keys(counts) as GenerationClassification[]) {
    if (counts[classification] !== items.filter(
      (item) => item.classification === classification
    ).length) {
      fail(`generations.counts.${classification}`, "must match items");
    }
  }
  const current = record.current_generation_id === null
    ? null
    : validateGenerationId(record.current_generation_id);
  const previous = record.previous_generation_id === null
    ? null
    : validateGenerationId(record.previous_generation_id);
  if (
    (current === null ? 0 : 1) !== counts.current ||
    (previous === null ? 0 : 1) !== counts.previous
  ) {
    fail("generations.items", "must agree with current and previous pointers");
  }
  if (
    current !== null &&
    !items.some(
      (item) => item.generation_id === current && item.classification === "current"
    )
  ) {
    fail("generations.current_generation_id", "must identify the current item");
  }
  if (
    previous !== null &&
    !items.some(
      (item) => item.generation_id === previous && item.classification === "previous"
    )
  ) {
    fail("generations.previous_generation_id", "must identify the previous item");
  }
  return {
    schema: schemaAt(record.schema, "generations.schema"),
    current_generation_id: current,
    previous_generation_id: previous,
    items,
    counts,
  };
}

export function validateSessionPresence(value: unknown): SessionPresenceResult {
  const record = objectAt(value, "find_session");
  if (!Array.isArray(record.results) || record.results.length > 1_000) {
    fail("find_session.results", "must be an array of at most 1000 entries");
  }
  if (
    !Array.isArray(record.verification_failures) ||
    record.verification_failures.length > 1_000
  ) {
    fail("find_session.verification_failures", "must be a bounded array");
  }
  const results = record.results.map((item, index) => {
    const path = `find_session.results[${index}]`;
    const parsed = validateInventoryItem(item, path);
    const itemRecord = objectAt(item, path);
    const result: SessionPresenceResult["results"][number] = {
      ...parsed,
      exact_match_count: integerAt(
        itemRecord.exact_match_count,
        `${path}.exact_match_count`,
      ),
      verified: booleanAt(itemRecord.verified, `${path}.verified`),
    };
    if (itemRecord.scanned_record_count !== undefined) {
      result.scanned_record_count = integerAt(
        itemRecord.scanned_record_count,
        `${path}.scanned_record_count`,
      );
    }
    return result;
  });
  const failures = record.verification_failures.map((item, index) => {
    const path = `find_session.verification_failures[${index}]`;
    const itemRecord = objectAt(item, path);
    return {
      ...validateInventoryItem(item, path),
      error: stringAt(itemRecord.error, `${path}.error`),
    };
  });
  const source = validateSessionSource(record.source);
  const sessionId = validateSessionId(record.session_id);
  const canonicalFile = stringAt(record.canonical_file, "find_session.canonical_file");
  if (canonicalFile !== `agent-history-central/${source}/${sessionId}.md`) {
    fail("find_session.canonical_file", "must match the exact session identity");
  }
  const total = integerAt(
    record.total_exact_match_count,
    "find_session.total_exact_match_count",
  );
  if (
    total !== results.reduce(
      (sum, item) => sum + (item.verified ? item.exact_match_count : 0),
      0,
    )
  ) {
    fail("find_session.total_exact_match_count", "must match verified results");
  }
  const verified = booleanAt(record.verified, "find_session.verified");
  if (verified !== (failures.length === 0)) {
    fail("find_session.verified", "must agree with verification failures");
  }
  return {
    schema: schemaAt(record.schema, "find_session.schema"),
    source,
    session_id: sessionId,
    canonical_file: canonicalFile,
    results,
    total_exact_match_count: total,
    verification_failures: failures,
    verified,
  };
}

export function validateDeleteGenerationResult(
  value: unknown,
): DeleteGenerationResult {
  const record = objectAt(value, "delete_generation");
  const classification = generationClassificationAt(
    record.classification,
    "delete_generation.classification",
  );
  if (classification === "current") {
    fail("delete_generation.classification", "must never be current");
  }
  const dryRun = booleanAt(record.dry_run, "delete_generation.dry_run");
  const deleted = booleanAt(record.deleted, "delete_generation.deleted");
  if (dryRun && deleted) fail("delete_generation.deleted", "must be false on dry run");
  const idempotent = booleanAt(
    record.idempotent,
    "delete_generation.idempotent",
  );
  const targetId = validateGenerationId(record.target_id);
  const currentId = validateGenerationId(record.current_generation_id);
  const targetType = generationTargetTypeAt(
    record.target_type,
    "delete_generation.target_type",
  );
  if (targetId === currentId) {
    fail("delete_generation.target_id", "must not equal current generation");
  }
  if ((targetType === "staged") !== (classification === "staged")) {
    fail("delete_generation.classification", "must agree with target type");
  }
  const operationId = record.operation_id === null
    ? null
    : stringAt(record.operation_id, "delete_generation.operation_id");
  if (
    operationId !== null &&
    !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(
      operationId,
    )
  ) {
    fail("delete_generation.operation_id", "must be a canonical UUID");
  }
  const receipt = record.receipt === null
    ? null
    : objectAt(
        assertJsonSerializable(record.receipt),
        "delete_generation.receipt",
      ) as Record<string, JsonValue>;
  if (deleted && (dryRun || idempotent)) {
    fail("delete_generation.deleted", "must describe a new forced deletion");
  }
  if (!dryRun && deleted === idempotent) {
    fail(
      "delete_generation.idempotent",
      "must identify either a new deletion or an idempotent completion",
    );
  }
  if ((deleted || idempotent) && (operationId === null || receipt === null)) {
    fail(
      "delete_generation.receipt",
      "must accompany every completed deletion result",
    );
  }
  if (!deleted && !idempotent && receipt !== null) {
    fail("delete_generation.receipt", "must be null for an uncompleted dry run");
  }
  if (receipt !== null) {
    schemaAt(receipt.schema, "delete_generation.receipt.schema");
    if (receipt.state !== "complete") {
      fail("delete_generation.receipt.state", "must equal complete");
    }
    if (
      receipt.operation_id !== operationId ||
      receipt.target_id !== targetId ||
      receipt.target_type !== targetType ||
      receipt.classification !== classification ||
      receipt.expected_current_generation_id !== currentId
    ) {
      fail(
        "delete_generation.receipt",
        "must agree with the completed result identity",
      );
    }
    const counts = objectAt(
      receipt.counts,
      "delete_generation.receipt.counts",
    );
    if (
      integerAt(
        counts.directories_deleted,
        "delete_generation.receipt.counts.directories_deleted",
      ) !== 1
    ) {
      fail(
        "delete_generation.receipt.counts.directories_deleted",
        "must equal 1",
      );
    }
    const exactMatches = integerAt(
      counts.exact_match_count,
      "delete_generation.receipt.counts.exact_match_count",
    );
    const verification = objectAt(
      receipt.verification,
      "delete_generation.receipt.verification",
    );
    if (
      verification.target_absent !== true ||
      verification.pointer_consistent !== true ||
      verification.current_generation_id !== currentId ||
      verification.exact_match_count !== exactMatches ||
      verification.previous_generation_id === targetId
    ) {
      fail(
        "delete_generation.receipt.verification",
        "must prove target absence and pointer consistency",
      );
    }
  }
  return {
    schema: schemaAt(record.schema, "delete_generation.schema"),
    dry_run: dryRun,
    deleted,
    idempotent,
    target_id: targetId,
    target_type: targetType,
    classification,
    current_generation_id: currentId,
    operation_id: operationId,
    receipt,
  };
}

export function assertJsonSerializable(value: unknown): JsonValue {
  const visit = (current: unknown, path: string): JsonValue => {
    if (
      current === null ||
      typeof current === "string" ||
      typeof current === "boolean"
    ) {
      return current;
    }
    if (typeof current === "number") {
      if (!Number.isFinite(current)) fail(path, "must be a finite JSON number");
      return current;
    }
    if (Array.isArray(current)) {
      return current.map((item, index) => visit(item, `${path}[${index}]`));
    }
    if (typeof current === "object") {
      const prototype = Object.getPrototypeOf(current);
      if (prototype !== Object.prototype && prototype !== null) {
        fail(path, "must be a plain JSON object");
      }
      return Object.fromEntries(
        Object.entries(current).map(([key, item]) => [
          key,
          visit(item, `${path}.${key}`),
        ]),
      );
    }
    fail(path, "must be JSON serializable");
  }
  return visit(value, "value");
}
