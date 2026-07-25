export const APP_NAME = "agentkb";
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
  validation?: IndexValidation;
}

export interface IndexValidation {
  sqlite_count: number;
  fts_count: number;
  plaid_mapping_count: number;
  plaid_reverse_mapping_count: number;
  index_tree_hash: string;
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
  ready: true;
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
  relative_path?: string;
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
  validation: IndexValidation;
  duration_ms: number;
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
  path = "validation",
): IndexValidation {
  const record = objectAt(value, path);
  const indexTreeHash = stringAt(
    record.index_tree_hash,
    `${path}.index_tree_hash`,
  );
  if (!SHA256_PATTERN.test(indexTreeHash)) {
    fail(`${path}.index_tree_hash`, "must be lowercase SHA-256");
  }
  return {
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
}

export function validateManifest(
  value: unknown,
  path = "manifest",
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
  if (record.validation !== undefined) {
    result.validation = validateIndexValidation(
      record.validation,
      `${path}.validation`,
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
      : validateManifest(record.previous_manifest, "status.previous_manifest");
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
  };
  for (const key of [
    "relative_path",
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
  return {
    schema: schemaAt(record.schema, "warm.schema"),
    generation_id: validateGenerationId(record.generation_id),
    model: stringAt(record.model, "warm.model"),
    corpus_count: integerAt(record.corpus_count, "warm.corpus_count", 1),
    ready: true,
  };
}

export function validateBuildResult(value: unknown): BuildResult {
  const record = objectAt(value, "build");
  const corpusHash = stringAt(record.corpus_hash, "build.corpus_hash");
  if (!SHA256_PATTERN.test(corpusHash)) {
    fail("build.corpus_hash", "must be lowercase SHA-256");
  }
  return {
    schema: schemaAt(record.schema, "build.schema"),
    generation_id: validateGenerationId(record.generation_id),
    previous_generation_id:
      record.previous_generation_id === null
        ? null
        : validateGenerationId(record.previous_generation_id),
    model: stringAt(record.model, "build.model"),
    corpus_count: integerAt(record.corpus_count, "build.corpus_count", 1),
    corpus_hash: corpusHash,
    validation: validateIndexValidation(record.validation),
    duration_ms: finiteNumberAt(record.duration_ms, "build.duration_ms"),
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
