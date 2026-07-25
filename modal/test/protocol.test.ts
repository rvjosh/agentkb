import { describe, expect, test } from "bun:test";

import {
  assertJsonSerializable,
  createGenerationId,
  generationPaths,
  validateBuildResult,
  validateGenerationId,
  validateSearchRequest,
  validateStatus,
  validateWarmResult,
} from "../src/protocol";

const ID = "g-20260725T123456Z-001122aabbcc";

function immutableCertificate(documentCount: number) {
  return {
    schema: 1 as const,
    fast_plaid_version: "1.3.0.290",
    num_chunks: 1,
    nbits: 4,
    document_count: documentCount,
    padding_rows: 0,
    artifacts: {
      "metadata.json": { size_bytes: 1 },
      "doclens.0.json": { size_bytes: 1 },
      "centroids.npy": { size_bytes: 1, dtype: "float16", shape: [2, 128] },
      "avg_residual.npy": { size_bytes: 1, dtype: "float32", shape: [1] },
      "bucket_cutoffs.npy": { size_bytes: 1, dtype: "float32", shape: [1] },
      "bucket_weights.npy": { size_bytes: 1, dtype: "float32", shape: [1] },
      "ivf.npy": { size_bytes: 1, dtype: "int64", shape: [1] },
      "ivf_lengths.npy": { size_bytes: 1, dtype: "int32", shape: [1] },
      "merged_codes.npy": { size_bytes: 1, dtype: "int64", shape: [1] },
      "merged_residuals.npy": { size_bytes: 1, dtype: "uint8", shape: [1, 16] },
    },
  };
}

describe("generation IDs and paths", () => {
  test("creates deterministic path-safe IDs", () => {
    expect(
      createGenerationId(
        new Date("2026-07-25T12:34:56.789Z"),
        Uint8Array.from([0, 17, 34, 170, 187, 204]),
      ),
    ).toBe(ID);
    expect(generationPaths(ID)).toEqual({
      stagedCorpus: `staged/${ID}/corpus.jsonl`,
      stagedManifest: `staged/${ID}/manifest.json`,
      generationIndex: `generations/${ID}/index`,
      generationManifest: `generations/${ID}/manifest.json`,
    });
  });

  test.each([
    "../escape",
    "g-20260725T123456Z-UPPERCASE000",
    "g-20260725T123456Z-abc",
    " g-20260725T123456Z-001122aabbcc",
  ])("rejects invalid generation ID %s", (value) => {
    expect(() => validateGenerationId(value)).toThrow();
    expect(() => generationPaths(value)).toThrow();
  });
});

test("validates search input bounds", () => {
  expect(validateSearchRequest({ query: "agent memory", k: 10 })).toEqual({
    query: "agent memory",
    k: 10,
  });
  expect(() => validateSearchRequest({ query: " ", k: 10 })).toThrow();
  expect(() => validateSearchRequest({ query: "x", k: 0 })).toThrow();
  expect(() => validateSearchRequest({ query: "x", k: 101 })).toThrow();
});

test("validates warm startup timing breakdown", () => {
  const warm = {
    schema: 1,
    generation_id: ID,
    model: "model",
    corpus_count: 1,
    startup_timing_ms: {
      artifact_mount: 10,
      certificate: 2,
      model: 20,
      index_load: 30,
      total: 62,
    },
    ready: true,
  };
  expect(validateWarmResult(warm).startup_timing_ms.index_load).toBe(30);
  expect(() =>
    validateWarmResult({
      ...warm,
      startup_timing_ms: { ...warm.startup_timing_ms, certificate: -1 },
    }),
  ).toThrow(/must be non-negative/);
  expect(() =>
    validateWarmResult({ ...warm, startup_timing_ms: undefined }),
  ).toThrow(/must be an object/);
});

test("validates auditable build batch metrics", () => {
  const build = {
    schema: 1,
    generation_id: ID,
    previous_generation_id: null,
    model: "model",
    corpus_count: 513,
    corpus_hash: "a".repeat(64),
    document_batch_size: 256,
    document_batch_count: 3,
    embedding_dimension: 128,
    staged_embedding_bytes: 123456,
    plaid_create_count: 1,
    plaid_kmeans_sample_size: 16_384,
    plaid_permutation_algorithm: "sha256-key-sort-v1",
    validation: {
      sqlite_count: 513,
      fts_count: 513,
      plaid_mapping_count: 513,
      plaid_reverse_mapping_count: 513,
      index_tree_hash: "b".repeat(64),
      immutable_premerged: immutableCertificate(513),
    },
    duration_ms: 1,
  };
  expect(validateBuildResult(build).document_batch_count).toBe(3);
  expect(() =>
    validateBuildResult({ ...build, document_batch_size: 512 }),
  ).toThrow(/must equal 256/);
  expect(() =>
    validateBuildResult({ ...build, document_batch_count: 2 }),
  ).toThrow(/cover corpus_count/);
  expect(() =>
    validateBuildResult({ ...build, plaid_create_count: 2 }),
  ).toThrow(/must equal 1/);
  expect(() =>
    validateBuildResult({ ...build, plaid_kmeans_sample_size: 8_192 }),
  ).toThrow(/between 16384 and 32768/);
  expect(() =>
    validateBuildResult({ ...build, plaid_permutation_algorithm: "none" }),
  ).toThrow(/sha256-key-sort-v1/);
});

test("validates empty and populated status responses", () => {
  expect(
    validateStatus({
      schema: 1,
      current_generation_id: null,
      previous_generation_id: null,
      published_at: null,
      current_manifest: null,
      previous_manifest: null,
    }).current_generation_id,
  ).toBeNull();

  expect(() =>
    validateStatus({
      schema: 1,
      current_generation_id: ID,
      previous_generation_id: null,
      published_at: "2026-07-25T12:34:56Z",
      current_manifest: null,
      previous_manifest: null,
    }),
  ).toThrow(/must agree/);

  const previousId = "g-20260725T120000Z-aabbccddeeff";
  const legacyValidation = {
    sqlite_count: 1,
    fts_count: 1,
    plaid_mapping_count: 1,
    plaid_reverse_mapping_count: 1,
    index_tree_hash: "b".repeat(64),
  };
  const status = validateStatus({
    schema: 1,
    current_generation_id: ID,
    previous_generation_id: previousId,
    published_at: "2026-07-25T12:34:56Z",
    current_manifest: {
      schema: 1,
      generation_id: ID,
      model: "model",
      corpus_count: 1,
      corpus_hash: "a".repeat(64),
      validation: {
        ...legacyValidation,
        immutable_premerged: immutableCertificate(1),
      },
    },
    previous_manifest: {
      schema: 1,
      generation_id: previousId,
      model: "model",
      corpus_count: 1,
      corpus_hash: "c".repeat(64),
      validation: legacyValidation,
    },
  });
  expect(status.previous_manifest?.validation?.immutable_premerged).toBeUndefined();
});

test("rejects values that JSON would silently discard", () => {
  expect(() => assertJsonSerializable({ omitted: undefined })).toThrow(
    /JSON serializable/,
  );
  expect(() => assertJsonSerializable(Number.NaN)).toThrow(/finite JSON number/);
});
