import { describe, expect, test } from "bun:test";

import {
  assertJsonSerializable,
  createGenerationId,
  generationPaths,
  validateGenerationId,
  validateSearchRequest,
  validateStatus,
} from "../src/protocol";

const ID = "g-20260725T123456Z-001122aabbcc";

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
});

test("rejects values that JSON would silently discard", () => {
  expect(() => assertJsonSerializable({ omitted: undefined })).toThrow(
    /JSON serializable/,
  );
  expect(() => assertJsonSerializable(Number.NaN)).toThrow(/finite JSON number/);
});
