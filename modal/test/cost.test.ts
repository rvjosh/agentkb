import { expect, test } from "bun:test";

import {
  addDecimalStrings,
  reportCost,
  validateBillingRows,
} from "../src/cost";

function row(overrides: Record<string, unknown> = {}) {
  return {
    object_id: "ap-1",
    description: "agentkb",
    environment: "main",
    interval_start: "2026-07-25T12:00:00+00:00",
    resource: "T4",
    cost: "0.1",
    ...overrides,
  };
}

test("adds decimal strings exactly including exponent forms", () => {
  expect(addDecimalStrings(["0.1", "0.2", "0E-8", "1.25e-2", "2E+1"]))
    .toBe("20.3125");
  expect(addDecimalStrings(["999999999999999999.9", "0.1"]))
    .toBe("1000000000000000000");
});

test("reports exact AgentKB totals by resource without floats", () => {
  const report = reportCost(7, () => ({
    exitCode: 0,
    stderr: "",
    stdout: JSON.stringify([
      row({ cost: "0.1" }),
      row({ object_id: "ap-2", resource: "CPU", cost: "0.2" }),
      row({ object_id: "ap-3", cost: "0E-8" }),
      row({ object_id: "ap-4", description: "not-agentkb", cost: "100" }),
    ]),
  }));
  expect(report.metered_cost).toBe("0.3");
  expect(report.totals_by_resource).toEqual({ CPU: "0.2", T4: "0.1" });
  expect(report.rows).toHaveLength(3);
  expect(report.note).toContain("before credits and reservations");
});

test("external JSON and every billing row field fail closed", () => {
  expect(() => reportCost(1, () => ({
    exitCode: 0,
    stderr: "",
    stdout: "not json",
  }))).toThrow(/invalid JSON/);
  expect(() => validateBillingRows([row({ cost: 0.1 })])).toThrow(/must be a string/);
  expect(() => validateBillingRows([row({ unexpected: "field" })])).toThrow(/exactly/);
  expect(() => validateBillingRows([row({ interval_start: "not-a-date" })]))
    .toThrow(/ISO date-time/);
  expect(() => validateBillingRows([row({ cost: "-1" })])).toThrow(/nonnegative/);
});

test("billing subprocess failures are runtime errors", () => {
  expect(() => reportCost(1, () => ({
    exitCode: 1,
    stdout: "",
    stderr: "not authenticated",
  }))).toThrow(/not authenticated/);
  expect(() => reportCost(1, () => {
    throw new Error("uvx missing");
  })).toThrow(/could not start/);
});
