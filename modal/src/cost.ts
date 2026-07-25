import { APP_NAME, assertJsonSerializable, type JsonValue } from "./protocol";

export const COST_NOTE =
  "Metered usage reports can lag and totals are before credits and reservations.";

export interface BillingProcessResult {
  exitCode: number | null;
  stdout: string;
  stderr: string;
}

export type BillingSpawn = (command: string[]) => BillingProcessResult;

export interface BillingRow {
  object_id: string;
  description: string;
  environment: string;
  interval_start: string;
  resource: string;
  cost: string;
}

export interface CostReport {
  schema: 1;
  queried_range: {
    start: string;
    resolution: "h";
  };
  days: number;
  app_name: string;
  metered_cost: string;
  totals_by_resource: Record<string, string>;
  rows: BillingRow[];
  note: string;
}

interface Decimal {
  units: bigint;
  scale: number;
}

function parseDecimal(value: string, path: string): Decimal {
  const match = /^(\d+)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/.exec(value);
  if (!match) throw new TypeError(`${path} must be a nonnegative decimal string`);
  const exponent = Number(match[3] ?? "0");
  if (!Number.isSafeInteger(exponent) || Math.abs(exponent) > 1_000) {
    throw new TypeError(`${path} exponent is outside the supported range`);
  }
  const fraction = match[2] ?? "";
  let units = BigInt(`${match[1]}${fraction}` || "0");
  let scale = fraction.length - exponent;
  if (scale < 0) {
    units *= 10n ** BigInt(-scale);
    scale = 0;
  }
  return { units, scale };
}

function addDecimal(left: Decimal, right: Decimal): Decimal {
  const scale = Math.max(left.scale, right.scale);
  return {
    units:
      left.units * 10n ** BigInt(scale - left.scale) +
      right.units * 10n ** BigInt(scale - right.scale),
    scale,
  };
}

function formatDecimal(decimal: Decimal): string {
  if (decimal.units === 0n) return "0";
  if (decimal.scale === 0) return decimal.units.toString();
  const digits = decimal.units.toString().padStart(decimal.scale + 1, "0");
  const whole = digits.slice(0, -decimal.scale);
  const fraction = digits.slice(-decimal.scale).replace(/0+$/, "");
  return fraction ? `${whole}.${fraction}` : whole;
}

export function addDecimalStrings(values: string[]): string {
  return formatDecimal(
    values.reduce(
      (total, value, index) =>
        addDecimal(total, parseDecimal(value, `values[${index}]`)),
      { units: 0n, scale: 0 },
    ),
  );
}

function stringField(
  record: Record<string, unknown>,
  key: keyof BillingRow,
  index: number,
): string {
  const value = record[key];
  if (typeof value !== "string") {
    throw new TypeError(`billing rows[${index}].${key} must be a string`);
  }
  if (value === "") {
    throw new TypeError(`billing rows[${index}].${key} must not be empty`);
  }
  return value;
}

export function validateBillingRows(value: unknown): BillingRow[] {
  if (!Array.isArray(value)) throw new TypeError("billing report must be an array");
  return value.map((item, index) => {
    if (typeof item !== "object" || item === null || Array.isArray(item)) {
      throw new TypeError(`billing rows[${index}] must be an object`);
    }
    const record = item as Record<string, unknown>;
    const expected = [
      "object_id",
      "description",
      "environment",
      "interval_start",
      "resource",
      "cost",
    ];
    const keys = Object.keys(record);
    if (
      keys.length !== expected.length ||
      expected.some((key) => !Object.hasOwn(record, key))
    ) {
      throw new TypeError(
        `billing rows[${index}] must contain exactly: ${expected.join(", ")}`,
      );
    }
    const row = Object.fromEntries(
      expected.map((key) => [
        key,
        stringField(record, key as keyof BillingRow, index),
      ]),
    ) as unknown as BillingRow;
    if (Number.isNaN(Date.parse(row.interval_start))) {
      throw new TypeError(`billing rows[${index}].interval_start must be an ISO date-time`);
    }
    parseDecimal(row.cost, `billing rows[${index}].cost`);
    assertJsonSerializable(row as unknown as JsonValue);
    return row;
  });
}

export const defaultBillingSpawn: BillingSpawn = (command) => {
  const result = Bun.spawnSync(command, {
    stdin: "ignore",
    stdout: "pipe",
    stderr: "pipe",
  });
  return {
    exitCode: result.exitCode,
    stdout: result.stdout.toString(),
    stderr: result.stderr.toString(),
  };
};

export function reportCost(
  days: number,
  spawn: BillingSpawn = defaultBillingSpawn,
): CostReport {
  if (!Number.isInteger(days) || days < 1 || days > 7) {
    throw new TypeError("days must be an integer between 1 and 7");
  }
  const start = `${days} days ago`;
  const command = [
    "uvx",
    "--from",
    "modal==1.5.3",
    "modal",
    "billing",
    "report",
    "--start",
    start,
    "--resolution",
    "h",
    "--show-resources",
    "--json",
  ];
  let processResult: BillingProcessResult;
  try {
    processResult = spawn(command);
  } catch (error) {
    throw new Error(
      `could not start Modal billing report: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
  if (processResult.exitCode !== 0) {
    const detail = processResult.stderr.trim();
    throw new Error(
      `Modal billing report failed${
        detail ? `: ${detail}` : ` with exit ${processResult.exitCode ?? "unknown"}`
      }`,
    );
  }
  let external: unknown;
  try {
    external = JSON.parse(processResult.stdout);
  } catch (error) {
    throw new TypeError(
      `Modal billing report returned invalid JSON: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
  const rows = validateBillingRows(external).filter(
    (row) => row.description === APP_NAME,
  );
  const costsByResource = new Map<string, string[]>();
  for (const row of rows) {
    const costs = costsByResource.get(row.resource) ?? [];
    costs.push(row.cost);
    costsByResource.set(row.resource, costs);
  }
  const totalsByResource = Object.fromEntries(
    [...costsByResource.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([resource, costs]) => [resource, addDecimalStrings(costs)]),
  );
  return {
    schema: 1,
    queried_range: { start, resolution: "h" },
    days,
    app_name: APP_NAME,
    metered_cost: addDecimalStrings(rows.map((row) => row.cost)),
    totals_by_resource: totalsByResource,
    rows,
    note: COST_NOTE,
  };
}
