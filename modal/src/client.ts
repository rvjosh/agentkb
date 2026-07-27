import { ModalClient } from "modal";

import {
  APP_NAME,
  type BuildResult,
  type DeleteGenerationResult,
  type GenerationInventory,
  type GenerationTargetType,
  type PrunePreviousResult,
  type SearchResult,
  type SessionPresenceResult,
  type StatusResponse,
  type WarmResult,
  validateBuildResult,
  validateDeleteGenerationResult,
  validateGenerationInventory,
  validateGenerationId,
  validatePrunePreviousResult,
  validateSessionId,
  validateSessionPresence,
  validateSessionSource,
  validateSearchRequest,
  validateSearchResult,
  validateStatus,
  validateWarmResult,
} from "./protocol";
import { basename } from "node:path";
import { localPath, type PathRoots } from "./refresh";

export interface AgentKbClient {
  status(): Promise<StatusResponse>;
  generations(): Promise<GenerationInventory>;
  findSession(source: "claude" | "codex", sessionId: string): Promise<SessionPresenceResult>;
  deleteGeneration(
    generationId: string,
    targetType: GenerationTargetType,
    expectedCurrent: string,
    force: boolean,
    actor: string,
    reason: string,
    exactSessionKey?: string,
  ): Promise<DeleteGenerationResult>;
  warm(): Promise<WarmResult>;
  warmDetached(): Promise<void>;
  search(
    query: string,
    k: number,
    transcriptSessions?: boolean,
  ): Promise<SearchResult>;
  build(generationId: string): Promise<BuildResult>;
  prunePrevious(
    generationId: string,
    dryRun: boolean,
  ): Promise<PrunePreviousResult>;
  close(): void;
}

export class ModalAgentKbClient implements AgentKbClient {
  readonly #modal: ModalClient;
  readonly #roots: PathRoots | undefined;

  constructor(modalClient = new ModalClient(), roots?: PathRoots) {
    this.#modal = modalClient;
    this.#roots = roots;
  }

  async #call(functionName: string, args: unknown[] = []): Promise<unknown> {
    const fn = await this.#modal.functions.fromName(APP_NAME, functionName);
    return fn.remote(args);
  }

  async status(): Promise<StatusResponse> {
    return validateStatus(await this.#call("status"));
  }

  async generations(): Promise<GenerationInventory> {
    return validateGenerationInventory(await this.#call("generations"));
  }

  async findSession(
    source: "claude" | "codex",
    sessionId: string,
  ): Promise<SessionPresenceResult> {
    const exactSource = validateSessionSource(source);
    const exactSessionId = validateSessionId(sessionId);
    return validateSessionPresence(
      await this.#call("find_session", [exactSource, exactSessionId]),
    );
  }

  async deleteGeneration(
    generationId: string,
    targetType: GenerationTargetType,
    expectedCurrent: string,
    force: boolean,
    actor: string,
    reason: string,
    exactSessionKey?: string,
  ): Promise<DeleteGenerationResult> {
    const id = validateGenerationId(generationId);
    const expected = validateGenerationId(expectedCurrent);
    if (targetType !== "generation" && targetType !== "staged") {
      throw new TypeError("target type must be generation or staged");
    }
    if (!actor.trim() || actor.length > 200) {
      throw new TypeError("actor must be a non-empty bounded string");
    }
    if (!reason.trim() || reason.length > 1_000) {
      throw new TypeError("reason must be a non-empty bounded string");
    }
    if (exactSessionKey !== undefined) {
      const [source, sessionId, ...rest] = exactSessionKey.split("/");
      if (rest.length || sessionId === undefined) {
        throw new TypeError("exact session key must be source/session-id");
      }
      validateSessionSource(source);
      validateSessionId(sessionId);
    }
    return validateDeleteGenerationResult(
      await this.#call("delete_generation_exact", [
        id,
        targetType,
        expected,
        force,
        actor.trim(),
        reason.trim(),
        exactSessionKey ?? null,
      ]),
    );
  }

  async warm(): Promise<WarmResult> {
    return validateWarmResult(await this.#call("warm_current"));
  }

  async warmDetached(): Promise<void> {
    const fn = await this.#modal.functions.fromName(APP_NAME, "warm_current");
    await fn.spawn([]);
  }

  async search(
    query: string,
    k: number,
    transcriptSessions = false,
  ): Promise<SearchResult> {
    const request = validateSearchRequest({
      query,
      k,
      transcript_sessions: transcriptSessions,
    });
    const args = request.transcript_sessions
      ? [request.query, request.k, true]
      : [request.query, request.k];
    const result = validateSearchResult(
      await this.#call("search_current", args),
    );
    return this.#roots ? localizeSearchResult(result, this.#roots) : result;
  }

  async build(generationId: string): Promise<BuildResult> {
    const id = validateGenerationId(generationId);
    return validateBuildResult(await this.#call("build_generation", [id]));
  }

  async prunePrevious(
    generationId: string,
    dryRun: boolean,
  ): Promise<PrunePreviousResult> {
    const id = validateGenerationId(generationId);
    return validatePrunePreviousResult(
      await this.#call("prune_previous", [id, dryRun]),
    );
  }

  close(): void {
    this.#modal.close();
  }
}

export function createModalAgentKbClient(roots?: PathRoots): AgentKbClient {
  return new ModalAgentKbClient(new ModalClient(), roots);
}

export function localizeSearchResult(
  result: SearchResult,
  roots: PathRoots,
): SearchResult {
  return {
    ...result,
    results: result.results.map((hit) => {
      const external = Object.entries(roots.externalRoots)
        .sort(([left], [right]) => right.length - left.length)
        .find(([prefix]) => hit.relative_path.startsWith(prefix));
      if (external) {
        const [prefix, root] = external;
        const path = localPath(root, hit.relative_path.slice(prefix.length));
        return { ...hit, file: path, path, filename: basename(path) };
      }
      const root =
        hit.collection === "chats"
          ? roots.chatsReadableRoot
          : hit.collection === "wiki" || hit.collection === "wiki:source"
            ? roots.wikiRoot
            : undefined;
      if (!root) {
        throw new TypeError(`cannot localize collection: ${hit.collection}`);
      }
      const path = localPath(root, hit.relative_path);
      return { ...hit, file: path, path, filename: basename(path) };
    }),
  };
}
