import { ModalClient } from "modal";

import {
  APP_NAME,
  type BuildResult,
  type SearchResult,
  type StatusResponse,
  type WarmResult,
  validateBuildResult,
  validateGenerationId,
  validateSearchRequest,
  validateSearchResult,
  validateStatus,
  validateWarmResult,
} from "./protocol";
import { basename } from "node:path";
import { localPath, type PathRoots } from "./refresh";

export interface AgentKbClient {
  status(): Promise<StatusResponse>;
  warm(): Promise<WarmResult>;
  warmDetached(): Promise<void>;
  search(query: string, k: number): Promise<SearchResult>;
  build(generationId: string): Promise<BuildResult>;
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

  async warm(): Promise<WarmResult> {
    return validateWarmResult(await this.#call("warm_current"));
  }

  async warmDetached(): Promise<void> {
    const fn = await this.#modal.functions.fromName(APP_NAME, "warm_current");
    await fn.spawn([]);
  }

  async search(query: string, k: number): Promise<SearchResult> {
    const request = validateSearchRequest({ query, k });
    const result = validateSearchResult(
      await this.#call("search_current", [request.query, request.k]),
    );
    return this.#roots ? localizeSearchResult(result, this.#roots) : result;
  }

  async build(generationId: string): Promise<BuildResult> {
    const id = validateGenerationId(generationId);
    return validateBuildResult(await this.#call("build_generation", [id]));
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
