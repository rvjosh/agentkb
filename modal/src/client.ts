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

export interface AgentKbClient {
  status(): Promise<StatusResponse>;
  warm(): Promise<WarmResult>;
  search(query: string, k: number): Promise<SearchResult>;
  build(generationId: string): Promise<BuildResult>;
  close(): void;
}

export class ModalAgentKbClient implements AgentKbClient {
  readonly #modal: ModalClient;

  constructor(modalClient = new ModalClient()) {
    this.#modal = modalClient;
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

  async search(query: string, k: number): Promise<SearchResult> {
    const request = validateSearchRequest({ query, k });
    return validateSearchResult(
      await this.#call("search_current", [request.query, request.k]),
    );
  }

  async build(generationId: string): Promise<BuildResult> {
    const id = validateGenerationId(generationId);
    return validateBuildResult(await this.#call("build_generation", [id]));
  }

  close(): void {
    this.#modal.close();
  }
}
