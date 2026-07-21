import {
  chatJobAcceptedSchema,
  chatJobStatusSchema,
  warmupJobAcceptedSchema,
  warmupJobStatusSchema,
  type ChatRequest,
  type ConnectionConfig,
} from "../types/api";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export type JobProgress = "queued" | "running";
export type JobProgressHandler = (status: JobProgress) => void;

function normalizeApiUrl(value: string): string {
  const normalized = value.trim().replace(/\/+$/, "");
  if (normalized.startsWith("/")) return normalized;
  const url = new URL(normalized);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("API adresi HTTP veya HTTPS kullanmalıdır.");
  }
  return url.toString().replace(/\/$/, "");
}

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string" && body.detail.trim()) {
      return body.detail;
    }
  } catch {
    // The status text below is sufficient for non-JSON failures.
  }
  return response.statusText || `HTTP ${response.status}`;
}

function abortableDelay(milliseconds: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timeout = globalThis.setTimeout(resolve, milliseconds);
    signal?.addEventListener(
      "abort",
      () => {
        globalThis.clearTimeout(timeout);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

function isTransientStatus(status: number): boolean {
  return status === 502 || status === 503 || status === 504;
}

async function pollFetch(url: string, headers: Record<string, string>, signal?: AbortSignal) {
  const retryDelays = [500, 1_000, 2_000];
  for (let attempt = 0; ; attempt += 1) {
    try {
      const response = await fetch(url, { headers, signal });
      if (!isTransientStatus(response.status) || attempt >= retryDelays.length) return response;
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") throw error;
      if (!(error instanceof TypeError) || attempt >= retryDelays.length) throw error;
    }
    await abortableDelay(retryDelays[attempt], signal);
  }
}

export class ScentAIClient {
  readonly apiUrl: string;
  readonly apiKey: string;

  constructor(config: ConnectionConfig) {
    this.apiUrl = normalizeApiUrl(config.apiUrl);
    this.apiKey = config.apiKey.trim();
  }

  private headers(extra?: Record<string, string>): Record<string, string> {
    return {
      ...extra,
      ...(this.apiKey ? { "X-API-Key": this.apiKey } : {}),
    };
  }

  async live(signal?: AbortSignal): Promise<boolean> {
    const response = await fetch(`${this.apiUrl}/health/live`, { signal });
    if (!response.ok) {
      throw new ApiError(response.status, await readError(response));
    }
    return true;
  }

  async warmup(signal?: AbortSignal, onProgress?: JobProgressHandler): Promise<void> {
    const acceptedResponse = await fetch(`${this.apiUrl}/v1/runtime/warmup/jobs`, {
      method: "POST",
      headers: this.headers(),
      signal,
    });
    if (!acceptedResponse.ok) {
      throw new ApiError(acceptedResponse.status, await readError(acceptedResponse));
    }
    const accepted = warmupJobAcceptedSchema.parse(await acceptedResponse.json());
    onProgress?.(accepted.status);
    const deadline = Date.now() + 15 * 60 * 1000;
    let pollAfterMs = accepted.poll_after_ms;

    while (Date.now() < deadline) {
      await abortableDelay(pollAfterMs, signal);
      const statusResponse = await pollFetch(
        `${this.apiUrl}/v1/runtime/warmup/jobs/${encodeURIComponent(accepted.job_id)}`,
        this.headers(),
        signal,
      );
      if (!statusResponse.ok) {
        throw new ApiError(statusResponse.status, await readError(statusResponse));
      }
      const job = warmupJobStatusSchema.parse(await statusResponse.json());
      if (job.status === "queued" || job.status === "running") onProgress?.(job.status);
      pollAfterMs = job.poll_after_ms;
      if (job.status === "succeeded" && job.ready) return;
      if (job.status === "failed") {
        throw new ApiError(503, job.error ?? "ScentAI runtime could not start");
      }
    }
    throw new ApiError(408, "ScentAI runtime warm-up timed out");
  }

  async chat(payload: ChatRequest, signal?: AbortSignal, onProgress?: JobProgressHandler) {
    const acceptedResponse = await fetch(`${this.apiUrl}/v1/chat/jobs`, {
      method: "POST",
      headers: {
        ...this.headers({ "Content-Type": "application/json" }),
      },
      body: JSON.stringify(payload),
      signal,
    });
    if (!acceptedResponse.ok) {
      throw new ApiError(acceptedResponse.status, await readError(acceptedResponse));
    }
    const accepted = chatJobAcceptedSchema.parse(await acceptedResponse.json());
    onProgress?.(accepted.status);
    const deadline = Date.now() + 12 * 60 * 1000;
    let pollAfterMs = accepted.poll_after_ms;

    while (Date.now() < deadline) {
      await abortableDelay(pollAfterMs, signal);
      const statusResponse = await pollFetch(
        `${this.apiUrl}/v1/chat/jobs/${encodeURIComponent(accepted.job_id)}`,
        this.headers(),
        signal,
      );
      if (!statusResponse.ok) {
        throw new ApiError(statusResponse.status, await readError(statusResponse));
      }
      const job = chatJobStatusSchema.parse(await statusResponse.json());
      if (job.status === "queued" || job.status === "running") onProgress?.(job.status);
      pollAfterMs = job.poll_after_ms;
      if (job.status === "succeeded" && job.response) return job.response;
      if (job.status === "failed") {
        throw new ApiError(job.error_status ?? 500, job.error ?? "Chat job failed");
      }
    }
    throw new ApiError(408, "ScentAI job polling timed out");
  }

  async deleteSession(sessionId: string, signal?: AbortSignal): Promise<void> {
    const response = await fetch(`${this.apiUrl}/v1/sessions/${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
      headers: this.headers(),
      signal,
    });
    if (!response.ok && response.status !== 404) {
      throw new ApiError(response.status, await readError(response));
    }
  }
}
