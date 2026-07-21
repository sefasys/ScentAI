import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, ScentAIClient } from "./api-client";

const validResponse = {
  request_id: "request-1",
  session_id: "session-1",
  answer: "A grounded answer.",
  route: "llm_grounded",
  language: "en",
  recommendations: [
    { perfume_id: 1, label: "Example by House", name: "Example", brand: "House" },
  ],
  validation_passed: true,
  generation_attempts: 1,
  total_seconds: 8.4,
  debug: null,
};

afterEach(() => vi.unstubAllGlobals());

describe("ScentAIClient", () => {
  it("supports a same-origin gateway without exposing an API key", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: "ok" })));
    vi.stubGlobal("fetch", fetchMock);
    const client = new ScentAIClient({ apiUrl: "/api/scentai", apiKey: "" });

    await expect(client.live()).resolves.toBe(true);
    expect(fetchMock).toHaveBeenCalledWith("/api/scentai/health/live", {
      signal: undefined,
    });
  });

  it("warms the runtime through a short polling job", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({ job_id: "warmup-1", status: "queued", poll_after_ms: 1 }),
          { status: 202, headers: { "Content-Type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            job_id: "warmup-1",
            status: "succeeded",
            ready: true,
            report: { ready: true },
            error: null,
            poll_after_ms: 1,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );
    vi.stubGlobal("fetch", fetchMock);
    const client = new ScentAIClient({ apiUrl: "https://api.example.test", apiKey: "secret" });

    await expect(client.warmup()).resolves.toBeUndefined();
    expect(fetchMock.mock.calls[0][0]).toBe("https://api.example.test/v1/runtime/warmup/jobs");
    expect(fetchMock.mock.calls[1][0]).toBe("https://api.example.test/v1/runtime/warmup/jobs/warmup-1");
  });

  it("sends the API key and existing session to Modal", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({ job_id: "job-1", status: "queued", poll_after_ms: 1 }),
          { status: 202, headers: { "Content-Type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            job_id: "job-1",
            status: "succeeded",
            response: validResponse,
            error: null,
            error_status: null,
            poll_after_ms: 1,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );
    vi.stubGlobal("fetch", fetchMock);
    const client = new ScentAIClient({ apiUrl: "https://api.example.test/", apiKey: "secret" });

    const result = await client.chat({ query: "More options", session_id: "session-1" });

    expect(result.session_id).toBe("session-1");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const [url, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://api.example.test/v1/chat/jobs");
    expect(options.headers).toEqual({
      "Content-Type": "application/json",
      "X-API-Key": "secret",
    });
    expect(JSON.parse(String(options.body))).toEqual({
      query: "More options",
      session_id: "session-1",
    });
    expect(fetchMock.mock.calls[1][0]).toBe("https://api.example.test/v1/chat/jobs/job-1");
  });

  it("turns an API failure into a status-aware error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Invalid API key" }), {
          status: 401,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );
    const client = new ScentAIClient({ apiUrl: "https://api.example.test", apiKey: "bad" });

    try {
      await client.chat({ query: "Hello" });
      expect.fail("Expected chat to reject");
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect(error).toMatchObject({ status: 401, detail: "Invalid API key" });
    }
  });

  it("retries transient polling failures without creating a duplicate chat job", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({ job_id: "job-retry", status: "queued", poll_after_ms: 1 }),
          { status: 202, headers: { "Content-Type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: "Gateway restarting" }), {
          status: 503,
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            job_id: "job-retry",
            status: "succeeded",
            response: validResponse,
            error: null,
            error_status: null,
            poll_after_ms: 1,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );
    vi.stubGlobal("fetch", fetchMock);
    const client = new ScentAIClient({ apiUrl: "https://api.example.test", apiKey: "secret" });

    await expect(client.chat({ query: "Office scent" })).resolves.toMatchObject({
      request_id: "request-1",
    });

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls.filter(([, options]) => options?.method === "POST")).toHaveLength(1);
  });

  it("treats an already expired session as successfully deleted", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 404 })));
    const client = new ScentAIClient({ apiUrl: "https://api.example.test", apiKey: "secret" });

    await expect(client.deleteSession("expired-session")).resolves.toBeUndefined();
  });
});
