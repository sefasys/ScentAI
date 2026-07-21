import { afterEach, describe, expect, it, vi } from "vitest";

import { handleRequest, type Env } from "./index";

function limiter(success = true) {
  return { limit: vi.fn().mockResolvedValue({ success }) };
}

function environment(overrides: Partial<Env> = {}): Env {
  return {
    ASSETS: {
      fetch: vi.fn().mockResolvedValue(new Response("asset", { status: 200 })),
    },
    SCENTAI_API_URL: "https://modal.example.test",
    SCENTAI_API_KEY: "server-secret",
    CHAT_RATE_LIMITER: limiter(),
    WARMUP_RATE_LIMITER: limiter(),
    POLL_RATE_LIMITER: limiter(),
    ...overrides,
  };
}

afterEach(() => vi.unstubAllGlobals());

describe("ScentAI public gateway", () => {
  it("injects the server secret and strips untrusted chat fields", async () => {
    const upstream = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ job_id: "job-1", status: "queued", poll_after_ms: 1000 }), {
        status: 202,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", upstream);
    const env = environment();
    const request = new Request("https://scentai.example/api/scentai/v1/chat/jobs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "CF-Connecting-IP": "203.0.113.8",
        Origin: "https://scentai.example",
        "X-API-Key": "browser-forgery",
      },
      body: JSON.stringify({ query: "  office scent  ", session_id: "session-1", debug: true }),
    });

    const response = await handleRequest(request, env);

    expect(response.status).toBe(202);
    const [url, init] = upstream.mock.calls[0] as [URL, RequestInit];
    expect(url.toString()).toBe("https://modal.example.test/v1/chat/jobs");
    expect(new Headers(init.headers).get("X-API-Key")).toBe("server-secret");
    expect(JSON.parse(String(init.body))).toEqual({
      query: "office scent",
      session_id: "session-1",
    });
    expect(response.headers.get("Content-Security-Policy")).toContain("connect-src 'self'");
  });

  it("rejects unknown routes and cross-origin writes", async () => {
    const env = environment();
    const unknown = await handleRequest(
      new Request("https://scentai.example/api/scentai/internal/config"),
      env,
    );
    const crossOrigin = await handleRequest(
      new Request("https://scentai.example/api/scentai/v1/runtime/warmup/jobs", {
        method: "POST",
        headers: { Origin: "https://attacker.example" },
      }),
      env,
    );

    expect(unknown.status).toBe(404);
    expect(crossOrigin.status).toBe(403);
  });

  it("returns 429 before calling Modal when the chat budget is exhausted", async () => {
    const upstream = vi.fn();
    vi.stubGlobal("fetch", upstream);
    const env = environment({ CHAT_RATE_LIMITER: limiter(false) });
    const response = await handleRequest(
      new Request("https://scentai.example/api/scentai/v1/chat/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: "hello" }),
      }),
      env,
    );

    expect(response.status).toBe(429);
    expect(response.headers.get("Retry-After")).toBe("60");
    expect(upstream).not.toHaveBeenCalled();
  });

  it("rejects oversized or malformed chat payloads", async () => {
    const env = environment();
    const oversized = await handleRequest(
      new Request("https://scentai.example/api/scentai/v1/chat/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Content-Length": "9000" },
        body: JSON.stringify({ query: "hello" }),
      }),
      env,
    );
    const debugOnly = await handleRequest(
      new Request("https://scentai.example/api/scentai/v1/chat/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ debug: true }),
      }),
      env,
    );

    expect(oversized.status).toBe(413);
    expect(debugOnly.status).toBe(422);
  });

  it("serves non-API requests from the static asset binding", async () => {
    const env = environment();
    const response = await handleRequest(new Request("https://scentai.example/app"), env);

    expect(response.status).toBe(200);
    expect(await response.text()).toBe("asset");
    expect(env.ASSETS.fetch).toHaveBeenCalledOnce();
    expect(response.headers.get("X-Frame-Options")).toBe("DENY");
  });

  it("answers public liveness checks without waking the Modal upstream", async () => {
    const upstream = vi.fn();
    vi.stubGlobal("fetch", upstream);
    const response = await handleRequest(
      new Request("https://scentai.example/api/scentai/health/live"),
      environment(),
    );

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ status: "ok", service: "scentai-web-gateway" });
    expect(upstream).not.toHaveBeenCalled();
    expect(response.headers.get("Cross-Origin-Resource-Policy")).toBe("same-origin");
  });
});
