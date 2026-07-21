import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

function apiResponse(requestId: string, answer: string) {
  return {
    request_id: requestId,
    session_id: "session-42",
    answer,
    route: "llm_grounded",
    language: "tr",
    recommendations: [],
    validation_passed: true,
    generation_attempts: 1,
    total_seconds: 4.2,
    debug: null,
  };
}

describe("ScentAI chat session", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("reuses the server session and deletes it when a new conversation starts", async () => {
    const chatBodies: Array<Record<string, unknown>> = [];
    let activeJob = 0;
    const fetchMock = vi.fn().mockImplementation(async (url: string, options?: RequestInit) => {
      if (url.endsWith("/v1/runtime/warmup/jobs") && options?.method === "POST") {
        return new Response(JSON.stringify({
          job_id: "warmup-1",
          status: "queued",
          poll_after_ms: 1,
        }), { status: 202, headers: { "Content-Type": "application/json" } });
      }
      if (url.endsWith("/v1/runtime/warmup/jobs/warmup-1")) {
        return new Response(JSON.stringify({
          job_id: "warmup-1",
          status: "succeeded",
          ready: true,
          report: { ready: true },
          error: null,
          poll_after_ms: 1,
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      if (url.endsWith("/v1/chat/jobs") && options?.method === "POST") {
        const body = JSON.parse(String(options?.body)) as Record<string, unknown>;
        chatBodies.push(body);
        activeJob = chatBodies.length;
        return new Response(JSON.stringify({
          job_id: `job-${activeJob}`,
          status: "queued",
          poll_after_ms: 1,
        }), {
          status: 202,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.includes("/v1/chat/jobs/job-")) {
        return new Response(JSON.stringify({
          job_id: `job-${activeJob}`,
          status: "succeeded",
          response: apiResponse(`request-${activeJob}`, `Yanıt ${activeJob}`),
          error: null,
          error_status: null,
          poll_after_ms: 1,
        }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.includes("/v1/sessions/")) return new Response(null, { status: 204 });
      return new Response(JSON.stringify({ status: "ok" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    expect(screen.getByText("Parfüm danışmanı hazırlanıyor")).toBeInTheDocument();
    const composer = await screen.findByLabelText("Mesaj");
    fireEvent.change(composer, { target: { value: "İlk önerim" } });
    fireEvent.click(screen.getByRole("button", { name: "Gönder" }));
    expect(await screen.findByText("Yanıt 1")).toBeInTheDocument();

    fireEvent.change(composer, { target: { value: "Başka seçenekler" } });
    fireEvent.click(screen.getByRole("button", { name: "Gönder" }));
    expect(await screen.findByText("Yanıt 2")).toBeInTheDocument();

    expect(chatBodies).toEqual([
      { query: "İlk önerim" },
      { query: "Başka seçenekler", session_id: "session-42" },
    ]);

    fireEvent.click(screen.getByRole("button", { name: "Yeni konuşma" }));
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining("/v1/sessions/session-42"),
        expect.objectContaining({ method: "DELETE" }),
      );
    });
    expect(screen.getByText("Nasıl bir parfüm arıyorsun?")).toBeInTheDocument();
  });

  it("restores the conversation from persistent browser storage", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(async (url: string) => {
      if (url.endsWith("/v1/runtime/warmup/jobs")) {
        return new Response(JSON.stringify({
          job_id: "warmup-restored",
          status: "queued",
          poll_after_ms: 1,
        }), { status: 202, headers: { "Content-Type": "application/json" } });
      }
      return new Response(JSON.stringify({
        job_id: "warmup-restored",
        status: "succeeded",
        ready: true,
        report: { ready: true },
        error: null,
        poll_after_ms: 1,
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }));
    window.localStorage.setItem(
      "scentai.conversation.v1",
      JSON.stringify({
        version: 2,
        apiUrl: "/api/scentai",
        sessionId: "session-restored",
        messages: [{ id: "user-1", role: "user", content: "Önceki mesaj" }],
        updatedAt: "2026-07-20T00:00:00.000Z",
      }),
    );

    render(<App />);

    expect(await screen.findByText("Önceki mesaj")).toBeInTheDocument();
  });

  it("migrates the previous tab-only conversation into persistent storage", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(async (url: string) => {
      if (url.endsWith("/v1/runtime/warmup/jobs")) {
        return new Response(JSON.stringify({
          job_id: "warmup-migration",
          status: "queued",
          poll_after_ms: 1,
        }), { status: 202, headers: { "Content-Type": "application/json" } });
      }
      return new Response(JSON.stringify({
        job_id: "warmup-migration",
        status: "succeeded",
        ready: true,
        report: { ready: true },
        error: null,
        poll_after_ms: 1,
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }));
    window.sessionStorage.setItem(
      "scentai.conversation.v1",
      JSON.stringify({
        version: 1,
        apiUrl: "/api/scentai",
        sessionId: "legacy-session",
        messages: [{ id: "legacy-user", role: "user", content: "Eski sekme mesajı" }],
      }),
    );

    render(<App />);

    expect(await screen.findByText("Eski sekme mesajı")).toBeInTheDocument();
    expect(window.localStorage.getItem("scentai.conversation.v1")).toContain("legacy-session");
    expect(window.sessionStorage.getItem("scentai.conversation.v1")).toBeNull();
  });

  it("replays once without a stale server session after a 404", async () => {
    const chatBodies: Array<Record<string, unknown>> = [];
    let chatCreateCount = 0;
    vi.stubGlobal("fetch", vi.fn().mockImplementation(async (url: string, options?: RequestInit) => {
      if (url.endsWith("/v1/runtime/warmup/jobs") && options?.method === "POST") {
        return new Response(JSON.stringify({
          job_id: "warmup-recovery",
          status: "queued",
          poll_after_ms: 1,
        }), { status: 202, headers: { "Content-Type": "application/json" } });
      }
      if (url.endsWith("/v1/runtime/warmup/jobs/warmup-recovery")) {
        return new Response(JSON.stringify({
          job_id: "warmup-recovery",
          status: "succeeded",
          ready: true,
          report: { ready: true },
          error: null,
          poll_after_ms: 1,
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      if (url.endsWith("/v1/chat/jobs") && options?.method === "POST") {
        chatCreateCount += 1;
        chatBodies.push(JSON.parse(String(options.body)) as Record<string, unknown>);
        if (chatCreateCount === 1) {
          return new Response(JSON.stringify({ detail: "Unknown session" }), {
            status: 404,
            headers: { "Content-Type": "application/json" },
          });
        }
        return new Response(JSON.stringify({
          job_id: "job-recovered",
          status: "queued",
          poll_after_ms: 1,
        }), { status: 202, headers: { "Content-Type": "application/json" } });
      }
      if (url.endsWith("/v1/chat/jobs/job-recovered")) {
        return new Response(JSON.stringify({
          job_id: "job-recovered",
          status: "succeeded",
          response: { ...apiResponse("request-recovered", "Kurtarılan yanıt"), session_id: "session-new" },
          error: null,
          error_status: null,
          poll_after_ms: 1,
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      return new Response(null, { status: 404 });
    }));
    window.localStorage.setItem(
      "scentai.conversation.v1",
      JSON.stringify({
        version: 2,
        apiUrl: "/api/scentai",
        sessionId: "session-stale",
        messages: [{ id: "assistant-old", role: "assistant", content: "Eski yanıt", recommendations: [], route: "test", language: "tr", totalSeconds: 1 }],
        updatedAt: "2026-07-20T00:00:00.000Z",
      }),
    );

    render(<App />);
    const composer = await screen.findByLabelText("Mesaj");
    fireEvent.change(composer, { target: { value: "Yeni sorum" } });
    fireEvent.click(screen.getByRole("button", { name: "Gönder" }));

    expect(await screen.findByText("Kurtarılan yanıt")).toBeInTheDocument();
    expect(chatBodies).toEqual([
      { query: "Yeni sorum", session_id: "session-stale" },
      { query: "Yeni sorum" },
    ]);
    expect(screen.getAllByText("Yeni sorum")).toHaveLength(1);
  });

  it("removes an unsent user bubble when chat creation fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(async (url: string, options?: RequestInit) => {
      if (url.endsWith("/v1/runtime/warmup/jobs") && options?.method === "POST") {
        return new Response(JSON.stringify({
          job_id: "warmup-failure",
          status: "queued",
          poll_after_ms: 1,
        }), { status: 202, headers: { "Content-Type": "application/json" } });
      }
      if (url.endsWith("/v1/runtime/warmup/jobs/warmup-failure")) {
        return new Response(JSON.stringify({
          job_id: "warmup-failure",
          status: "succeeded",
          ready: true,
          report: { ready: true },
          error: null,
          poll_after_ms: 1,
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      return new Response(JSON.stringify({ detail: "Temporary failure" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      });
    }));
    render(<App />);

    const composer = await screen.findByLabelText("Mesaj");
    fireEvent.change(composer, { target: { value: "Tekrar gönderilebilir mesaj" } });
    fireEvent.click(screen.getByRole("button", { name: "Gönder" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("geçici olarak yeniden başlatılıyor");
    expect(document.querySelector(".message-user")).not.toBeInTheDocument();
    expect(composer).toHaveValue("Tekrar gönderilebilir mesaj");
  });
});
