const API_PREFIX = "/api/scentai";
const MAX_CHAT_BODY_BYTES = 8_192;
const MAX_QUERY_LENGTH = 2_000;
const ID_PATTERN = /^[A-Za-z0-9_-]{1,128}$/;

interface RateLimitBinding {
  limit(options: { key: string }): Promise<{ success: boolean }>;
}

interface AssetBinding {
  fetch(request: Request): Promise<Response>;
}

export interface Env {
  ASSETS: AssetBinding;
  SCENTAI_API_URL: string;
  SCENTAI_API_KEY: string;
  CHAT_RATE_LIMITER: RateLimitBinding;
  WARMUP_RATE_LIMITER: RateLimitBinding;
  POLL_RATE_LIMITER: RateLimitBinding;
}

type RouteKind = "chat-create" | "warmup-create" | "poll";

interface GatewayRoute {
  kind: RouteKind;
  upstreamPath: string;
}

const SECURITY_HEADERS: Readonly<Record<string, string>> = {
  "Content-Security-Policy":
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; " +
    "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Resource-Policy": "same-origin",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
};

function gatewayHealthResponse(): Response {
  return withSecurityHeaders(
    new Response(JSON.stringify({ status: "ok", service: "scentai-web-gateway" }), {
      status: 200,
      headers: {
        "Cache-Control": "no-store",
        "Content-Type": "application/json; charset=utf-8",
      },
    }),
  );
}

function withSecurityHeaders(response: Response): Response {
  const secured = new Response(response.body, response);
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) {
    secured.headers.set(name, value);
  }
  return secured;
}

function jsonResponse(status: number, detail: string, extraHeaders?: HeadersInit): Response {
  const headers = new Headers(extraHeaders);
  headers.set("Content-Type", "application/json; charset=utf-8");
  headers.set("Cache-Control", "no-store");
  return withSecurityHeaders(new Response(JSON.stringify({ detail }), { status, headers }));
}

function exactIdPath(pathname: string, prefix: string): boolean {
  if (!pathname.startsWith(`${prefix}/`)) return false;
  return ID_PATTERN.test(pathname.slice(prefix.length + 1));
}

export function matchGatewayRoute(pathname: string, method: string): GatewayRoute | null {
  if (!pathname.startsWith(API_PREFIX)) return null;
  const upstreamPath = pathname.slice(API_PREFIX.length) || "/";

  if (upstreamPath === "/v1/chat/jobs" && method === "POST") {
    return { kind: "chat-create", upstreamPath };
  }
  if (exactIdPath(upstreamPath, "/v1/chat/jobs") && method === "GET") {
    return { kind: "poll", upstreamPath };
  }
  if (upstreamPath === "/v1/runtime/warmup/jobs" && method === "POST") {
    return { kind: "warmup-create", upstreamPath };
  }
  if (exactIdPath(upstreamPath, "/v1/runtime/warmup/jobs") && method === "GET") {
    return { kind: "poll", upstreamPath };
  }
  if (exactIdPath(upstreamPath, "/v1/sessions") && method === "DELETE") {
    return { kind: "poll", upstreamPath };
  }
  return null;
}

function hasKnownGatewayPath(pathname: string): boolean {
  if (!pathname.startsWith(API_PREFIX)) return false;
  const upstreamPath = pathname.slice(API_PREFIX.length) || "/";
  return (
    upstreamPath === "/v1/chat/jobs" ||
    exactIdPath(upstreamPath, "/v1/chat/jobs") ||
    upstreamPath === "/v1/runtime/warmup/jobs" ||
    exactIdPath(upstreamPath, "/v1/runtime/warmup/jobs") ||
    exactIdPath(upstreamPath, "/v1/sessions")
  );
}

async function actorKey(request: Request, scope: RouteKind): Promise<string> {
  const identity = request.headers.get("CF-Connecting-IP") ?? "local-development";
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(identity));
  const fingerprint = Array.from(new Uint8Array(digest).slice(0, 12), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
  return `${scope}:${fingerprint}`;
}

async function enforceRateLimit(
  request: Request,
  env: Env,
  kind: RouteKind,
): Promise<Response | null> {
  const limiter =
    kind === "chat-create"
      ? env.CHAT_RATE_LIMITER
      : kind === "warmup-create"
        ? env.WARMUP_RATE_LIMITER
        : env.POLL_RATE_LIMITER;
  const { success } = await limiter.limit({ key: await actorKey(request, kind) });
  if (success) return null;
  return jsonResponse(429, "Too many requests. Please wait before trying again.", {
    "Retry-After": "60",
  });
}

function sameOriginWrite(request: Request): boolean {
  const origin = request.headers.get("Origin");
  return !origin || origin === new URL(request.url).origin;
}

async function sanitizedChatBody(request: Request): Promise<string | Response> {
  const declaredLength = Number(request.headers.get("Content-Length") ?? "0");
  if (Number.isFinite(declaredLength) && declaredLength > MAX_CHAT_BODY_BYTES) {
    return jsonResponse(413, "Request body is too large.");
  }
  if (!request.headers.get("Content-Type")?.toLowerCase().startsWith("application/json")) {
    return jsonResponse(415, "Content-Type must be application/json.");
  }

  const raw = await request.text();
  if (new TextEncoder().encode(raw).byteLength > MAX_CHAT_BODY_BYTES) {
    return jsonResponse(413, "Request body is too large.");
  }

  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return jsonResponse(400, "Request body must contain valid JSON.");
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return jsonResponse(400, "Request body must be a JSON object.");
  }

  const body = value as Record<string, unknown>;
  const query = typeof body.query === "string" ? body.query.trim() : "";
  if (!query || query.length > MAX_QUERY_LENGTH) {
    return jsonResponse(422, `Query must contain between 1 and ${MAX_QUERY_LENGTH} characters.`);
  }
  const sessionId = body.session_id;
  if (sessionId !== undefined && (typeof sessionId !== "string" || !ID_PATTERN.test(sessionId))) {
    return jsonResponse(422, "session_id has an invalid format.");
  }

  return JSON.stringify({
    query,
    ...(typeof sessionId === "string" ? { session_id: sessionId } : {}),
  });
}

function upstreamUrl(env: Env, path: string): URL {
  const base = new URL(env.SCENTAI_API_URL);
  if (base.protocol !== "https:") {
    throw new Error("SCENTAI_API_URL must use HTTPS.");
  }
  return new URL(path, `${base.origin}/`);
}

async function proxyApi(request: Request, env: Env, route: GatewayRoute): Promise<Response> {
  if (request.method !== "GET" && !sameOriginWrite(request)) {
    return jsonResponse(403, "Cross-origin write requests are not allowed.");
  }

  const limited = await enforceRateLimit(request, env, route.kind);
  if (limited) return limited;

  let body: string | undefined;
  if (route.kind === "chat-create") {
    const sanitized = await sanitizedChatBody(request);
    if (sanitized instanceof Response) return sanitized;
    body = sanitized;
  }

  const headers = new Headers({
    Accept: "application/json",
    "X-API-Key": env.SCENTAI_API_KEY,
  });
  if (body !== undefined) headers.set("Content-Type", "application/json");

  let upstreamResponse: Response;
  try {
    upstreamResponse = await fetch(upstreamUrl(env, route.upstreamPath), {
      method: request.method,
      headers,
      body,
      redirect: "manual",
      signal: request.signal,
    });
  } catch {
    return jsonResponse(502, "ScentAI upstream service is unavailable.");
  }

  const responseHeaders = new Headers({
    "Cache-Control": "no-store",
    "Content-Type": upstreamResponse.headers.get("Content-Type") ?? "application/json; charset=utf-8",
  });
  for (const name of ["Retry-After", "X-Request-ID"]) {
    const value = upstreamResponse.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }
  return withSecurityHeaders(
    new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      headers: responseHeaders,
    }),
  );
}

export async function handleRequest(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  if (url.pathname === `${API_PREFIX}/health/live` && request.method === "GET") {
    return gatewayHealthResponse();
  }
  if (url.pathname.startsWith(API_PREFIX)) {
    const route = matchGatewayRoute(url.pathname, request.method);
    if (!route) {
      const knownPath = hasKnownGatewayPath(url.pathname);
      return jsonResponse(
        knownPath ? 405 : 404,
        knownPath ? "Method not allowed." : "API route not found.",
        knownPath ? { Allow: "GET, POST, DELETE" } : undefined,
      );
    }
    return proxyApi(request, env, route);
  }
  return withSecurityHeaders(await env.ASSETS.fetch(request));
}

export default {
  fetch: handleRequest,
};
