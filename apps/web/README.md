# ScentAI Frontend

Stage 7B serves the React client and a same-origin API gateway from one Cloudflare Worker. The
browser calls `/api/scentai`; the Worker injects the private Modal API key and proxies only the
allowlisted asynchronous chat, warm-up, and session routes. Gateway liveness is served locally so
health checks cannot keep the Modal GPU awake.

The account-free persistence and reliability work added after deployment is documented in
[`hardening.md`](../../deploy/release/hardening.md).

## Public architecture

```text
Browser -> Cloudflare Worker /api/scentai -> Modal Stage 6 API -> ScentAI model/retrieval
        -> Cloudflare static assets       -> React application
```

The Modal API key is never bundled into JavaScript, returned to the browser, or stored in browser
storage. The Worker also enforces same-origin writes, payload limits, route/method allowlisting,
security headers, and separate chat/warm-up/poll rate limits.

## Install and verify

```bash
npm install
npm run check
npx wrangler deploy --dry-run
```

`npm run check` runs all apps/web/API gateway tests and creates the production Vite build.

## Local end-to-end development

```bash
cp .dev.vars.example .dev.vars
# Put the Modal SCENTAI_API_KEY in .dev.vars.
npm run cf:dev
```

Wrangler serves the built SPA and local Worker together. `.dev.vars` is gitignored.

For direct API diagnostics only, create `.env.local` with the Modal URL and enable
`VITE_SCENTAI_DIRECT_CONNECTION=true`, then run `npm run dev`. This restores the Stage 7A
connection form; it is deliberately disabled in public builds.

## Production deployment

```bash
npx wrangler login
npx wrangler secret put SCENTAI_API_KEY
npm run deploy
```

The Worker name, Modal URL, static asset binding, required secret, and rate-limit bindings live in
`wrangler.jsonc`. Set the secret before the final deploy. Do not place it in a `VITE_*` variable.

## Current behavior

- Public frontend starts runtime warm-up before opening the composer.
- Chat and warm-up use browser-safe asynchronous jobs.
- Multi-turn state uses the server `session_id`. The current conversation is also retained locally
  across browser restarts, without an account or cloud-side message archive.
- If Modal scale-down invalidates a session, the current message is retried once as a fresh session.
- New conversation deletes the active server session when it still exists.
- Temporary polling failures are retried without submitting a duplicate model job.
- Public chat creation is limited to 6 requests/minute per network identity; warm-up to 3/minute;
  polling has its own 180/minute budget so cold starts do not self-throttle.

Cross-device history, optional user accounts, and final visual design remain later work. Account and
payment systems are not required for the anonymous beta. Cloudflare's rate-limit binding is an abuse
and cost guard, not a billing or exact quota ledger.
