# Stage 7B: Public Web Gateway

Stage 7B publishes the functional Stage 7A chat without exposing the Modal API key to browsers.
Cloudflare Workers serves both the Vite assets and the `/api/scentai` gateway.

## One-time Cloudflare setup

Run these commands from `apps/web/`:

```bash
npm install
npx wrangler login
npx wrangler secret put SCENTAI_API_KEY
```

Use the same `SCENTAI_API_KEY` configured in the Modal `scentai-api` secret. Never commit it or put
it in a `VITE_*` variable. Enter it interactively so it is not written to shell history:

```bash
npx wrangler secret put SCENTAI_API_KEY
```

## Verify and deploy

```bash
npm run check
npx wrangler deploy --dry-run
npm run deploy
```

Wrangler prints the public `workers.dev` URL after deployment. Open that URL and verify:

1. No API connection form or API key field is shown.
2. A cold runtime displays the warm-up gate and eventually opens the composer.
3. A warm request returns normally.
4. Refreshing the page restores the current tab's conversation.
5. Browser network requests target only `/api/scentai/...`, not the Modal hostname.

The reusable end-to-end smoke starts the runtime through the public gateway and sends one entity
resolution query without reading or transmitting a client-side API key:

```bash
python deploy/scripts/stage7b_public_smoke.py \
  --url https://scentai-web.sefasys.workers.dev
```

## Gateway policy

- Modal credentials exist only as the Worker secret `SCENTAI_API_KEY`.
- The Worker accepts only health, asynchronous warm-up/chat jobs, and session deletion.
- Chat JSON is reduced to `query` and an optional validated `session_id`; browser-supplied `debug`
  and API key values are discarded.
- Chat request bodies are capped at 8 KiB and queries at 2,000 characters.
- Cross-origin write requests are rejected.
- Rate-limit namespaces are separate for chat creation, runtime warm-up, and polling.
- Responses carry CSP, frame, MIME, referrer, permissions, and HSTS headers.

Current limits are public-beta cost controls, not user entitlements. Account-aware quotas and durable
conversation storage should be introduced together in a later stage.
