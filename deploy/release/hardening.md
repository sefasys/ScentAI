# Stage 7C: Anonymous Beta Hardening

Stage 7C keeps ScentAI account-free and payment-free while making the public beta safer and more
predictable. Final visual design is deliberately outside this stage.

## Delivered behavior

- The current conversation survives browser restarts in local browser storage.
- Existing Stage 7B tab-only conversations migrate automatically.
- Server-side `session_id` continuity remains in place; an expired Modal session is retried once as
  a fresh session without duplicating the visible user message.
- Failed or cancelled submissions restore the composer text and remove the unsent message bubble.
- Chat polling reports queued/running phases and retries temporary network or 502/503/504 failures.
- Cold start shows elapsed time and coarse startup phases instead of an indefinite loading message.
- `/api/scentai/health/live` is answered by the Cloudflare gateway and does not wake the Modal GPU.
- Same-origin enforcement, payload validation, secret injection, security headers, and Cloudflare
  rate limits remain active.

The locally retained conversation is not a cloud account history. If Modal has discarded its
in-memory session, old messages remain visible but the next turn starts a new server context.

## Verification

```bash
cd frontend
npm run check
```

The Stage 7C release passes 17 apps/web/gateway tests and a production TypeScript/Vite build.
The public smoke report is stored at `deploy/reports/stage7c_public_smoke.json`.

Production acceptance on 2026-07-20:

- gateway: `https://scentai-web.sefasys.workers.dev`
- cold warm-up: 367.0 seconds
- chat job: 35.8 seconds
- grounded validation: passed
- entity resolution: `YSL Y EDP` resolved to `Y Eau de Parfum by Yves Saint Laurent`

## Deferred by decision

- optional user accounts and cross-device history
- exact per-user quotas or billing
- final visual identity and frontend redesign
