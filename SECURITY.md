# Security

Do not report secrets through a public issue. Contact me(sefasys) privately for vulnerabilities involving deployed credentials or endpoints.

## Secret Handling

- API keys belong in Modal or Cloudflare secrets, never in `VITE_*` variables.
- Local secrets belong in `.env`, `.env.local`, or `.dev.vars`; all are ignored.
- The browser communicates through the same-origin Worker gateway.
- Debug payloads are disabled for public clients.


