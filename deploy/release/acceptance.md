# ScentAI V4 Release Acceptance

Behavioral release candidate: `scentai-v1.0-rc1`

Deployment packaging revision: `scentai-v1.0-rc2` (Modal Stage 6; no pipeline behavior change)

The frozen V4 evaluation contains 120 cases and all 120 functional cases pass. Language,
requested count, hard filters, entity resolution, unsupported routes, conversation no-repeat,
and performance calibration are all at 100%.

The only missed numerical gate is fallback rate: 8/120 (6.67%) versus a 5% target. These are
validated deterministic fallback answers, not unsafe outputs. The release candidate accepts this
as an operational quality exception; fallback rate remains a production metric and optimization
target.

The full Gemma 4 LoRA adapter is an external required artifact because it currently lives in
Google Drive. Deployment startup rejects the wrong base model, rank, DoRA mode, target modules,
or missing weights before vLLM starts.
