# acc-system-agent

Read-only AI integration service. Backend owns all business state, jobs and retries. Agent has no database or Redis dependencies.

## Local acceptance

```bash
uv sync --frozen
uv run pytest -q
docker build --build-arg APP_VERSION=b6.3-a3-local -t acc-system-agent:local .
```

Start through `acc-system-backend/deploy/compose.yml`. There is no public port; call Agent from the backend container over `http://agent:8000`.

- `GET /health/live`: 200 means the Agent process is running.
- `GET /health/ready`: verifies the documents mount and remote health endpoint. Without model configuration, expect `503 MODEL_NOT_CONFIGURED`; manual Backend workflows remain available.
- `POST /v1/analyze`: validates `schema_version`, `run_id`, document references and `Idempotency-Key: <run_id>:<turn>`. `CLASSIFY` returns only categories and target requirements. `REVIEW` returns structured extractions, findings, evidence and decimal-string amount relations, or a search request. Traversal, symlinks, non-regular files, excessive size, mismatched MIME and SHA-256 are rejected.

Documents are read from `DOCUMENT_PATH=/data/documents`, a read-only volume containing only scanned files. Each reference contains `document_id`, `storage_key`, `content_type` (PDF/PNG/JPEG) and lowercase hexadecimal `sha256`.

## Remote provider configuration

Copy `.env.example` only for direct local execution. In Compose set the values in the backend deployment environment file. Never commit credentials.

`CLASSIFICATION_PROVIDER=DISABLED` is the default. For local acceptance set `AGENT_CLASSIFICATION_PROVIDER=MOCK` and `ENVIRONMENT=development` in the backend Compose environment. MOCK reads complete files, uses deterministic name/content keywords and returns `mock-classifier-v1`; it is not trained-model inference and is rejected in production. Backend returns the MOCK marker to the UI. `OFF` requests use manual categorization without calling Agent.

For `REMOTE`, `MODEL_API_URL` receives a Bearer-authenticated JSON POST: the CLASSIFY envelope uses `snake_case`; each document replaces `storage_key` with `content_base64` containing the **complete file**, and retains `document_id`, `original_name`, `content_type`, and `sha256`. Return the strict `ClassificationResponse` in `app/schemas.py`. Unknown/missing IDs, changed order, wrong run IDs, unknown targets, non-finite confidence and extra review fields fail closed. Batches are limited to 100 files / 100 MiB and responses to 1 MiB. This is a provisional transport contract; no real provider API or credentials have been supplied.

`MODEL_HEALTH_URL` must explicitly identify a read-only health endpoint; no inference request is used as a health probe. The health transport expects Bearer-authorized GET and `200 {"status":"ok"}`. MOCK readiness requires only the readable document mount and a non-production environment.

Health requests use the connect timeout (default 10s), reject redirects and oversized/non-JSON responses, and do not retry. Classification uses the inference timeout (default 180s). Backend owns retries and durable results. Agent caches the last 128 successful request fingerprints/results within one process and serializes calls; restart/eviction may repeat inference, so a production provider must honor the forwarded idempotency key. Logs include IDs/status/timing, never file bodies, keys or provider responses.

## Post-submission review (A3)

Set `REVIEW_PROVIDER=DISABLED|MOCK|REMOTE` (`AGENT_REVIEW_PROVIDER` in Compose). The default is DISABLED. MOCK is rejected in production and returns `mock-reviewer-v1`; filenames containing G02/G03/B01/B03/F02/F04/F05 select fixed demonstration scenarios, not predictions. Ordinary files are escalated with unknown fields rather than invented extraction results. Scenario amounts and entities are synthetic, not extracted facts.

`app/analysis_schemas.py` defines the REVIEW contract and is kept identical in Backend. The remote POST uses the same complete-file transport as classification, plus request context, internal requirement hints, document associations, and up to three search turns. Agent returns `SEARCH_CURRENT/SEARCH_HISTORY`; only Backend searches authorized submitted documents and supplies the next turn. Final output must cover every supplied document and requirement. Amount operands carry document IDs and decimal strings; `SUM/SUBTRACT/MULTIPLY` allow multi-invoice, fee, conversion and retention relationships. Backend independently verifies arithmetic and evidence ownership.

Backend may automatically apply threshold-qualified `SATISFY` or `REQUEST_ACTION` item decisions after validating evidence and decimal arithmetic. Any automatic rejection returns the request to the client; an all-pass round remains `IN_REVIEW` until an accountant approves it. Human evidence override is supported, while waiver and whole-request approval remain human-only. Notification outbox records are currently suppressed rather than delivered. The real trained model API is still unconfigured; REMOTE transport is provisional.

## Publishing

GitHub Actions tests and builds on PRs; main, version tags and manual dispatch may publish an immutable SHA image. Configure `AWS_ROLE_ARN`, `AWS_REGION=ap-southeast-1`, `ECR_AGENT_REPOSITORY=acc-system-agent`. The backend deployment directory contains the corresponding ECR/OIDC policy templates. Set `AGENT_IMAGE=<registry>/acc-system-agent:<git-sha>` on deployment.
