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

### Novita-hosted DeepSeek-OCR-2 + DeepSeek V4.1 Flash

Set `CLASSIFICATION_PROVIDER=DEEPSEEK` and/or `REVIEW_PROVIDER=DEEPSEEK`. Every PNG/JPEG and **every PDF page** is rendered as an image and sent first to [Novita's DeepSeek-OCR-2](https://novita.ai/models/model-detail/deepseek-deepseek-ocr-2). Only OCR text, context and requirement IDs are sent to [Novita's DeepSeek V4.1 Flash](https://novita.ai/models/model-detail/deepseek-deepseek-v4.1-flash). Both are separate models behind Novita's OpenAI-compatible API; Flash never receives document images or PDF bytes. Neither model is bundled into the Agent image.

Required environment: `NOVITA_API_KEY` (used by both calls). Default chat URL: `https://api.novita.ai/openai/v1/chat/completions`; default health URL: `https://api.novita.ai/openai/v1/models`. Default models: `deepseek/deepseek-ocr-2` and `deepseek/deepseek-v4.1-flash`. `OCR_API_URL`, `OCR_HEALTH_URL`, `OCR_API_KEY`, `OCR_MODEL`, `MODEL_API_URL`, `MODEL_HEALTH_URL`, `MODEL_API_KEY`, `MODEL_NAME` are optional per-model overrides; explicit model/OCR keys take precedence over `NOVITA_API_KEY`. Do not put secrets in Git, Dockerfiles, URLs or shell history. Use HTTPS for any provider endpoint outside a trusted private network.

`GET /health/ready` checks both services. OCR failure is reported as `OCR_NOT_CONFIGURED`, `OCR_UNAVAILABLE`, `OCR_TIMEOUT` or `OCR_INVALID_RESPONSE`; Flash has the corresponding `MODEL_*` errors. The backend may retry or hand work to a person without changing business state. PDFs are limited to 20 pages, images to 16 megapixels, and total OCR text to 120,000 characters per analysis; limits fail explicitly, without silently skipping pages. OCR/Flash responses are not logged. Agent validates schema, supplied document IDs and Decimal arithmetic; Backend validates document ownership. Factual conclusions still require case-level evaluation; REVIEW findings do not contain confidence. `model_version` is stamped from the configured request model, never from generated JSON.

For a reusable service that does not share Folio's document volume, set `AGENT_API_KEY` and call `POST /v1/analyze-inline` with `Authorization: Bearer <AGENT_API_KEY>` and `Idempotency-Key: <run_id>:<turn>`. It accepts the same `CLASSIFY` or `REVIEW` envelope as `/v1/analyze`, but each document replaces `storage_key` with `content_base64` (complete PDF/PNG/JPEG). `sha256` and MIME signature must match. Responses and `snake_case` schemas are identical. Keep this endpoint behind HTTPS and an appropriate private gateway/rate limit if exposed outside Compose. It is disabled with `503 AGENT_AUTH_NOT_CONFIGURED` until its API key is set. Folio continues to use the read-only-volume `/v1/analyze` endpoint internally.

The OCR→Flash path has been exercised against the real Novita API and synthetic v5 accounting cases. Run `REVIEW_PROVIDER=DEEPSEEK uv run python -m scripts.evaluate_v5 --phase trajectory CASE_DIR` to reproduce the dataset's staged evidence; `--phase initial|complete` checks a single review stage. The evaluator mirrors Backend's REVIEW context fields and defaults, but never places `task.json` instructions, target transactions or `ground_truth.json` answers in the model request. All phases still reconstruct document availability from the dataset rather than the Backend database, so use an actual Backend request for end-to-end workflow acceptance. REVIEW sends no original filenames to Flash, so dataset filenames cannot hint at expected errors. `REMOTE` remains the prior provisional complete-file provider mode; it does **not** use the OCR→Flash pipeline.

Copy `.env.example` only for direct local execution. In Compose set the values in the backend deployment environment file. Never commit credentials.

`CLASSIFICATION_PROVIDER=DISABLED` is the default. For local acceptance set `AGENT_CLASSIFICATION_PROVIDER=MOCK` and `ENVIRONMENT=development` in the backend Compose environment. MOCK reads complete files, uses deterministic name/content keywords and returns `mock-classifier-v1`; it is not trained-model inference and is rejected in production. Backend returns the MOCK marker to the UI. `OFF` requests use manual categorization without calling Agent.

For `REMOTE`, `MODEL_API_URL` receives a Bearer-authenticated JSON POST: the CLASSIFY envelope uses `snake_case`; each document replaces `storage_key` with `content_base64` containing the **complete file**, and retains `document_id`, `original_name`, `content_type`, and `sha256`. Return the strict `ClassificationResponse` in `app/schemas.py`. Unknown/missing IDs, changed order, wrong run IDs, unknown targets, non-finite confidence and extra review fields fail closed. Batches are limited to 100 files / 100 MiB and responses to 1 MiB. This is a provisional transport contract; no real provider API or credentials have been supplied.

For the legacy `REMOTE` provider, `MODEL_HEALTH_URL` must explicitly identify a read-only health endpoint; no inference request is used as a health probe. Its transport expects Bearer-authorized GET and `200 {"status":"ok"}`. The Novita `DEEPSEEK` provider instead checks `/openai/v1/models` by default. MOCK readiness requires only the readable document mount and a non-production environment.

Health requests use the connect timeout (default 10s), reject redirects and oversized/non-JSON responses, and do not retry. REVIEW has a 290-second total OCR/inference budget; CLASSIFY keeps its 150-second budget. Backend owns durable retries and results. Agent caches the last 128 successful request fingerprints/results and the last 128 successful OCR texts by file hash within one process; restart/eviction may repeat inference. PDF rendering remains serial because concurrent native rendering is unsafe.

Agent emits JSON events to container stdout. Follow a run with `docker compose logs agent | grep <run_id>`: `analysis_started`, `document_read`, `ocr_document_started`, `ocr_page_started/finished`, `ocr_document_finished`, `model_call_started/finished`, `model_result_validation`, `review_search` or `review_finding`, `analysis_completed` or `analysis_failed`, and `request_finished`. Events include IDs, status, error code, counts and elapsed milliseconds; provider HTTP status is recorded separately. A started step without a finished step identifies the operation still in progress. Successful health probes are omitted to keep the log readable. Logs never include file names or bodies, OCR text, prompts, model responses, client messages, or keys.

## Post-submission review (A3)

Set `REVIEW_PROVIDER=DISABLED|MOCK|REMOTE|DEEPSEEK` (`AGENT_REVIEW_PROVIDER` in Compose). The default is DISABLED. MOCK is rejected in production and returns `mock-reviewer-v1`; filenames containing G02/G03/B01/B03/F02/F04/F05 select fixed demonstration scenarios, not predictions. Ordinary files are escalated with unknown fields rather than invented extraction results. Scenario amounts and entities are synthetic, not extracted facts.

`app/analysis_schemas.py` defines the REVIEW contract and is kept identical in Backend. Each review file may include the known `document_type` and positive `submission_round`; these are metadata hints, not proof of its contents. Backend sends submission rounds so a corrected file in a later client submission can supersede an earlier mismatch without deleting the old audit record. The remote POST uses the same complete-file transport as classification, plus request context, internal requirement hints, document associations, and up to three search turns. Backend normally sends empty requirement `instructions`; if an accountant explicitly enters a scope description, Agent must verify any stated transaction against the bank statement rather than treat that description as evidence. Dates, counterparties and amounts otherwise come from OCR of the submitted files, not dataset tasks. Without a target, the Agent must assess the requested scope rather than guess one transaction from a whole statement.

For incomplete bank evidence, Agent searches authorized current and historical material before contacting the client. Open-items registers are secondary records and cannot alone prove the underlying obligation. Only Backend executes searches and supplies the next turn. Final output must cover every supplied document and requirement. Amount operands carry document IDs and decimal strings; `SUM/SUBTRACT/MULTIPLY` allow multi-invoice, fee, conversion and retention relationships. Agent checks exact arithmetic, limits any SATISFY rounding difference to half a cent, and requires a structured FX conversion when cited foreign-currency evidence is present; an invalid response receives up to two concise correction attempts before failing closed. Backend independently verifies arithmetic, monetary differences and evidence ownership. OCR results are reused across search turns in the same process.

Backend sends `review_preference=CAUTIOUS|STANDARD|EFFICIENT` as a trusted REVIEW policy. It guides the model's choice between a specific `ASK_CLIENT` correction and `ESCALATE` to an accountant; it never relaxes factual or evidence requirements. Backend may automatically apply validated `SATISFY` or `REQUEST_ACTION` item decisions. A mismatch needs contradictory evidence; missing/incomplete support needs a prior search; unreadable or vague issues remain manual. Any automatic rejection returns the request to the client; an all-pass round remains `IN_REVIEW` until an accountant approves it. Human evidence override is supported, while waiver and whole-request approval remain human-only. Notification outbox records are currently suppressed rather than delivered. REMOTE transport is provisional.

## Publishing

GitHub Actions tests and builds on PRs; main, version tags and manual dispatch may publish an immutable SHA image. Configure `AWS_ROLE_ARN`, `AWS_REGION=ap-southeast-1`, `ECR_AGENT_REPOSITORY=acc-system-agent`. The backend deployment directory contains the corresponding ECR/OIDC policy templates. Set `AGENT_IMAGE=<registry>/acc-system-agent:<git-sha>` on deployment.
