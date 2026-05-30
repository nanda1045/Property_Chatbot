# Aker Property Assistant

Property-scoped AI chatbot prototype for answering questions about a selected Aker property, such as `115r`.

The assistant combines structured rent-roll data in MySQL with scraped public property website content. It supports runtime LLM switching, Markdown responses, streamed LLM output, property-scoped retrieval, and structured UI components such as KPI cards, charts, tables, and comparisons.

## Features

- Property-scoped chat by active `property_code`.
- Structured rent-roll analytics from MySQL.
- Unstructured website retrieval from scraped property pages.
- Hybrid retrieval using Chroma vector search, BM25 keyword search, and reciprocal rank fusion.
- Metadata filtering by `property_code` and optional page type.
- Runtime model switching through the UI and API.
- Markdown answers with source citations.
- Streamed responses for real LLM calls.
- Embedded UI components for KPIs, trends, charge breakdowns, vacant units, balances, and comparisons.
- LLM-assisted tool planning with backend validation and server-side `property_code` injection.
- Safe SQL approval workflow for custom structured rent-roll questions not covered by predefined tools.
- Golden dataset and evaluation scripts for retrieval and answer quality.

## Project Structure

```text
app/                         FastAPI backend, orchestration, tools, retrieval clients
frontend/                    React/Vite chatbot UI
scripts/                     Data loading, scraping, ingestion, and eval runners
Data/                        Structured input/output data and retrieval indexes
config/property_sources.json Property website source map
evals/                       Golden datasets and evaluation reports
tests/                       Unit and integration-style tests
docs/                        More detailed implementation notes
```

## Setup

Prerequisites:

- Python 3.12+
- `uv`
- Node.js 20+
- Docker Desktop or Docker Engine

### Docker-First Local Setup

If you want the easiest local setup, use Docker for MySQL and run the backend/frontend on your machine.
You do not need to install a local MySQL server or the `mysql` command-line client for this path; the loader connects to the Docker database through the Python MySQL connector installed by `uv sync`.

1. Install Docker Desktop (macOS/Windows) or Docker Engine (Linux):

- macOS/Windows: https://www.docker.com/products/docker-desktop/
- Linux: https://docs.docker.com/engine/install/

2. Install `uv` (Python package manager):

```bash
brew install uv
```

Windows (PowerShell):

```powershell
winget install --id Astral.uv -e
```

If Python is already installed:

```bash
pip install uv
```

3. Open the project folder in a terminal:

```bash
cd /path/to/AKER_Chatbot
```

4. Create the `.env` file, then copy the example values:

```bash
touch .env
cp .env.example .env
```

5. Add any real model keys in .env you want to use:

```bash
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GROQ_API_KEY=...
```

6. Install Python dependencies:

```bash
uv sync
```

7. Start the MySQL container in new terminal:

```bash
docker compose up -d mysql
```

8. Wait for MySQL to be healthy, then load the structured rent-roll data:

```bash
uv run python scripts/load_rent_roll_mysql.py --reset
```

The loader reads the rent-roll Excel files in `Data/RentRoll_LeaseCharges_NamesRedacted copy/` and creates normalized MySQL tables keyed by `property_code`.

9. First-time setup only: scrape websites and build retrieval indexes:

```bash
uv run python scripts/scrape_property_sites.py
uv run python scripts/ingest_unstructured.py --reset
```

10. In a second terminal, start the backend:

```bash
uv run aker-api
```

11. In a third terminal, start the frontend:

```bash
cd frontend
npm install
npm run dev
```

12. Open the app in your browser:

```text
http://127.0.0.1:5173/
```

### Unstructured Data

New users should run the scraper and ingestion steps at least once before starting the app so website questions have data to search.

To re-scrape public property websites:

```bash
uv run python scripts/scrape_property_sites.py
```

To rebuild retrieval indexes:

```bash
uv run python scripts/ingest_unstructured.py --reset
```

To manually test scoped retrieval:

```bash
uv run python scripts/search_unstructured.py "EV charging bike storage" --property-code 115r --page-type amenities
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## API Examples

Blocking chat response:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "property_code": "115r",
    "model": "anthropic:claude-haiku-4-5-20251001",
    "message": "What is the latest occupancy and market rent?"
  }'
```

Streaming chat response:

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "property_code": "115r",
    "model": "anthropic:claude-haiku-4-5-20251001",
    "message": "Give me a concise executive summary of this property."
  }'
```

## Architecture Overview

The system is organized as a scoped retrieval and orchestration pipeline.

```mermaid
flowchart LR
  User["User"] --> UI["React Chat UI"]
  UI --> API["FastAPI"]
  API --> Orchestrator["Backend Orchestrator"]
  Orchestrator --> Planner["LLM Planner"]
  Orchestrator --> Tools["Tools"]
  Tools --> MySQL["MySQL (Rent-Roll)"]
  Tools --> Retrieval["Website Retrieval"]
  Orchestrator --> SQLApproval["SQL Draft + Approval"]
  Orchestrator --> LLM["LLM Answer"]
  Orchestrator --> Response["Response + UI Components"]
  Response --> UI
```

1. The user selects a property in the React UI.
2. The frontend sends `property_code`, selected `model`, and the user message to FastAPI.
3. The backend loads the selected property profile and normalizes the active `property_code`.
4. The orchestrator creates `LLMToolPlanner`. The planner first applies deterministic guardrails for ambiguity, PII, unsafe SQL, unsupported external data, and cross-property requests.
5. For real LLM models, the planner can classify the request as `structured`, `retrieval`, `hybrid`, `sql_approval`, `unsupported`, or `clarification`.
6. If the LLM planner cannot return a valid plan, or if the mock model is used in tests, the system falls back to deterministic planning.
7. Tool names are validated against an allowlist; property scoping is injected server-side, never trusted from the LLM.
8. Common structured analytics are routed to bounded SQL-backed tools such as latest KPIs, occupancy trend, charge breakdown, top balances, vacant units, rent by unit type, and rent vs lease charges.
9. Website questions are routed to property-scoped retrieval over scraped website chunks.
10. Custom structured metrics that are not covered by predefined tools can route to `sql_approval`.
11. In `sql_approval`, the LLM drafts a read-only SQL query with `:property_code`; it does not execute SQL.
12. The backend validates SQL drafts before they reach the UI. The guard checks allowed tables and columns, blocks PII, blocks unsafe operations, requires active-property scoping, rejects comments/semicolons/UNION, and requires row limits for row-level queries.
13. Valid SQL drafts are shown in the UI for user approval before execution.
14. Approved SQL is executed only through the backend approval endpoint, which binds the active `property_code` server-side.
15. Every structured SQL query is filtered by active `property_code`.
16. Every retrieval query is filtered by active `property_code` metadata.
17. Retrieval uses Chroma vector search plus BM25 keyword search, fused with reciprocal rank fusion.
18. Retrieved chunks are annotated with evidence confidence before being used in the answer.
19. The API returns Markdown, sources, tool results, and structured UI component definitions.
20. The React UI renders the Markdown and component payloads as chat messages, KPI cards, charts, tables, comparisons, SQL approval cards, and source links.

## Design Decisions

MySQL was used for rent-roll data because the source files are structured and naturally relational. The schema separates property metadata, reports, summary groups, unit-level rows, and charge summaries. This makes analytical queries explicit, auditable, and property-scoped.

The public website content is treated as unstructured data and ingested into a retrieval layer. Chunks are created using HTML section-aware chunking so amenities, floorplans, fees, and page sections stay more coherent than arbitrary fixed-size chunks.

Hybrid retrieval was chosen over vector-only retrieval because property websites contain exact terms such as `EV charging`, `A07`, `bike storage`, and charge/floorplan labels. BM25 helps exact-match queries, while Chroma handles paraphrases. Reciprocal rank fusion combines both without adding a heavy search dependency.

The orchestrator does not let the LLM execute arbitrary actions. The LLM can help plan the route and draft SQL for approval, but the backend validates the plan, validates tool names, injects property scope, and controls execution.

Structured analytics first use predefined SQL-backed tools for common metrics, such as latest KPIs, occupancy trend, top balances, vacant units, charge breakdown, and average market rent by unit type. These are preferred because they are tested and scoped by design.

For custom structured metrics, the planner can route to a controlled SQL approval workflow. The LLM drafts a candidate read-only query with a `:property_code` placeholder, but the backend does not execute it immediately. The draft must pass validation for approved tables and columns, active-property filtering, no PII fields, no unsafe operations, and row limits. Only after approval does the backend bind the active property code and execute the query.

The LLM is used for natural-language synthesis, not as the source of truth. Numeric facts come from MySQL tools, website facts come from retrieved chunks, and UI components are generated from structured tool outputs.

Streaming is implemented with server-sent events on `/chat/stream`. Real LLM token output appears progressively, and the final event includes complete Markdown, sources, and UI components.

## Property Scoping

Property scoping is enforced in multiple places:

- The frontend always sends an active `property_code`.
- MySQL repository methods include `WHERE property_code = %s`.
- Chroma and BM25 retrieval both filter by `property_code`.
- LangChain tools require `property_code` as an input.
- The orchestrator passes only active-property tool results to the LLM.
- LLM-drafted SQL must use `:property_code`; the backend binds the active property code only after approval.
- Cross-property or all-property requests are blocked before tool execution.
- If the user mentions another property while a different property is selected, the assistant adds an inline scope note and still answers only for the selected property.

## Supported Query Types

Examples the assistant is designed to handle:

- latest occupancy, market rent, lease charges, and vacant count
- executive summary
- occupancy trend over available months
- rent vs lease charge comparison
- charge category breakdown
- top balances
- vacant units and bedroom categories
- average market rent by bedroom category and floorplan code
- custom structured SQL approval questions, such as lowest market rents, unit counts by unit type, or total market rent by unit type
- website amenities and apartment features
- EV charging, bike storage, parking, and other website-supported facts
- floorplans advertised on the website
- property location
- unavailable years, such as asking for 2024 when only 2025 data exists
- ambiguous short prompts, such as `charges`
- no-evidence website questions, such as reviews when reviews were not scraped

## Evaluations

Run the local test suite:

```bash
uv run pytest
```

Some integration-style tests are skipped unless MySQL and the representative test data are available locally.

Run the golden retrieval and generation dataset:

```bash
uv run python scripts/run_golden_evals.py --output-json evals/golden_report.json
```

Optional LLM-judged metrics:

```bash
uv run python scripts/run_llm_judge_evals.py --output-json evals/llm_judge_report.json
```

Evaluation coverage includes:

- property-scope isolation
- retrieval relevance
- retrieval precision@k
- MRR
- NDCG@k
- evidence recall
- answer faithfulness
- answer relevancy
- completeness
- citation quality
- planner routing and response behavior for supported, unsupported, hybrid, and SQL approval queries

## Assumptions

- The provided rent-roll Excel files are the source of truth for structured property facts.
- The available structured data currently covers the loaded report months only.
- Public property websites are acceptable sources for unstructured content.
- The user selects one active property at runtime, and answers should stay scoped to that property.


## Tradeoffs

- Chroma is simple to run locally and good for a prototype, but it is not a managed production vector database.
- BM25 is stored locally with SQLite, which is lightweight but not designed for large multi-tenant search workloads.
- The included local retrieval indexes make demos faster, but they can also be rebuilt from source data.
- Intent routing uses deterministic guardrails plus an LLM planner. This is safer than fully agentic tool calling, but new query families may still require examples, planner rules, or new predefined tools.
- The LLM is not given unrestricted tool access. This reduces risk but makes the orchestration layer more explicit.
- Structured analytics use curated SQL-backed tools for known metrics and a guarded SQL approval workflow for custom metrics. This improves safety, testability, and property scoping, but it is less automatic than unrestricted agentic SQL execution.
- The frontend renders a curated set of UI components rather than arbitrary LLM-generated HTML, which is safer but less flexible.

## Limitations

- Conversation memory is intentionally limited. The system handles many follow-ups, but it is not a full long-term conversational memory system.
- Website content is only as complete as the latest scrape.
- If a website hides data behind JavaScript or external APIs, the scraper may not capture every detail.
- Real LLM calls require valid API keys and available credits.
- The prototype is designed for local development, not production deployment.
- There is no authentication or multi-user authorization layer.
- MySQL must be running and loaded before structured-data questions will work.
- The assistant does not execute custom database metrics automatically. Custom structured metrics can produce a proposed read-only SQL query, but the user must approve it before execution.
- The SQL approval guard is intentionally strict. Complex valid SQL may be rejected if it cannot prove every referenced table is scoped to the active property or if it references columns outside the allowlist.
- The SQL approval route improves flexibility, but production-grade analytics would still benefit from a formal metric catalog and more validated tools for metric families such as renewal trends, bad debt percentage, lease expirations, or move-in/move-out analytics.
- If retrieval indexes are deleted, run `scripts/ingest_unstructured.py` again before testing website questions.

## Packaging Notes

Do not include local secrets in the final zip:

- exclude `.env`
- exclude `.venv/`
- exclude `frontend/node_modules/`
- exclude `frontend/dist/`
- exclude Python caches and test caches

Include `.env.example`, source code, scripts, docs, config, tests, eval files, and the representative data needed to reproduce the prototype.

## Additional Docs

- `docs/structured-data-load.md`
- `docs/unstructured-source-plan.md`
- `docs/retrieval-ingestion.md`
- `docs/langchain-orchestration.md`
- `docs/frontend-ui.md`
- `docs/evals.md`
