# LangChain Tool Orchestration

The backend uses LangChain for tool definitions, prompt construction, and runtime model adapters.

## Why LangChain

LangChain gives the prototype a recognizable orchestration layer while keeping the case-study requirement visible: every tool requires an active `property_code`.

The implementation uses:

- `langchain_core.tools.tool` for tool schemas.
- `ChatPromptTemplate` for model prompts.
- `langchain-openai` and `langchain-anthropic` for runtime LLM switching.
- A deterministic router before model generation so every tool call receives the API-level `property_code`.

## Tools

Structured MySQL tools:

- `get_property_profile`
- `get_latest_property_kpis`
- `get_occupancy_trend`
- `get_charge_breakdown`
- `get_top_balances`
- `get_vacant_units`
- `get_rent_by_unit_type`

Unstructured hybrid retrieval tool:

- `search_property_content`

Each tool schema requires `property_code`, and SQL, Chroma, and BM25 queries also apply that same value as a filter.

## Runtime Model Switching

The `/chat` endpoint accepts:

```json
{
  "property_code": "115r",
  "model": "mock:mock-property-assistant",
  "message": "What is occupancy and what amenities are available?"
}
```

Supported model ID shape:

- `mock:mock-property-assistant`
- `openai:<model-name>`
- `anthropic:<model-name>`

The mock model is the default so the prototype runs without API keys.

## API

```bash
uv run uvicorn app.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "property_code": "115r",
    "model": "mock:mock-property-assistant",
    "message": "What is the latest occupancy and what amenities mention EV charging?"
  }'
```

The response includes:

- `answer_markdown`
- `components`
- `sources`
- `tool_results`
