# React Chat UI

The prototype includes a Vite + React frontend under `frontend/`.

## Responsibilities

- Fetch available properties from `/properties`.
- Fetch available runtime model options from `/models`.
- Send chat requests to `/chat` with `property_code`, `model`, and `message`.
- Render `answer_markdown` with Markdown support.
- Render backend-provided `components` deterministically.
- Render retrieved website `sources`.

## Component Contract

The backend returns UI components as JSON:

```json
{
  "type": "kpi_card",
  "title": "Occupancy",
  "data": {
    "value": 94.8,
    "unit": "%",
    "report_month": "2024-04"
  }
}
```

Supported component types:

- `kpi_card`
- `bar_chart`
- `line_chart`
- `table`
- `comparison_view`

This keeps data generation in the backend/tool layer and keeps visual rendering in React.

## Run

Start the backend first:

```bash
uv run uvicorn app.main:app --reload
```

Then run the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```
