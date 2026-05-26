# Evaluation Layer

The project includes deterministic evals for the three risk surfaces in the property assistant:

- Structured MySQL tools
- Property-scoped hybrid retrieval
- End-to-end `/chat` orchestration

## Run

Start MySQL and ingest retrieval indexes first:

```bash
docker compose up -d mysql
uv run python scripts/load_rent_roll_mysql.py --reset
uv run python scripts/ingest_unstructured.py --reset
```

Then run:

```bash
uv run pytest
```

## Eval Fixtures

- `evals/retrieval_cases.json`
- `evals/chat_cases.json`
- `evals/golden_cases.json`

## What Is Checked

Structured tests check:

- Known latest KPI facts for `115r`
- Charge breakdown availability
- Top balances sorted descending
- Occupancy trend ordering

Retrieval tests check:

- Chroma and BM25 counts match the current `Data/unstructured/property_chunks.jsonl`
- All retrieval results match the requested `property_code`
- Expected terms appear in the top results
- Expected page types are returned

Chat tests check:

- `/chat` returns Markdown
- Expected structured facts are present
- Expected UI component types are returned
- Tool outputs include expected keys
- Retrieval sources never leak another `property_code`

These evals are intentionally deterministic and use the mock model, so they can run without LLM API keys.

## Golden Dataset Runner

For a higher-signal pass/fail report across retrieval and generation, run:

```bash
uv run python scripts/run_golden_evals.py --output-json evals/golden_report.json
```

The golden dataset covers representative user workflows:

- Structured KPI questions
- Missing-year guardrails
- Amenity yes/no retrieval with citations
- Amenity list retrieval
- Floorplan category and detail retrieval
- Charge breakdown chart answers
- Unit-type rent comparison chart answers
- Top-balance table answers
- Cross-property retrieval scoping

The runner checks retrieval separately from generation:

- Retrieval has results
- Retrieval is scoped to the requested `property_code`
- Expected page type is present
- Expected evidence terms appear in retrieved chunks
- Hybrid search contributes both vector and keyword results

Generation checks:

- Required answer facts are present
- Forbidden leakage or stale fallback text is absent
- Expected UI component types are returned
- Expected tools were invoked
- Sources are property-scoped
- Source count is constrained for answers like floorplans, where many detail pages should collapse to one main source link

The latest local run produced:

```text
Retrieval pass rate:  4/4 (100%)
Generation pass rate: 9/9 (100%)
```

## LLM-Judged Metrics

For more realistic quality metrics, the project also includes an optional Groq-backed LLM judge:

```bash
uv run python scripts/run_llm_judge_evals.py
```

The judge reads these environment variables:

```bash
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

It uses the same golden cases, but asks the judge model to score retrieved chunks and final answers.

Retrieval metrics:

- `precision_at_k`: fraction of retrieved chunks judged useful evidence
- `mrr`: reciprocal rank of the first useful evidence chunk
- `ndcg_at_k`: ranking quality using 0-3 judged relevance scores
- `mean_relevance_0_to_3`: average judged relevance across retrieved chunks
- `evidence_term_recall`: fraction of expected evidence terms found in retrieved context
- `property_scope_accuracy`: whether all retrieved chunks stayed within the active property code

Generation metrics:

- `faithfulness`: answer support from retrieved/structured context, scored 1-5
- `answer_relevancy`: how directly the answer addresses the user question, scored 1-5
- `completeness`: whether the answer covers the expected facts, scored 1-5
- `citation_quality`: citation/source appropriateness, scored 1-5
- `required_answer_term_recall`: fraction of expected answer facts present
- `generation_overall`: average normalized generation score

The default answer model is still `mock:mock-property-assistant`, so this evaluates the current deterministic system. You can evaluate a runtime LLM answer by passing another answer model:

```bash
uv run python scripts/run_llm_judge_evals.py \
  --answer-model anthropic:claude-haiku-4-5-20251001
```
