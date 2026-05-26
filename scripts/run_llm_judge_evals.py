#!/usr/bin/env python3
"""Run LLM-judged retrieval and generation metrics for the golden dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from app.core.config import get_settings
from app.services.langchain_orchestrator import LangChainOrchestrator

DEFAULT_CASES_PATH = Path("evals/golden_cases.json")
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


@dataclass
class CaseMetrics:
    name: str
    property_code: str
    retrieval: dict[str, Any] | None
    generation: dict[str, Any] | None
    judge: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--answer-model", default="mock:mock-property-assistant")
    parser.add_argument("--judge-model", default=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL))
    parser.add_argument(
        "--judge-base-url",
        default=os.getenv("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL),
    )
    parser.add_argument("--output-json", default="evals/llm_judge_report.json")
    parser.add_argument("--max-context-chars", type=int, default=1200)
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def truncate(value: str, max_chars: int) -> str:
    value = value.strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def required_terms(config: dict[str, Any] | None) -> list[str]:
    if not config:
        return []
    return list(config.get("required_terms", []))


def term_recall(text: str, terms: list[str]) -> float | None:
    if not terms:
        return None
    normalized = text.lower()
    matched = sum(1 for term in terms if term.lower() in normalized)
    return matched / len(terms)


def dcg(relevance_scores: list[int]) -> float:
    return sum(
        ((2**score - 1) / math.log2(rank + 1))
        for rank, score in enumerate(relevance_scores, start=1)
    )


def ndcg(relevance_scores: list[int]) -> float | None:
    if not relevance_scores:
        return None
    ideal = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(relevance_scores) / ideal_dcg


def parse_judge_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def build_context_payload(
    results: list[dict[str, Any]],
    max_context_chars: int,
) -> list[dict[str, Any]]:
    payload = []
    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        payload.append(
            {
                "rank": rank,
                "property_code": metadata.get("property_code"),
                "page_type": metadata.get("page_type"),
                "section": metadata.get("section_heading"),
                "title": metadata.get("title"),
                "source_url": metadata.get("source_url"),
                "vector_rank": result.get("vector_rank"),
                "keyword_rank": result.get("keyword_rank"),
                "content": truncate(result.get("content", ""), max_context_chars),
            }
        )
    return payload


def judge_case(
    client: OpenAI,
    judge_model: str,
    case: dict[str, Any],
    retrieval_results: list[dict[str, Any]],
    answer_payload: dict[str, Any] | None,
    max_context_chars: int,
) -> dict[str, Any]:
    retrieval_config = case.get("retrieval") or {}
    generation_config = case.get("generation") or {}
    payload = {
        "case_name": case["name"],
        "property_code": case["property_code"],
        "question": case["message"],
        "gold_retrieval_terms": required_terms(retrieval_config),
        "gold_answer_terms": required_terms(generation_config),
        "retrieved_contexts": build_context_payload(retrieval_results, max_context_chars),
        "answer_payload": answer_payload,
    }
    system = (
        "You are a strict evaluator for a property-scoped RAG chatbot. "
        "Judge only using the provided question, retrieved contexts, answer, and gold hints. "
        "Do not reward facts that are not supported by the retrieved context or "
        "structured tool payload. "
        "Return only valid JSON."
    )
    user = (
        "Evaluate this case and return JSON with this exact shape:\n"
        "{\n"
        '  "retrieval": {\n'
        '    "per_result": [{"rank": 1, "relevance": 0, "reason": "short"}],\n'
        '    "missed_evidence": ["short"],\n'
        '    "property_scope_issue": false\n'
        "  },\n"
        '  "generation": {\n'
        '    "faithfulness": 1,\n'
        '    "answer_relevancy": 1,\n'
        '    "completeness": 1,\n'
        '    "citation_quality": 1,\n'
        '    "grounded": false,\n'
        '    "issues": ["short"]\n'
        "  }\n"
        "}\n\n"
        "Retrieval relevance must be 0-3: 0 irrelevant, 1 weakly related, "
        "2 useful evidence, 3 directly answers the question. "
        "Generation scores must be 1-5, where 5 is best. "
        "If there is no answer_payload, still judge retrieval and set generation fields "
        "to neutral 0/false.\n\n"
        f"CASE JSON:\n{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )
    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    return parse_judge_json(content)


def retrieval_metrics(
    case: dict[str, Any],
    results: list[dict[str, Any]],
    judge: dict[str, Any],
) -> dict[str, Any] | None:
    if not case.get("retrieval"):
        return None

    per_result = judge.get("retrieval", {}).get("per_result", [])
    judged_scores = [
        int(item.get("relevance", 0))
        for item in sorted(per_result, key=lambda item: item.get("rank", 999))
    ]
    if len(judged_scores) < len(results):
        judged_scores.extend([0] * (len(results) - len(judged_scores)))
    judged_scores = judged_scores[: len(results)]
    relevant = [score >= 2 for score in judged_scores]
    first_relevant_rank = next(
        (index + 1 for index, is_relevant in enumerate(relevant) if is_relevant),
        None,
    )
    combined_context = "\n".join(result.get("content", "") for result in results)
    property_codes = {result["metadata"].get("property_code") for result in results}

    return {
        "k": len(results),
        "precision_at_k": sum(relevant) / len(results) if results else 0.0,
        "mrr": 1 / first_relevant_rank if first_relevant_rank else 0.0,
        "ndcg_at_k": ndcg(judged_scores),
        "mean_relevance_0_to_3": (
            sum(judged_scores) / len(judged_scores) if judged_scores else 0.0
        ),
        "evidence_term_recall": term_recall(
            combined_context,
            required_terms(case.get("retrieval")),
        ),
        "property_scope_accuracy": 1.0 if property_codes == {case["property_code"]} else 0.0,
        "relevance_scores": judged_scores,
    }


def generation_metrics(
    case: dict[str, Any],
    answer_payload: dict[str, Any] | None,
    judge: dict[str, Any],
) -> dict[str, Any] | None:
    if not case.get("generation") or not answer_payload:
        return None

    generation = judge.get("generation", {})
    answer = answer_payload["answer_markdown"]
    scores = {
        "faithfulness": float(generation.get("faithfulness", 0)),
        "answer_relevancy": float(generation.get("answer_relevancy", 0)),
        "completeness": float(generation.get("completeness", 0)),
        "citation_quality": float(generation.get("citation_quality", 0)),
    }
    normalized_scores = {key: value / 5 for key, value in scores.items()}
    return {
        **scores,
        "overall_0_to_1": sum(normalized_scores.values()) / len(normalized_scores),
        "required_answer_term_recall": term_recall(
            answer,
            required_terms(case.get("generation")),
        ),
        "grounded": bool(generation.get("grounded", False)),
        "issues": generation.get("issues", []),
    }


def average(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def print_metric(name: str, value: float | None) -> None:
    if value is None:
        print(f"{name}: n/a")
    else:
        print(f"{name}: {value:.3f}")


def main() -> int:
    load_dotenv()
    args = parse_args()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY is required. Add it to .env or export it before running this script."
        )

    client = OpenAI(api_key=api_key, base_url=args.judge_base_url)
    orchestrator = LangChainOrchestrator(get_settings())
    cases = load_cases(Path(args.cases))
    case_metrics: list[CaseMetrics] = []

    print("LLM Judge Eval Results")
    print("----------------------")
    print(f"Cases: {len(cases)}")
    print(f"Answer model: {args.answer_model}")
    print(f"Judge model: {args.judge_model}")

    for case in cases:
        retrieval_results: list[dict[str, Any]] = []
        if case.get("retrieval"):
            config = case["retrieval"]
            retrieval_results = orchestrator._call_tool(
                "search_property_content",
                property_code=case["property_code"],
                query=case["message"],
                page_type=config.get("page_type"),
                n_results=config.get("n_results", 5),
            )

        answer_payload = None
        if case.get("generation"):
            response = orchestrator.answer(
                property_code=case["property_code"],
                message=case["message"],
                model=args.answer_model,
            )
            answer_payload = {
                "answer_markdown": response.answer_markdown,
                "components": [component.model_dump() for component in response.components],
                "sources": [source.model_dump() for source in response.sources],
                "tool_result_keys": sorted(response.tool_results),
            }

        judge = judge_case(
            client=client,
            judge_model=args.judge_model,
            case=case,
            retrieval_results=retrieval_results,
            answer_payload=answer_payload,
            max_context_chars=args.max_context_chars,
        )
        r_metrics = retrieval_metrics(case, retrieval_results, judge)
        g_metrics = generation_metrics(case, answer_payload, judge)
        case_metrics.append(
            CaseMetrics(
                name=case["name"],
                property_code=case["property_code"],
                retrieval=r_metrics,
                generation=g_metrics,
                judge=judge,
            )
        )

        print()
        print(f"{case['name']} ({case['property_code']})")
        if r_metrics:
            print(
                "  retrieval: "
                f"precision@k={r_metrics['precision_at_k']:.3f}, "
                f"MRR={r_metrics['mrr']:.3f}, "
                f"NDCG@k={r_metrics['ndcg_at_k']:.3f}, "
                f"evidence_recall={r_metrics['evidence_term_recall']:.3f}"
            )
        if g_metrics:
            print(
                "  generation: "
                f"faithfulness={g_metrics['faithfulness']:.1f}/5, "
                f"relevancy={g_metrics['answer_relevancy']:.1f}/5, "
                f"completeness={g_metrics['completeness']:.1f}/5, "
                f"citation={g_metrics['citation_quality']:.1f}/5, "
                f"overall={g_metrics['overall_0_to_1']:.3f}"
            )

    retrieval_values = [case.retrieval for case in case_metrics if case.retrieval]
    generation_values = [case.generation for case in case_metrics if case.generation]

    print()
    print("Aggregate Metrics")
    print("-----------------")
    print_metric(
        "retrieval_precision_at_k",
        average([metrics["precision_at_k"] for metrics in retrieval_values]),
    )
    print_metric("retrieval_mrr", average([metrics["mrr"] for metrics in retrieval_values]))
    print_metric(
        "retrieval_ndcg_at_k",
        average([metrics["ndcg_at_k"] for metrics in retrieval_values]),
    )
    print_metric(
        "retrieval_evidence_term_recall",
        average([metrics["evidence_term_recall"] for metrics in retrieval_values]),
    )
    print_metric(
        "retrieval_property_scope_accuracy",
        average([metrics["property_scope_accuracy"] for metrics in retrieval_values]),
    )
    print_metric(
        "generation_faithfulness",
        average([metrics["faithfulness"] / 5 for metrics in generation_values]),
    )
    print_metric(
        "generation_answer_relevancy",
        average([metrics["answer_relevancy"] / 5 for metrics in generation_values]),
    )
    print_metric(
        "generation_completeness",
        average([metrics["completeness"] / 5 for metrics in generation_values]),
    )
    print_metric(
        "generation_citation_quality",
        average([metrics["citation_quality"] / 5 for metrics in generation_values]),
    )
    print_metric(
        "generation_required_answer_term_recall",
        average([metrics["required_answer_term_recall"] for metrics in generation_values]),
    )
    print_metric(
        "generation_overall",
        average([metrics["overall_0_to_1"] for metrics in generation_values]),
    )

    report = {
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
        "cases": [
            {
                "name": case.name,
                "property_code": case.property_code,
                "retrieval": case.retrieval,
                "generation": case.generation,
                "judge": case.judge,
            }
            for case in case_metrics
        ],
        "aggregate": {
            "retrieval_precision_at_k": average(
                [metrics["precision_at_k"] for metrics in retrieval_values]
            ),
            "retrieval_mrr": average([metrics["mrr"] for metrics in retrieval_values]),
            "retrieval_ndcg_at_k": average(
                [metrics["ndcg_at_k"] for metrics in retrieval_values]
            ),
            "retrieval_evidence_term_recall": average(
                [metrics["evidence_term_recall"] for metrics in retrieval_values]
            ),
            "retrieval_property_scope_accuracy": average(
                [metrics["property_scope_accuracy"] for metrics in retrieval_values]
            ),
            "generation_faithfulness": average(
                [metrics["faithfulness"] / 5 for metrics in generation_values]
            ),
            "generation_answer_relevancy": average(
                [metrics["answer_relevancy"] / 5 for metrics in generation_values]
            ),
            "generation_completeness": average(
                [metrics["completeness"] / 5 for metrics in generation_values]
            ),
            "generation_citation_quality": average(
                [metrics["citation_quality"] / 5 for metrics in generation_values]
            ),
            "generation_required_answer_term_recall": average(
                [metrics["required_answer_term_recall"] for metrics in generation_values]
            ),
            "generation_overall": average(
                [metrics["overall_0_to_1"] for metrics in generation_values]
            ),
        },
    }
    Path(args.output_json).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote JSON report to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
