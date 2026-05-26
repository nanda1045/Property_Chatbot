#!/usr/bin/env python3
"""Run deterministic golden retrieval and generation evals."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.services.langchain_orchestrator import LangChainOrchestrator

DEFAULT_CASES_PATH = Path("evals/golden_cases.json")


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--model", default="mock:mock-property-assistant")
    parser.add_argument("--output-json", help="Optional path for machine-readable results.")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def contains_term(haystack: str, term: str) -> bool:
    return term.lower() in haystack.lower()


def missing_terms(haystack: str, terms: list[str]) -> list[str]:
    return [term for term in terms if not contains_term(haystack, term)]


def present_forbidden_terms(haystack: str, terms: list[str]) -> list[str]:
    return [term for term in terms if contains_term(haystack, term)]


def evaluate_retrieval(
    orchestrator: LangChainOrchestrator,
    case: dict[str, Any],
) -> tuple[bool, list[CheckResult], dict[str, Any] | None]:
    config = case.get("retrieval")
    if not config:
        return True, [], None

    results = orchestrator._call_tool(
        "search_property_content",
        property_code=case["property_code"],
        query=case["message"],
        page_type=config.get("page_type"),
        n_results=config.get("n_results", 5),
    )

    checks: list[CheckResult] = []
    result_text = "\n".join(
        "\n".join(
            [
                result.get("content") or "",
                result.get("metadata", {}).get("title") or "",
                result.get("metadata", {}).get("source_url") or "",
            ]
        )
        for result in results
    )
    result_property_codes = {
        result.get("metadata", {}).get("property_code") for result in results
    }
    result_page_types = {
        result.get("metadata", {}).get("page_type") for result in results
    }

    checks.append(
        CheckResult(
            "has_results",
            bool(results),
            f"{len(results)} result(s)",
        )
    )
    checks.append(
        CheckResult(
            "property_scope",
            result_property_codes == {case["property_code"]},
            f"property_codes={sorted(code for code in result_property_codes if code)}",
        )
    )
    expected_page_type = config.get("expected_page_type")
    if expected_page_type:
        checks.append(
            CheckResult(
                "page_type",
                expected_page_type in result_page_types,
                f"page_types={sorted(page_type for page_type in result_page_types if page_type)}",
            )
        )

    required_terms = config.get("required_terms", [])
    missing = missing_terms(result_text, required_terms)
    checks.append(
        CheckResult(
            "required_terms",
            not missing,
            f"missing={missing or []}",
        )
    )
    checks.append(
        CheckResult(
            "hybrid_signal",
            any(result.get("vector_rank") for result in results)
            and any(result.get("keyword_rank") for result in results),
            "requires at least one vector-ranked and one keyword-ranked result",
        )
    )

    artifact = {
        "top_results": [
            {
                "rank": index,
                "property_code": result["metadata"].get("property_code"),
                "page_type": result["metadata"].get("page_type"),
                "section": result["metadata"].get("section_heading"),
                "source_url": result["metadata"].get("source_url"),
                "score": result.get("score"),
                "vector_rank": result.get("vector_rank"),
                "keyword_rank": result.get("keyword_rank"),
            }
            for index, result in enumerate(results, start=1)
        ]
    }
    return all(check.passed for check in checks), checks, artifact


def evaluate_generation(
    orchestrator: LangChainOrchestrator,
    case: dict[str, Any],
    model: str,
) -> tuple[bool, list[CheckResult], dict[str, Any] | None]:
    config = case.get("generation")
    if not config:
        return True, [], None

    response = orchestrator.answer(
        property_code=case["property_code"],
        message=case["message"],
        model=model,
    )
    answer = response.answer_markdown
    component_types = {component.type for component in response.components}
    source_urls = {source.source_url for source in response.sources if source.source_url}
    source_property_codes = {source.property_code for source in response.sources}
    tool_keys = set(response.tool_results)

    checks: list[CheckResult] = []

    required_terms = config.get("required_terms", [])
    missing = missing_terms(answer, required_terms)
    checks.append(
        CheckResult(
            "answer_required_terms",
            not missing,
            f"missing={missing or []}",
        )
    )

    forbidden = present_forbidden_terms(answer, config.get("forbidden_terms", []))
    checks.append(
        CheckResult(
            "answer_forbidden_terms",
            not forbidden,
            f"present={forbidden or []}",
        )
    )

    expected_components = set(config.get("expected_component_types", []))
    checks.append(
        CheckResult(
            "component_types",
            expected_components.issubset(component_types),
            f"expected={sorted(expected_components)}, actual={sorted(component_types)}",
        )
    )

    expected_tool_keys = set(config.get("expected_tool_keys", []))
    checks.append(
        CheckResult(
            "tool_keys",
            expected_tool_keys.issubset(tool_keys),
            f"expected={sorted(expected_tool_keys)}, actual={sorted(tool_keys)}",
        )
    )

    expected_source_urls = set(config.get("expected_source_urls", []))
    if expected_source_urls:
        checks.append(
            CheckResult(
                "source_urls",
                expected_source_urls == source_urls,
                f"expected={sorted(expected_source_urls)}, actual={sorted(source_urls)}",
            )
        )

    expected_source_code = config.get("expected_source_property_code")
    if expected_source_code:
        checks.append(
            CheckResult(
                "source_property_scope",
                source_property_codes == {expected_source_code},
                f"actual={sorted(source_property_codes)}",
            )
        )

    if "max_source_count" in config:
        checks.append(
            CheckResult(
                "source_count",
                len(response.sources) <= config["max_source_count"],
                f"max={config['max_source_count']}, actual={len(response.sources)}",
            )
        )

    artifact = {
        "answer_markdown": answer,
        "component_types": sorted(component_types),
        "source_urls": sorted(source_urls),
        "tool_keys": sorted(tool_keys),
    }
    return all(check.passed for check in checks), checks, artifact


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def print_checks(case_name: str, phase: str, checks: list[CheckResult]) -> None:
    status = "PASS" if all(check.passed for check in checks) else "FAIL"
    print(f"{status:4} {phase:10} {case_name}")
    for check in checks:
        marker = "ok" if check.passed else "!!"
        print(f"      {marker} {check.name}: {check.details}")


def main() -> int:
    args = parse_args()
    if args.retrieval_only and args.generation_only:
        raise SystemExit("Choose either --retrieval-only or --generation-only, not both.")

    cases = load_cases(Path(args.cases))
    orchestrator = LangChainOrchestrator(get_settings())
    run_retrieval = not args.generation_only
    run_generation = not args.retrieval_only
    output_cases = []
    retrieval_passes = 0
    retrieval_total = 0
    generation_passes = 0
    generation_total = 0

    print_section("Golden Eval Results")
    print(f"Cases: {len(cases)}")
    print(f"Model: {args.model}")

    for case in cases:
        case_output: dict[str, Any] = {
            "name": case["name"],
            "property_code": case["property_code"],
            "message": case["message"],
            "tags": case.get("tags", []),
        }

        if run_retrieval and case.get("retrieval"):
            passed, checks, artifact = evaluate_retrieval(orchestrator, case)
            retrieval_total += 1
            retrieval_passes += int(passed)
            print_checks(case["name"], "retrieval", checks)
            case_output["retrieval"] = {
                "passed": passed,
                "checks": [check.__dict__ for check in checks],
                "artifact": artifact,
            }

        if run_generation and case.get("generation"):
            passed, checks, artifact = evaluate_generation(orchestrator, case, args.model)
            generation_total += 1
            generation_passes += int(passed)
            print_checks(case["name"], "generation", checks)
            case_output["generation"] = {
                "passed": passed,
                "checks": [check.__dict__ for check in checks],
                "artifact": artifact,
            }

        output_cases.append(case_output)

    print_section("Summary")
    retrieval_rate = retrieval_passes / retrieval_total if retrieval_total else 1.0
    generation_rate = generation_passes / generation_total if generation_total else 1.0
    print(f"Retrieval pass rate:  {retrieval_passes}/{retrieval_total} ({retrieval_rate:.0%})")
    print(f"Generation pass rate: {generation_passes}/{generation_total} ({generation_rate:.0%})")

    summary = {
        "model": args.model,
        "retrieval": {
            "passed": retrieval_passes,
            "total": retrieval_total,
            "pass_rate": retrieval_rate,
        },
        "generation": {
            "passed": generation_passes,
            "total": generation_total,
            "pass_rate": generation_rate,
        },
        "cases": output_cases,
    }
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote JSON report to {args.output_json}")

    return 0 if retrieval_passes == retrieval_total and generation_passes == generation_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
