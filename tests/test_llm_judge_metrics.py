from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_llm_judge_evals.py"
SPEC = importlib.util.spec_from_file_location("run_llm_judge_evals", MODULE_PATH)
assert SPEC and SPEC.loader
judge_module = importlib.util.module_from_spec(SPEC)
sys.modules["run_llm_judge_evals"] = judge_module
SPEC.loader.exec_module(judge_module)

generation_metrics = judge_module.generation_metrics
retrieval_metrics = judge_module.retrieval_metrics


def test_retrieval_metrics_compute_precision_mrr_ndcg_and_scope() -> None:
    case = {
        "name": "sample",
        "property_code": "115r",
        "retrieval": {"required_terms": ["EV Charging", "Bike Racks"]},
    }
    results = [
        {
            "content": "EV Charging Stations",
            "metadata": {"property_code": "115r"},
        },
        {
            "content": "Generic amenity text",
            "metadata": {"property_code": "115r"},
        },
        {
            "content": "Bike Racks & Storage Lockers",
            "metadata": {"property_code": "115r"},
        },
    ]
    judge = {
        "retrieval": {
            "per_result": [
                {"rank": 1, "relevance": 3},
                {"rank": 2, "relevance": 1},
                {"rank": 3, "relevance": 2},
            ]
        }
    }

    metrics = retrieval_metrics(case, results, judge)

    assert metrics is not None
    assert metrics["precision_at_k"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["evidence_term_recall"] == pytest.approx(1.0)
    assert metrics["property_scope_accuracy"] == pytest.approx(1.0)
    assert metrics["ndcg_at_k"] > 0


def test_generation_metrics_normalize_judge_scores_and_term_recall() -> None:
    case = {
        "name": "sample",
        "property_code": "115r",
        "generation": {
            "required_terms": ["EV charging", "bike storage"],
        },
    }
    answer_payload = {
        "answer_markdown": "Yes, this property has EV charging and bike storage.",
    }
    judge = {
        "generation": {
            "faithfulness": 5,
            "answer_relevancy": 4,
            "completeness": 4,
            "citation_quality": 3,
            "grounded": True,
            "issues": [],
        }
    }

    metrics = generation_metrics(case, answer_payload, judge)

    assert metrics is not None
    assert metrics["faithfulness"] == 5
    assert metrics["overall_0_to_1"] == pytest.approx(0.8)
    assert metrics["required_answer_term_recall"] == pytest.approx(1.0)
    assert metrics["grounded"] is True
