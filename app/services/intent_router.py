from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

from app.core.config import Settings
from app.retrieval.embeddings import LocalHashEmbedder, TextEmbedder, build_embedder

STRUCTURED_INTENTS = {
    "latest_kpis",
    "executive_summary",
    "occupancy_trend",
    "charge_breakdown",
    "rent_lease_comparison",
    "vacant_units",
    "top_balances",
    "rent_by_unit_type",
}
RETRIEVAL_INTENTS = {
    "amenity_list",
    "floorplans",
    "gallery",
    "location",
    "website_content",
}

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]{1,}", re.IGNORECASE)

CROSS_PROPERTY_PHRASES = [
    "another property",
    "other property",
    "other properties",
    "different property",
    "compare properties",
    "compare this property",
    "across properties",
    "across all properties",
    "between properties",
    "portfolio",
    "portfolio-wide",
    "all properties",
    "every property",
    "each property",
    "which property",
    "best property",
    "worst property",
]

AMBIGUOUS_SHORT_REQUESTS = {
    "analyze",
    "balance",
    "balances",
    "charge",
    "charges",
    "compare",
    "fee",
    "fees",
    "help",
    "info",
    "lease",
    "more",
    "occupancy",
    "rent",
    "show",
    "summarize",
    "vacancy",
    "vacant",
}

INTENT_EXAMPLES = {
    "latest_kpis": [
        "latest occupancy market rent lease charges and vacant count",
        "current rent roll snapshot",
        "show the main property KPIs",
        "what are the key metrics for this property",
        "how is the property performing right now",
    ],
    "executive_summary": [
        "quick executive summary of this property",
        "give me a high level summary",
        "summarize the property's performance for leadership",
        "brief management overview",
        "what is the overall read on this asset",
    ],
    "occupancy_trend": [
        "occupancy trend over time",
        "how has occupancy changed across available months",
        "occupancy month to month",
        "show the occupancy history",
        "what changed in occupancy over the reporting period",
        "how did occupancy move each month",
    ],
    "charge_breakdown": [
        "biggest charge categories",
        "largest fee buckets",
        "breakdown of lease charges",
        "show charge mix by category",
        "which income buckets are largest",
        "fee category breakdown chart",
        "what charges make up revenue",
    ],
    "rent_lease_comparison": [
        "compare rent and lease charges",
        "rent vs lease charges comparison",
        "market rent against lease charges",
        "lease charges compared to rent",
        "difference between market rent and lease charges",
        "rent versus billed lease charges",
    ],
    "vacant_units": [
        "which units are vacant",
        "list open units and their floorplan types",
        "show vacant units with bedroom category",
        "available units and layouts",
        "give me all vacant units",
        "vacant unit table",
    ],
    "top_balances": [
        "which units have the highest balances",
        "top resident balances",
        "largest outstanding balances",
        "show balance table",
        "highest delinquency by unit",
    ],
    "rent_by_unit_type": [
        "average market rent by unit type",
        "compare rent by bedroom category",
        "rent by floorplan code",
        "average rent by apartment layout",
        "rent levels by unit size",
        "market rent by floorplan",
    ],
    "amenity_list": [
        "what amenities are listed",
        "list apartment features",
        "community amenities on the website",
        "what features are included",
    ],
    "floorplans": [
        "what floorplans are advertised",
        "which floor plans are available on the website",
        "list bedroom floorplan categories",
    ],
    "gallery": [
        "show gallery photos",
        "what images are on the property website",
    ],
    "location": [
        "where is this property located",
        "what is the property address",
        "property location",
    ],
    "website_content": [
        "does the property have ev charging",
        "does the website mention bike storage",
        "is parking listed on the website",
        "what reviews are mentioned on the property website",
        "are there any reviews for this property",
        "website source for this property",
    ],
}


@dataclass(frozen=True)
class IntentRoute:
    intent: str | None
    confidence: float
    needs_clarification: bool = False
    reason: str = "unknown"
    matched_example: str | None = None


@dataclass(frozen=True)
class _EmbeddedExample:
    intent: str
    text: str
    vector: list[float]


class IntentRouter:
    """Hybrid intent classifier for selecting tools and UI components.

    Deterministic rules handle obvious high-risk cases, and local embeddings
    cover paraphrases that do not contain the exact trigger words.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        min_confidence: float = 0.45,
        min_margin: float = 0.025,
    ) -> None:
        self.embedder = embedder
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.examples = self._embed_examples()

    def route(self, message: str) -> IntentRoute:
        normalized = self._normalized(message)
        if not normalized:
            return IntentRoute(
                intent=None,
                confidence=1.0,
                needs_clarification=True,
                reason="empty",
            )

        rule_route = self._rule_route(normalized)
        if rule_route:
            return rule_route

        if not self.examples:
            return IntentRoute(intent=None, confidence=0.0, reason="no_examples")

        query_vector = self.embedder.embed(message)
        scored = sorted(
            (
                (self._dot(query_vector, example.vector), example)
                for example in self.examples
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best_example = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_score - second_score
        if best_score >= self.min_confidence and margin >= self.min_margin:
            return IntentRoute(
                intent=best_example.intent,
                confidence=round(best_score, 4),
                reason="semantic",
                matched_example=best_example.text,
            )

        return IntentRoute(
            intent=None,
            confidence=round(best_score, 4),
            reason="low_confidence",
            matched_example=best_example.text,
        )

    def _embed_examples(self) -> list[_EmbeddedExample]:
        texts = [
            example_text
            for examples in INTENT_EXAMPLES.values()
            for example_text in examples
        ]
        vectors = self.embedder.embed_many(texts)
        embedded = []
        index = 0
        for intent, examples in INTENT_EXAMPLES.items():
            for example_text in examples:
                embedded.append(
                    _EmbeddedExample(
                        intent=intent,
                        text=example_text,
                        vector=vectors[index],
                    )
                )
                index += 1
        return embedded

    @staticmethod
    def _rule_route(normalized: str) -> IntentRoute | None:
        tokens = normalized.split()

        if any(phrase in normalized for phrase in CROSS_PROPERTY_PHRASES):
            return IntentRoute(
                intent=None,
                confidence=1.0,
                needs_clarification=True,
                reason="cross_property",
            )

        if normalized in AMBIGUOUS_SHORT_REQUESTS or (
            len(tokens) <= 2 and all(token in AMBIGUOUS_SHORT_REQUESTS for token in tokens)
        ):
            return IntentRoute(
                intent="clarify_charges"
                if any(token in {"charge", "charges", "fee", "fees", "lease"} for token in tokens)
                else None,
                confidence=1.0,
                needs_clarification=True,
                reason="ambiguous_short_request",
            )

        if "review" in tokens or "reviews" in tokens:
            return IntentRoute("website_content", 1.0, reason="rule")

        if (
            ("rent" in tokens or "market rent" in normalized)
            and ("lease charge" in normalized or "lease charges" in normalized)
            and any(
                phrase in normalized
                for phrase in [
                    "against",
                    "compare",
                    "compared",
                    "comparison",
                    "difference",
                    "versus",
                    "vs",
                ]
            )
        ):
            return IntentRoute("rent_lease_comparison", 1.0, reason="rule")

        if "occupancy" in tokens and any(
            phrase in normalized
            for phrase in [
                "across available months",
                "changed",
                "history",
                "month to month",
                "monthly",
                "moved",
                "over time",
                "trend",
            ]
        ):
            return IntentRoute("occupancy_trend", 1.0, reason="rule")

        if ("vacant" in tokens or "vacancy" in tokens) and any(
            term in tokens for term in ["count", "number", "total"]
        ):
            return IntentRoute("latest_kpis", 1.0, reason="rule")

        has_vacancy_term = any(term in tokens for term in ["open", "vacant", "vacancy"])
        has_unit_or_layout = any(
            term in tokens for term in ["apartment", "apartments", "unit", "units", "layouts", "layout"]
        )
        if (has_vacancy_term and has_unit_or_layout) or (
            "available" in tokens and any(term in tokens for term in ["apartment", "apartments", "unit", "units"])
        ):
            return IntentRoute("vacant_units", 1.0, reason="rule")

        if any(term in tokens for term in ["bucket", "buckets", "category", "categories", "mix"]) and any(
            term in tokens for term in ["charge", "charges", "fee", "fees", "income", "revenue"]
        ):
            return IntentRoute("charge_breakdown", 1.0, reason="rule")

        if "balance" in tokens or "balances" in tokens or "delinquency" in tokens:
            return IntentRoute("top_balances", 1.0, reason="rule")

        return None

    @staticmethod
    def _normalized(message: str) -> str:
        return " ".join(
            match.group(0).lower().replace("'", "")
            for match in TOKEN_RE.finditer(message)
        )

    @staticmethod
    def _dot(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=False))


def get_intent_router(settings: Settings) -> IntentRouter:
    return _cached_intent_router(
        settings.embedding_provider,
        settings.embedding_model,
        str(settings.embedding_cache_path),
    )


@lru_cache(maxsize=4)
def _cached_intent_router(
    embedding_provider: str,
    embedding_model: str,
    embedding_cache_path: str,
) -> IntentRouter:
    try:
        embedder = build_embedder(
            Settings(
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_cache_path=Path(embedding_cache_path),
            )
        )
    except Exception:
        embedder = LocalHashEmbedder()
    return IntentRouter(embedder)
