from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RAGAS_FAST_METRIC_NAMES = ("answer_relevancy", "context_precision")
EMBEDDING_METRIC_NAMES = frozenset({"answer_relevancy", "answer_correctness"})
RAGAS_FULL_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_recall",
)


@dataclass(frozen=True)
class ResolvedMetric:
    name: str
    metric: Any


def ragas_metric_names_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "fast":
        return RAGAS_FAST_METRIC_NAMES
    if mode == "full":
        return RAGAS_FULL_METRIC_NAMES
    return ()


def resolve_ragas_metrics(mode: str) -> list[ResolvedMetric]:
    names = ragas_metric_names_for_mode(mode)
    return [ResolvedMetric(name=name, metric=_resolve_metric(name)) for name in names]


def metrics_require_embeddings(metrics: list[ResolvedMetric]) -> bool:
    return any(metric.name in EMBEDDING_METRIC_NAMES for metric in metrics)


def _resolve_metric(metric_name: str) -> Any:
    try:
        import ragas.metrics as ragas_metrics
    except ImportError as exc:
        raise RuntimeError(
            "ragas is required for RAGAS metrics. Install project dependencies first."
        ) from exc

    candidates = {
        "faithfulness": ("Faithfulness", "faithfulness"),
        "answer_relevancy": ("ResponseRelevancy", "AnswerRelevancy", "answer_relevancy"),
        "answer_correctness": (
            "FactualCorrectness",
            "AnswerCorrectness",
            "factual_correctness",
            "answer_correctness",
        ),
        "context_precision": (
            "LLMContextPrecisionWithReference",
            "ContextPrecision",
            "context_precision",
        ),
        "context_recall": ("LLMContextRecall", "ContextRecall", "context_recall"),
    }

    for candidate in candidates[metric_name]:
        if not hasattr(ragas_metrics, candidate):
            continue
        metric = getattr(ragas_metrics, candidate)
        if isinstance(metric, type):
            return metric()
        return metric

    raise RuntimeError(f"Installed ragas version does not expose metric {metric_name}.")


def metric_output_name(resolved_metric: ResolvedMetric) -> str:
    metric = resolved_metric.metric
    return str(getattr(metric, "name", "") or resolved_metric.name)
