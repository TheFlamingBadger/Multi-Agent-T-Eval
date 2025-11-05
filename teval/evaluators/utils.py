from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
from mmengine import dump

Numeric = (int, float, np.floating, np.integer)


def _to_builtin(value):
    """Convert numpy numeric types to native Python types for JSON serialization."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _compute_score(metrics: Dict[str, float]) -> float:
    """Compute the average numeric score across available metrics."""
    numeric_values = [
        float(_to_builtin(val)) for val in metrics.values() if isinstance(val, Numeric)
    ]
    if not numeric_values:
        return 0.0
    return float(np.mean(numeric_values))


def annotate_dataset(
    raw_dataset: Dict[str, Dict],
    per_item_metrics: Dict[str, Dict[str, float]],
    evaluator_name: str,
    dataset_path: str,
    annotation_path: Optional[str] = None,
    evaluated_at: Optional[str] = None,
) -> None:
    """Attach evaluation metadata to each sample and persist the dataset."""
    if raw_dataset is None:
        return
    if evaluated_at is None:
        evaluated_at = datetime.now(timezone.utc).isoformat()
    target_path = annotation_path or dataset_path
    for sample_id, metrics in per_item_metrics.items():
        if sample_id not in raw_dataset:
            continue
        metrics_builtin = {key: _to_builtin(val) for key, val in metrics.items()}
        score = _compute_score(metrics_builtin)
        score = max(0.0, min(1.0, score))
        raw_dataset[sample_id]["evaluation_result"] = score
    dump(raw_dataset, target_path)
