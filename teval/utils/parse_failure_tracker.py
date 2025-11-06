from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional


def _preview(text: Optional[str], max_length: int = 600) -> Optional[str]:
    """Generate a shortened preview string for long completions."""
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_length:
        return text
    truncated = text[: max_length - 20].rstrip()
    return f"{truncated}… (+{len(text) - len(truncated)} chars)"


class ParseFailureTracker:
    """Collects per-sample parsing failures and persists diagnostics to disk."""

    def __init__(self, dataset_path: str, evaluator_name: str) -> None:
        self.dataset_path = Path(dataset_path)
        self.evaluator_name = evaluator_name
        self._entries: Dict[str, Dict[str, Any]] = {}

    @property
    def detail_path(self) -> Path:
        return self.dataset_path.with_name(
            f"{self.dataset_path.stem}_parse_failures.jsonl"
        )

    @property
    def summary_path(self) -> Path:
        return self.dataset_path.with_name(
            f"{self.dataset_path.stem}_parse_failure_summary.json"
        )

    def record(
        self,
        sample_id: str,
        *,
        mode: str,
        detail: Optional[str] = None,
        response_format: Optional[str] = None,
        prediction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a parse failure for the given sample."""
        entry: Dict[str, Any] = {
            "sample_id": sample_id,
            "mode": mode,
            "evaluator": self.evaluator_name,
            "dataset": str(self.dataset_path),
        }
        if response_format:
            entry["response_format"] = response_format
        if detail:
            entry["detail"] = detail
        preview = _preview(prediction)
        if preview is not None:
            entry["prediction_preview"] = preview
        self._entries[sample_id] = entry
        return entry

    def clear(self, sample_id: str) -> None:
        """Remove failure diagnostics for a sample (e.g., after re-run success)."""
        if sample_id in self._entries:
            self._entries.pop(sample_id, None)

    def summary_counts(self) -> Dict[str, int]:
        counts = Counter(entry["mode"] for entry in self._entries.values())
        return dict(sorted(counts.items()))

    def write_files(self) -> None:
        """Persist failure diagnostics to disk."""
        detail_path = self.detail_path
        summary_path = self.summary_path

        if not self._entries:
            # Remove stale files if present to avoid confusion.
            for path in (detail_path, summary_path):
                if path.exists():
                    try:
                        path.unlink()
                    except OSError:
                        pass
            return

        detail_path.parent.mkdir(parents=True, exist_ok=True)
        entries = [
            self._entries[sample_id] for sample_id in sorted(self._entries.keys())
        ]
        with detail_path.open("w", encoding="utf-8") as detail_file:
            for entry in entries:
                detail_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        summary_payload = {
            "evaluator": self.evaluator_name,
            "dataset": str(self.dataset_path),
            "total_failures": int(sum(self.summary_counts().values())),
            "modes": self.summary_counts(),
        }
        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary_payload, summary_file, ensure_ascii=False, indent=2)

