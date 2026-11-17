"""JSONL writer for :class:`StepRecord` rows (Resource.md Section F schema)."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from jevmaster.types import StepRecord


class StepLogger:
    def __init__(self, output_dir: str | Path, run_name: str = "run"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / f"{run_name}.jsonl"
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, record: StepRecord) -> None:
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def log_summary(self, summary: dict) -> None:
        with self._lock:
            self._fh.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "StepLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
