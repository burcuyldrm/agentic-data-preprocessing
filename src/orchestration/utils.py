"""
Orkestrasyon yardımcıları — dosya hash ve skor karşılaştırma.
"""

import hashlib


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_better(new_score: float, best_score: float, task_type: str) -> bool:
    if task_type == "classification":
        return new_score > best_score
    return new_score < best_score  # regression: daha düşük RMSE daha iyi
