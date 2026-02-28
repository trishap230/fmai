import csv
import json
import os
from datetime import datetime
from typing import Dict, Any


LOG_PATH = "models/experiments.csv"


def ensure_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "model_path", "params", "accuracy", "f1_macro"])


def save_experiment(model_path: str, params: Dict[str, Any], accuracy: float, f1_macro: float):
    ensure_log()
    row = [datetime.utcnow().isoformat(), model_path, json.dumps(params, ensure_ascii=False), f"{accuracy:.6f}", f"{f1_macro:.6f}"]
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
