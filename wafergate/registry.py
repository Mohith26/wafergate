"""MLflow tracking + model registry, and a hash-chained audit log of gate decisions."""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

MODEL_NAME = "wafer-defect-classifier"
CHAMPION = "champion"


class WaferPyfunc(mlflow.pyfunc.PythonModel):
    """Serving interface: input (N,2,48,48) array, output label or ABSTAIN per wafer."""

    def load_context(self, context):
        with open(context.artifacts["wafer_model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input, params=None):
        import numpy as np

        labels, _, _ = self.model.predict(np.asarray(model_input, dtype=np.float32))
        return list(labels)


def setup(tracking_dir: str | Path):
    tracking_dir = Path(tracking_dir).resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{tracking_dir / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    exp = "wafergate"
    if mlflow.get_experiment_by_name(exp) is None:
        mlflow.create_experiment(exp, artifact_location=(tracking_dir / "artifacts").as_uri())
    mlflow.set_experiment(exp)
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def model_sha256(model) -> str:
    return hashlib.sha256(pickle.dumps(model)).hexdigest()


def log_and_register(client, model, name: str, params: dict, metrics: dict, work_dir: str | Path, code_path: str):
    """Log one candidate run and register it as a new model version. Returns (run_id, version)."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    pkl = work_dir / f"{name}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(model, f)
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        flat.update({f"recall_{c}": r for c, r in metrics.get("per_class_recall", {}).items()})
        mlflow.log_metrics(flat)
        mlflow.log_dict(metrics, "eval_metrics.json")
        info = mlflow.pyfunc.log_model(
            name="model", python_model=WaferPyfunc(), artifacts={"wafer_model": str(pkl)},
            code_paths=[code_path],
        )
    mv = mlflow.register_model(info.model_uri, MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, mv.version, "candidate", name)
    return run.info.run_id, int(mv.version)


def apply_decision(client, version: int, decision) -> None:
    client.set_model_version_tag(MODEL_NAME, str(version), "gate", "passed" if decision.promoted else "blocked")
    if decision.violations:
        client.set_model_version_tag(MODEL_NAME, str(version), "violations", " | ".join(decision.violations)[:4900])
    if decision.promoted:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, str(version))


def current_champion(client):
    try:
        return int(client.get_model_version_by_alias(MODEL_NAME, CHAMPION).version)
    except Exception:
        return None


def load_champion():
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION}")


class AuditLog:
    """Append-only JSONL where each record carries the hash of the previous one,
    so editing or deleting a past decision breaks the chain."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _records(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, record: dict) -> dict:
        recs = self._records()
        prev = recs[-1]["hash"] if recs else "0" * 64
        body = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record, "prev_hash": prev}
        body["hash"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        with self.path.open("a") as f:
            f.write(json.dumps(body, sort_keys=True) + "\n")
        return body

    def verify(self) -> bool:
        prev = "0" * 64
        for r in self._records():
            h = r.pop("hash")
            if r["prev_hash"] != prev:
                return False
            if hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() != h:
                return False
            prev = h
        return True
