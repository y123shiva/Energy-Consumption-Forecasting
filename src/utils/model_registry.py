import mlflow
from typing import Dict, Optional


def register_model(run_id: str, artifact_path: str, model_name: str) -> Optional[mlflow.entities.model_registry.RegisteredModel]:
    """Register an artifact from a run into the MLflow Model Registry.

    Args:
        run_id: MLflow run id where the artifact was logged.
        artifact_path: Path to the artifact relative to the run (e.g. "xgb_model.pkl").
        model_name: Desired registered model name in MLflow.

    Returns:
        The RegisteredModel or None on failure.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    try:
        rm = mlflow.register_model(model_uri, model_name)
        print(f"Registered model {model_name} -> {rm.version}")
        return rm
    except Exception as exc:
        print(f"Failed to register model {model_name}: {exc}")
        return None


def register_models_from_run(run_id: str, registrations: Dict[str, str]) -> Dict[str, Optional[mlflow.entities.model_registry.RegisteredModel]]:
    """Register multiple artifacts from a run.

    registrations: mapping of model_name -> artifact_path
    """
    results = {}
    for name, path in registrations.items():
        results[name] = register_model(run_id, path, name)
    return results
