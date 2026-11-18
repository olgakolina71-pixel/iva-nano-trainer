"""Optional experiment tracking for IVA Nano training.

Set IVA_TRACKER=mlflow or IVA_TRACKER=wandb to enable tracking.
The default is disabled and never requires external credentials.
"""
import os


class ExperimentTracker:
    def __init__(self, backend, run=None):
        self.backend = backend
        self.run = run

    def log_metrics(self, values, step):
        clean = {str(key): float(value) for key, value in values.items() if value is not None}
        if not clean:
            return
        if self.backend == "mlflow":
            import mlflow
            mlflow.log_metrics(clean, step=int(step))
        elif self.backend == "wandb":
            self.run.log(clean, step=int(step))

    def finish(self):
        if self.backend == "mlflow":
            import mlflow
            mlflow.end_run()
        elif self.backend == "wandb":
            self.run.finish()


def start_experiment(name, config):
    backend = os.environ.get("IVA_TRACKER", "none").strip().lower()
    if backend in {"", "none", "off", "disabled"}:
        return None
    if backend == "mlflow":
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("Install optional tracker dependencies: pip install -r requirements-tracking.txt") from exc
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "IVA Nano"))
        mlflow.start_run(run_name=name)
        flat = {}
        for section, values in config.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, (str, int, float, bool)):
                        flat[f"{section}.{key}"] = value
            elif isinstance(values, (str, int, float, bool)):
                flat[section] = values
        mlflow.log_params(flat)
        return ExperimentTracker("mlflow")
    if backend == "wandb":
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("Install optional tracker dependencies: pip install -r requirements-tracking.txt") from exc
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "iva-nano"),
            name=name,
            config=config,
        )
        return ExperimentTracker("wandb", run)
    raise ValueError("IVA_TRACKER must be one of: none, mlflow, wandb")
