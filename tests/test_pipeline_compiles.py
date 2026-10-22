import yaml
from kfp import compiler

from pipelines.risk_train_pipeline import risk_train_eval


def _tasks(spec: dict) -> dict:
    return spec["root"]["dag"]["tasks"]


def test_pipeline_compiles_with_gate_and_registry(tmp_path):
    out = tmp_path / "p.yaml"
    compiler.Compiler().compile(risk_train_eval, str(out))
    spec = yaml.safe_load(out.read_text())

    inner = spec["components"]["comp-exit-handler-1"]["dag"]["tasks"]
    assert {"build-features", "train", "evaluate", "importer", "condition-branches-2"} <= set(inner)
    assert "notify-if-failed" in _tasks(spec)

    executors = spec["deploymentSpec"]["executors"]
    assert {"exec-register", "exec-notify-rejected"} <= set(executors)
    images = {e["container"]["image"] for e in executors.values() if "container" in e}
    assert len(images) == 1, "every component should run on the pipeline image"

    params = spec["root"]["inputDefinitions"]["parameters"]
    assert {"min_f1", "max_regression", "registry_model", "alias", "gold_uri", "source_table"} <= set(params)
