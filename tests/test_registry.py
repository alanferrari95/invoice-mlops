import joblib
import pytest

from src import registry
from src.api import main

MODEL = "projects/invoice-mlops/locations/europe-west4/models/123"


def fake_fetch(versions: dict):
    """versions: alias -> (version_id, artifact_uri). Records the URLs it was asked for."""
    calls = []

    def fetch(url):
        calls.append(url)
        alias = url.rsplit("@", 1)[1]
        if alias not in versions:
            return None
        version_id, uri = versions[alias]
        return {"versionId": version_id, "artifactUri": uri}

    fetch.calls = calls
    return fetch


def test_resolve_builds_regional_alias_url():
    fetch = fake_fetch({"production": ("3", "gs://b/models/run-1/model")})
    v = registry.resolve(MODEL, "production", fetch=fetch)
    assert fetch.calls == [f"https://europe-west4-aiplatform.googleapis.com/v1/{MODEL}@production"]
    assert v.version_id == "3" and v.label == "v3"
    assert v.model_uri == "gs://b/models/run-1/model/model.joblib"


def test_resolve_missing_alias_returns_none():
    assert registry.resolve(MODEL, "production", fetch=fake_fetch({})) is None


def test_rejects_non_resource_names():
    with pytest.raises(ValueError):
        registry.location_of("risk-classifier")


def test_api_loads_the_alias_version(tmp_path, tiny_model, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    joblib.dump(tiny_model, model_dir / "model.joblib")
    monkeypatch.setenv("REGISTRY_MODEL", MODEL)
    monkeypatch.setenv("MODEL_ALIAS", "production")

    loaded = main.load_model(fetch=fake_fetch({"production": ("7", str(model_dir))}))
    assert loaded.version == "v7"
    assert loaded.uri.endswith("model.joblib")


def test_api_fails_fast_when_alias_is_missing(monkeypatch):
    monkeypatch.setenv("REGISTRY_MODEL", MODEL)
    with pytest.raises(RuntimeError, match="matches 'production'"):
        main.load_model(fetch=fake_fetch({}))


def test_pinned_version_wins_over_alias(tmp_path, tiny_model, monkeypatch):
    joblib.dump(tiny_model, tmp_path / "model.joblib")
    monkeypatch.setenv("REGISTRY_MODEL", MODEL)
    monkeypatch.setenv("MODEL_ALIAS", "production")
    monkeypatch.setenv("MODEL_VERSION_ID", "4")
    fetch = fake_fetch({"4": ("4", str(tmp_path)), "production": ("9", "gs://unused")})
    assert main.load_model(fetch=fetch).version == "v4"
    assert fetch.calls[-1].endswith("@4")
