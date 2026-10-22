"""Vertex AI Model Registry helpers shared by the API and the pipeline.

Reads go through the REST API with google-auth (light enough for the serving
image). Writes (uploading a version, moving an alias) use the Vertex AI SDK and
only run inside the pipeline image.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

ModelFetch = Callable[[str], Optional[dict]]

_RESOURCE = re.compile(r"^projects/[^/]+/locations/(?P<location>[^/]+)/models/[^/@]+$")


@dataclass
class RegistryVersion:
    model: str  # projects/.../models/<id>
    version_id: str
    artifact_uri: str
    labels: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Human/BigQuery-facing version string, e.g. 'v3'."""
        return f"v{self.version_id}"

    @property
    def model_uri(self) -> str:
        return self.artifact_uri.rstrip("/") + "/model.joblib"


def location_of(model: str) -> str:
    match = _RESOURCE.match(model)
    if not match:
        raise ValueError(f"Not a model resource name: {model!r}")
    return match.group("location")


def _authorized_get(url: str) -> Optional[dict]:
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    response = AuthorizedSession(credentials).get(url, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def resolve(model: str, ref: str, fetch: ModelFetch = _authorized_get) -> Optional[RegistryVersion]:
    """Version of `model` named by `ref` (a version id like "4" or an alias like
    "production"), or None if there is no such version."""
    url = f"https://{location_of(model)}-aiplatform.googleapis.com/v1/{model}@{ref}"
    body = fetch(url)
    if body is None:
        return None
    return RegistryVersion(
        model=model,
        version_id=str(body["versionId"]),
        artifact_uri=body.get("artifactUri", ""),
        labels=body.get("labels", {}),
    )


def model_exists(model: str, fetch: ModelFetch = _authorized_get) -> bool:
    return fetch(f"https://{location_of(model)}-aiplatform.googleapis.com/v1/{model}") is not None


def label_value(value: str) -> str:
    """Vertex label values: lowercase letters, digits, '_' and '-', at most 63 chars."""
    return re.sub(r"[^a-z0-9_-]", "-", str(value).lower())[:63]


def register_version(
    model: str,
    artifact_uri: str,
    alias: str,
    serving_image: str,
    description: str,
    labels: Optional[dict] = None,
    display_name: Optional[str] = None,
) -> RegistryVersion:
    """Upload artifact_uri as a new version of `model` and move `alias` onto it.

    If `model` does not exist yet, it is created with the id at the end of its resource
    name (e.g. .../models/invoice-extractor) and this upload becomes version 1.
    """
    from google.cloud import aiplatform

    project = model.split("/")[1]
    location = location_of(model)
    model_id = model.rsplit("/", 1)[1]
    aiplatform.init(project=project, location=location)

    exists = model_exists(model)
    previous = resolve(model, alias) if exists else None
    target = {"parent_model": model} if exists else {"model_id": model_id, "display_name": display_name or model_id}
    uploaded = aiplatform.Model.upload(
        **target,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_image,
        is_default_version=True,
        version_description=description,
        labels={k: label_value(v) for k, v in (labels or {}).items()},
        sync=True,
    )
    registry = aiplatform.models.ModelRegistry(model=model)
    if previous is not None and previous.version_id != uploaded.version_id:
        registry.remove_version_aliases(target_aliases=[alias], version=previous.version_id)
    registry.add_version_aliases(new_aliases=[alias], version=uploaded.version_id)
    return RegistryVersion(model=model, version_id=str(uploaded.version_id), artifact_uri=artifact_uri,
                           labels=labels or {})
