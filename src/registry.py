"""Vertex AI Model Registry helpers shared by the API and the pipeline.

Reads go through the REST API with google-auth (light enough for the serving
image). Writes (uploading a version, moving an alias) use the Vertex AI SDK and
only run inside the pipeline image.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

ModelFetch = Callable[[str], Optional[dict]]

_RESOURCE = re.compile(r"^projects/[^/]+/locations/(?P<location>[^/]+)/models/[^/@]+$")


@dataclass
class RegistryVersion:
    model: str  # projects/.../models/<id>
    version_id: str
    artifact_uri: str

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
    return RegistryVersion(model=model, version_id=str(body["versionId"]), artifact_uri=body["artifactUri"])


def register_version(
    model: str,
    artifact_uri: str,
    alias: str,
    serving_image: str,
    description: str,
    labels: Optional[dict] = None,
) -> RegistryVersion:
    """Upload artifact_uri as a new version of `model` and move `alias` onto it."""
    from google.cloud import aiplatform

    project = model.split("/")[1]
    location = location_of(model)
    aiplatform.init(project=project, location=location)

    previous = resolve(model, alias)
    uploaded = aiplatform.Model.upload(
        parent_model=model,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_image,
        is_default_version=True,
        version_description=description,
        labels=labels or {},
        sync=True,
    )
    registry = aiplatform.models.ModelRegistry(model=model)
    if previous is not None and previous.version_id != uploaded.version_id:
        registry.remove_version_aliases(target_aliases=[alias], version=previous.version_id)
    registry.add_version_aliases(new_aliases=[alias], version=uploaded.version_id)
    return RegistryVersion(model=model, version_id=str(uploaded.version_id), artifact_uri=artifact_uri)
