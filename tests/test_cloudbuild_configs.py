"""The Cloud Build configs are not executed by the PR build, so at least parse them here."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted(ROOT.glob("cloudbuild*.yaml")), ids=lambda p: p.name)
def test_cloudbuild_config_is_valid_yaml_with_steps(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["steps"], path.name
    for step in config["steps"]:
        assert "name" in step
        for arg in step.get("args", []):
            assert not str(arg).rstrip().endswith("\\n\\"), "stray line continuation"
