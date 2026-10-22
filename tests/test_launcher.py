from datetime import datetime, timedelta, timezone

import pytest

from src.launcher import JobInfo, launch

NOW = datetime(2026, 10, 22, 6, 0, tzinfo=timezone.utc)


class FakeApi:
    def __init__(self, jobs=()):
        self.jobs = list(jobs)
        self.submitted = []

    def list_jobs(self):
        return self.jobs

    def submit(self, display_name, parameters, labels):
        self.submitted.append((display_name, parameters, labels))
        return f"projects/p/locations/l/pipelineJobs/{display_name}-1"


def job(state, hours_ago, trigger="schedule"):
    return JobInfo("projects/p/locations/l/pipelineJobs/x", state, NOW - timedelta(hours=hours_ago), {"trigger": trigger})


def test_weekly_run_is_submitted_with_production_alias():
    api = FakeApi([job("PIPELINE_STATE_SUCCEEDED", 168)])
    result = launch("schedule", api, now=NOW)
    assert result.submitted
    name, params, labels = api.submitted[0]
    assert name == "risk-train-eval-schedule"
    assert params == {"alias": "production"}
    assert labels == {"pipeline": "risk-train-eval", "trigger": "schedule"}


def test_never_submits_while_a_run_is_active():
    api = FakeApi([job("PIPELINE_STATE_RUNNING", 1)])
    result = launch("schedule", api, now=NOW)
    assert not result.submitted and "already active" in result.reason
    assert api.submitted == []


def test_drift_respects_cooldown():
    api = FakeApi([job("PIPELINE_STATE_SUCCEEDED", 5, trigger="drift")])
    assert not launch("drift", api, now=NOW).submitted
    assert launch("drift", api, cooldown_hours=4, now=NOW).submitted


def test_cooldown_does_not_block_the_weekly_run():
    api = FakeApi([job("PIPELINE_STATE_SUCCEEDED", 5, trigger="drift")])
    assert launch("schedule", api, now=NOW).submitted


def test_unknown_trigger_is_rejected():
    with pytest.raises(ValueError):
        launch("cron", FakeApi(), now=NOW)
