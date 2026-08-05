import pytest

from web.jobs import InvalidTransition, Job, JobState


def make(tmp_path) -> Job:
    return Job(id="abc123", video=tmp_path / "raw.mp4", script="hello")


def test_a_new_job_is_queued(tmp_path):
    assert make(tmp_path).state is JobState.QUEUED


def test_the_happy_path_runs_in_order(tmp_path):
    job = make(tmp_path)
    for state in (JobState.PROBING, JobState.TRANSCRIBING, JobState.ANALYSING,
                  JobState.READY, JobState.RENDERING, JobState.DONE):
        job.advance(state)
    assert job.state is JobState.DONE


def test_skipping_a_state_is_rejected(tmp_path):
    job = make(tmp_path)
    with pytest.raises(InvalidTransition):
        job.advance(JobState.READY)


def test_ready_cannot_jump_straight_to_done(tmp_path):
    """The review gate. Rendering must be asked for, never assumed."""
    job = make(tmp_path)
    for state in (JobState.PROBING, JobState.TRANSCRIBING, JobState.ANALYSING,
                  JobState.READY):
        job.advance(state)
    with pytest.raises(InvalidTransition):
        job.advance(JobState.DONE)


def test_failure_is_reachable_from_any_state(tmp_path):
    from web.jobs import ALLOWED_TRANSITIONS

    for reached in (JobState.QUEUED, JobState.TRANSCRIBING, JobState.READY):
        job = make(tmp_path)
        while job.state is not reached:
            job.advance(next(iter(ALLOWED_TRANSITIONS[job.state])))
        job.fail("boom")
        assert job.state is JobState.FAILED
        assert job.error == "boom"


def test_terminal_states_go_nowhere(tmp_path):
    job = make(tmp_path)
    job.fail("boom")
    with pytest.raises(InvalidTransition):
        job.advance(JobState.PROBING)


def test_as_dict_carries_what_the_ui_polls_for(tmp_path):
    job = make(tmp_path)
    job.progress = 0.42
    payload = job.as_dict()
    assert payload["id"] == "abc123"
    assert payload["state"] == "queued"
    assert payload["progress"] == 0.42
    assert payload["report"] is None
    assert payload["error"] is None
