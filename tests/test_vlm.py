import json
from pathlib import Path

import pytest

from cortex.pipeline import run_labeling
from cortex.sources import InMemoryEpisodeSource
from cortex.vlm import BackendVlmLabeler, VlmLabeler
from cortex.vlm_backend import FakeVlm, ResponseCache
from cortex.vlm_schema import LabelParseError, cover_response
from shared.episodes import EpisodeRecord, make_episode_record
from shared.labels import FailureMode, FoldPhase, SubTask


class _FixedVlm(VlmLabeler):
    def __init__(self, payload: str) -> None:
        super().__init__("fake-vlm")
        self.payload = payload
        self.seen_prompt = ""

    def complete(self, prompt: str, image_refs: tuple[str, ...]) -> str:
        self.seen_prompt = prompt
        return self.payload


def _episode(num_frames: int = 2) -> EpisodeRecord:
    states = [[float(index)] * 12 for index in range(num_frames)]
    return make_episode_record(
        episode_index=2,
        task="Fold the towel",
        fps=30,
        states=states,
        actions=states,
        image_refs=("frame-0.jpg",),
    )


def test_prompt_lists_segments_and_parser_accepts_schema() -> None:
    payload = cover_response(
        5,
        confidence=0.8,
        success=False,
        quality_notes="corner slipped",
        failures=[
            {"mode": "missed_grasp", "start_frame": 0, "end_frame": 1, "detail": "empty hand"}
        ],
    )
    labeler = _FixedVlm(json.dumps(payload))
    episode = _episode(5)
    annotation = labeler.label(episode)
    assert "fold1" in labeler.seen_prompt
    assert "fold2" in labeler.seen_prompt
    assert "quality_notes" in labeler.seen_prompt
    assert "double_layer_grasp" in labeler.seen_prompt
    assert annotation.labeler == "vlm:fake-vlm"
    assert annotation.success is False
    assert annotation.quality_notes == "corner slipped"
    assert annotation.segments[0].phase == SubTask.GRASP
    assert annotation.phases[0].phase == FoldPhase.GRASP_CORNER
    assert annotation.failures[0].mode == FailureMode.MISSED_GRASP
    assert annotation.min_phase_confidence == 0.8


def test_fenced_json_parses_and_prose_does_not() -> None:
    body = json.dumps(cover_response(2, confidence=0.4, success=True, quality_notes="flat enough"))
    labeler = _FixedVlm(f"```json\n{body}\n```")
    annotation = labeler.label(_episode())
    assert annotation.failures == ()
    assert annotation.success is True
    labeler.payload = "I think it folded."
    with pytest.raises(LabelParseError):
        labeler.label(_episode())


def test_legacy_phase_json_still_parses() -> None:
    payload = {
        "phases": [{"phase": "fold", "start_frame": 0, "end_frame": 2, "confidence": 0.4}],
        "failures": [],
    }
    annotation = _FixedVlm(json.dumps(payload)).label(_episode())
    assert annotation.phases[0].phase == FoldPhase.FOLD
    assert annotation.segments == ()
    assert annotation.success is None


def test_segments_must_cover_every_frame() -> None:
    payload = {
        "segments": [
            {"phase": "grasp", "start_frame": 0, "end_frame": 1, "confidence": 0.5},
        ],
        "success": False,
        "quality_notes": "stopped early",
        "failures": [],
    }
    with pytest.raises(LabelParseError, match="episode has"):
        _FixedVlm(json.dumps(payload)).label(_episode())


def test_schema_retry_then_cache(tmp_path: Path) -> None:
    episode = _episode(4)
    valid = json.dumps(
        cover_response(4, confidence=0.91, success=True, quality_notes="both folds landed")
    )
    backend = FakeVlm(script=("not json", valid))
    cache = ResponseCache(tmp_path)
    labeler = BackendVlmLabeler(backend, attempts=2, cache=cache)
    first = labeler.label(episode)
    assert backend.calls == 2
    assert "failed validation" in backend.seen_prompts[1]
    assert first.success is True
    assert first.min_phase_confidence == 0.91
    second = labeler.label(episode)
    assert backend.calls == 2
    assert second.quality_notes == first.quality_notes


def test_low_confidence_and_failure_feed_the_review_queue() -> None:
    episode = _episode(5)
    uncertain = BackendVlmLabeler(FakeVlm(confidence=0.42, success=False))
    run = run_labeling(InMemoryEpisodeSource((episode,)), uncertain)
    reasons = {item.reason.value for item in run.queue.pending()}
    assert reasons == {"failure", "uncertain_phase"}
    details = {item.detail for item in run.queue.pending()}
    assert any(detail.startswith("success=false") for detail in details)
    assert any(detail == "min_confidence=0.42" for detail in details)

    confident = BackendVlmLabeler(FakeVlm(confidence=0.9, success=True))
    clean = run_labeling(InMemoryEpisodeSource((episode,)), confident)
    assert clean.queue.pending() == []
