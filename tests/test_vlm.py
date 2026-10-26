import json

import pytest

from cortex.vlm import LabelParseError, VlmLabeler
from shared.episodes import EpisodeRecord, make_episode_record
from shared.labels import FailureMode, FoldPhase


class _FixedVlm(VlmLabeler):
    def __init__(self, payload: str) -> None:
        super().__init__("fake-vlm")
        self.payload = payload
        self.seen_prompt = ""

    def complete(self, prompt: str, image_refs: tuple[str, ...]) -> str:
        self.seen_prompt = prompt
        return self.payload


def _episode() -> EpisodeRecord:
    return make_episode_record(
        episode_index=2,
        task="Fold the towel",
        fps=30,
        states=[[0.0] * 12, [1.0] * 12],
        actions=[[0.0] * 12, [1.0] * 12],
        image_refs=("frame-0.jpg",),
    )


def test_prompt_lists_the_taxonomy_and_parser_accepts_json() -> None:
    payload = {
        "phases": [
            {"phase": "grasp_corner", "start_frame": 0, "end_frame": 1, "confidence": 0.8},
            {"phase": "flatten", "start_frame": 1, "end_frame": 2, "confidence": 0.7},
        ],
        "failures": [
            {"mode": "missed_grasp", "start_frame": 0, "end_frame": 1, "detail": "empty hand"}
        ],
    }
    labeler = _FixedVlm(json.dumps(payload))
    episode = _episode()
    annotation = labeler.label(episode)
    assert "grasp_corner" in labeler.seen_prompt
    assert "double_layer_grasp" in labeler.seen_prompt
    assert annotation.labeler == "vlm:fake-vlm"
    assert annotation.phases[0].phase == FoldPhase.GRASP_CORNER
    assert annotation.failures[0].mode == FailureMode.MISSED_GRASP


def test_fenced_json_parses_and_prose_does_not() -> None:
    body = json.dumps(
        {
            "phases": [{"phase": "fold", "start_frame": 0, "end_frame": 2, "confidence": 0.4}],
            "failures": [],
        }
    )
    labeler = _FixedVlm(f"```json\n{body}\n```")
    annotation = labeler.label(_episode())
    assert annotation.failures == ()
    labeler.payload = "I think it folded."
    with pytest.raises(LabelParseError):
        labeler.label(_episode())
