import json
from pathlib import Path

from cortex.cli import main as cortex_main
from cortex.curation import CurationConfig, curate, load_annotations
from cortex.vlm import annotation_to_dict
from shared.labels import (
    EpisodeAnnotation,
    FailureEvent,
    FailureMode,
    FoldPhase,
    PhaseSpan,
    annotation_from_mapping,
)
from shared.splits import load_episode_split

ROOT = Path(__file__).resolve().parents[1]


def _annotation(
    index: int,
    *,
    success: bool,
    confidence: float,
    failure: FailureMode | None = None,
) -> EpisodeAnnotation:
    failures = ()
    if failure is not None:
        failures = (FailureEvent(failure, 0, 4, failure.value),)
    return EpisodeAnnotation(
        episode_index=index,
        labeler="vlm",
        phases=(PhaseSpan(FoldPhase.FOLD, 0, 4, confidence),),
        failures=failures,
        success=success,
        quality_notes="fold",
    )


def _pool() -> tuple[EpisodeAnnotation, ...]:
    return (
        _annotation(0, success=True, confidence=0.95),
        _annotation(1, success=True, confidence=0.5),
        _annotation(2, success=False, confidence=0.9, failure=FailureMode.DROP),
        _annotation(3, success=False, confidence=0.2, failure=FailureMode.MISSED_GRASP),
    )


def test_annotation_json_roundtrip() -> None:
    original = _pool()[2]
    restored = annotation_from_mapping(annotation_to_dict(original))
    assert restored.episode_index == 2
    assert restored.success is False
    assert restored.failures[0].mode is FailureMode.DROP
    assert restored.phases[0].confidence == 0.9


def test_quality_beats_the_failure_episodes() -> None:
    split = curate(_pool(), CurationConfig(strategy="quality", train_fraction=0.5, seed=0))
    assert split.train_episodes == (0, 1)
    assert split.held_out_episodes == (2, 3)
    assert split.comparison.selected_mean_quality > split.comparison.random_mean_quality
    assert 0.0 <= split.comparison.train_overlap_with_random <= 1.0


def test_diversity_includes_a_failure_and_reward_ranks_scores() -> None:
    diversity = curate(_pool(), CurationConfig(strategy="diversity", train_fraction=0.5, seed=0))
    assert 0 in diversity.train_episodes
    assert diversity.train_episodes != (0, 1)

    rewards = {0: 0.2, 1: 0.1, 2: 0.5, 3: 1.0}
    ranked = curate(
        _pool(),
        CurationConfig(strategy="reward", train_fraction=0.5, seed=0),
        rewards,
    )
    assert ranked.train_episodes == (2, 3)
    assert ranked.comparison.selected_mean_reward == 0.75
    assert ranked.scores[3].reward == 1.0


def test_curate_cli_writes_a_training_split(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.jsonl"
    lines = [json.dumps(annotation_to_dict(item)) for item in _pool()]
    annotations.write_text("\n".join(lines) + "\n")
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"0": 0.2, "1": 0.1, "2": 0.4, "3": 0.9}) + "\n")
    output = tmp_path / "split.json"
    code = cortex_main(
        [
            "curate",
            "--annotations",
            str(annotations),
            "--output",
            str(output),
            "--config",
            str(ROOT / "configs" / "curation" / "quality.yaml"),
            "--strategy",
            "reward",
            "--reward-scores",
            str(scores),
            "--train-fraction",
            "0.5",
            "--dataset",
            "humjie/bimanual-so101-fold-towel",
        ]
    )
    assert code == 0
    loaded = load_episode_split(output)
    assert loaded.strategy == "reward"
    assert loaded.train_episodes == (2, 3)
    assert load_annotations(annotations)[0].success is True
