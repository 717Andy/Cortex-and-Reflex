from cortex.dashboard import build_dashboard, render_dashboard
from cortex.labeling import StubLabeler
from cortex.pipeline import run_labeling
from cortex.queue import DataSelectionQueue, SelectionReason, enqueue_annotation
from cortex.sources import InMemoryEpisodeSource
from shared.episodes import make_episode_record
from shared.labels import EpisodeAnnotation, FailureEvent, FailureMode, FoldPhase, PhaseSpan


def test_failures_outrank_uncertain_phases() -> None:
    annotation = EpisodeAnnotation(
        episode_index=4,
        labeler="stub",
        phases=(PhaseSpan(FoldPhase.GRASP_CORNER, 0, 2, 0.2),),
        failures=(FailureEvent(FailureMode.DROP, 0, 2, "opened early"),),
    )
    queue = DataSelectionQueue()
    enqueue_annotation(queue, annotation)
    enqueue_annotation(queue, annotation)
    pending = queue.pending()
    assert len(pending) == 2
    first = queue.pop_next()
    assert first is not None
    assert first.reason == SelectionReason.FAILURE
    queue.mark(first, "reviewed")
    second = queue.pop_next()
    assert second is not None
    assert second.reason == SelectionReason.UNCERTAIN_PHASE


def test_pipeline_dashboard_lists_the_stub_failure() -> None:
    states = [[0.0] * 12 for _ in range(40)]
    episode = make_episode_record(
        episode_index=0,
        task="Fold the towel",
        fps=30,
        states=states,
        actions=states,
    )
    run = run_labeling(InMemoryEpisodeSource((episode,)), StubLabeler())
    assert run.dashboard.failure_counts["missed_grasp"] == 1
    assert run.dashboard.queue_pending >= 1
    text = render_dashboard(build_dashboard(list(run.annotations), run.queue))
    assert "Cortex dashboard" in text
    assert "missed_grasp: 1" in text
