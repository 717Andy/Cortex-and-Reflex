"""Run a labeler across an episode source and fill the review queue."""

from __future__ import annotations

from dataclasses import dataclass

from cortex.dashboard import Dashboard, build_dashboard
from cortex.labeling import FoldLabeler
from cortex.queue import DataSelectionQueue, enqueue_annotation
from cortex.sources import EpisodeSource
from shared.labels import EpisodeAnnotation


@dataclass(frozen=True)
class LabelingRun:
    annotations: tuple[EpisodeAnnotation, ...]
    queue: DataSelectionQueue
    dashboard: Dashboard


def run_labeling(source: EpisodeSource, labeler: FoldLabeler) -> LabelingRun:
    queue = DataSelectionQueue()
    annotations: list[EpisodeAnnotation] = []
    for index in range(len(source)):
        annotation = labeler.label(source.load_episode(index))
        annotations.append(annotation)
        enqueue_annotation(queue, annotation)
    dashboard = build_dashboard(annotations, queue)
    return LabelingRun(annotations=tuple(annotations), queue=queue, dashboard=dashboard)
