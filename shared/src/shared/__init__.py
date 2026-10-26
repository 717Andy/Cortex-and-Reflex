"""Contracts shared by the Cortex data engine and the Reflex policy stack."""

from shared.contract import (
    ACTION_DIM,
    BIMANUAL_JOINT_NAMES,
    CAMERA_FEATURE_KEYS,
    CAMERA_ROLES,
    DATA_PATH_TEMPLATE,
    PUBLIC_TOWEL_REPO_ID,
    VIDEO_PATH_TEMPLATE,
    DatasetContractError,
    DatasetInfo,
    FeatureSpec,
    Severity,
    ValidationReport,
    make_dataset_info,
    validate_dataset_info,
    validate_feature_stats,
)
from shared.episodes import EpisodeRecord, make_episode_record
from shared.labels import (
    EpisodeAnnotation,
    FailureEvent,
    FailureMode,
    FoldPhase,
    PhaseSpan,
    annotation_from_mapping,
)
from shared.splits import EpisodeSplit, load_episode_split, save_episode_split

__version__ = "0.1.0"

__all__ = [
    "ACTION_DIM",
    "BIMANUAL_JOINT_NAMES",
    "CAMERA_FEATURE_KEYS",
    "CAMERA_ROLES",
    "DATA_PATH_TEMPLATE",
    "PUBLIC_TOWEL_REPO_ID",
    "VIDEO_PATH_TEMPLATE",
    "DatasetContractError",
    "DatasetInfo",
    "EpisodeAnnotation",
    "EpisodeRecord",
    "EpisodeSplit",
    "FailureEvent",
    "FailureMode",
    "FeatureSpec",
    "FoldPhase",
    "PhaseSpan",
    "Severity",
    "ValidationReport",
    "annotation_from_mapping",
    "load_episode_split",
    "make_dataset_info",
    "make_episode_record",
    "save_episode_split",
    "validate_dataset_info",
    "validate_feature_stats",
]
