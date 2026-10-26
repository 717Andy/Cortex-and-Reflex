"""Policy families this stack will train."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PolicyStatus(StrEnum):
    READY = "ready"
    PLANNED = "planned"


@dataclass(frozen=True)
class PolicyEntry:
    name: str
    status: PolicyStatus
    lerobot_type: str
    extra: str | None
    notes: str


POLICIES: dict[str, PolicyEntry] = {
    "act": PolicyEntry(
        name="act",
        status=PolicyStatus.READY,
        lerobot_type="act",
        extra=None,
        notes="Action Chunking Transformer. CPU smoke test, plus reflex.cli train.",
    ),
    "diffusion": PolicyEntry(
        name="diffusion",
        status=PolicyStatus.PLANNED,
        lerobot_type="diffusion",
        extra="diffusion",
        notes="Diffusion Policy. Needs the lerobot diffusion extra and a GPU for real runs.",
    ),
    "smolvla": PolicyEntry(
        name="smolvla",
        status=PolicyStatus.READY,
        lerobot_type="smolvla",
        extra="smolvla",
        notes="Full fine-tune config. CPU dry-run skips SmolVLM weights.",
    ),
    "pi0.5": PolicyEntry(
        name="pi0.5",
        status=PolicyStatus.READY,
        lerobot_type="pi05",
        extra="pi",
        notes="Full fine-tune config. CPU dry-run skips PaliGemma weights. GPU needs lerobot[pi].",
    ),
    "student": PolicyEntry(
        name="student",
        status=PolicyStatus.READY,
        lerobot_type="student",
        extra=None,
        notes="Distillation student. GPU config is 50-150M params. CPU bench is tiny.",
    ),
}


def ready_policies() -> tuple[PolicyEntry, ...]:
    return tuple(entry for entry in POLICIES.values() if entry.status == PolicyStatus.READY)
