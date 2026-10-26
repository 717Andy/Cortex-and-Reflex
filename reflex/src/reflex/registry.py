"""Policy families this stack will train. Only ACT has a CPU smoke path today."""

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
        notes="Action Chunking Transformer. CPU smoke test trains one tiny step.",
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
        status=PolicyStatus.PLANNED,
        lerobot_type="smolvla",
        extra="smolvla",
        notes="SmolVLA. Needs the lerobot smolvla extra.",
    ),
    "pi0.5": PolicyEntry(
        name="pi0.5",
        status=PolicyStatus.PLANNED,
        lerobot_type="pi05",
        extra="pi",
        notes="pi0.5. A later stage distills this into a small student policy.",
    ),
}


def ready_policies() -> tuple[PolicyEntry, ...]:
    return tuple(entry for entry in POLICIES.values() if entry.status == PolicyStatus.READY)
