"""Turn episode image refs into pictures a VLM can read.

Refs that are not local image or video files are left in the prompt and are
not attached. Video decode is best-effort: the public towel set is AV1, and a
CPU torch install may not decode it. A failure to decode skips those frames
instead of failing the label run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov"}
_MAX_IMAGES = 6


def load_label_images(image_refs: tuple[str, ...], *, frames_per_video: int = 2) -> list[Any]:
    """Open up to ``_MAX_IMAGES`` RGB frames from local paths."""
    if frames_per_video < 1:
        raise ValueError("frames_per_video must be >= 1")
    try:
        from PIL import Image
    except ImportError:
        return []
    images: list[Any] = []
    for ref in image_refs:
        if len(images) >= _MAX_IMAGES:
            break
        path = Path(ref)
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            images.append(Image.open(path).convert("RGB"))
            continue
        if suffix in _VIDEO_SUFFIXES:
            remaining = _MAX_IMAGES - len(images)
            images.extend(_sample_video(path, min(frames_per_video, remaining)))
    return images


def _sample_video(path: Path, count: int) -> list[Any]:
    if count < 1:
        return []
    try:
        from PIL import Image
        from torchvision.io import read_video
    except ImportError:
        return []
    try:
        frames, _audio, _info = read_video(str(path), pts_unit="sec")
    except (OSError, RuntimeError, ValueError):
        return []
    total = int(frames.shape[0]) if hasattr(frames, "shape") else 0
    if total < 1:
        return []
    if total == 1:
        indices = [0]
    else:
        indices = [0]
        if count > 1:
            indices.append(total // 2)
    images: list[Any] = []
    for index in indices[:count]:
        frame = frames[index]
        array = frame.numpy() if hasattr(frame, "numpy") else frame
        images.append(Image.fromarray(array))
    return images
