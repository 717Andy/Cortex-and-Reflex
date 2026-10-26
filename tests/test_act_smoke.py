import math
import re
from pathlib import Path

import pytest

from reflex.cli import main


@pytest.mark.smoke
def test_act_smoke_trains_one_cpu_step(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "ckpt"
    assert main(["act-smoke", "--steps", "1", "--output", str(output)]) == 0
    text = capsys.readouterr().out
    match = re.search(r"loss=([0-9eE.+-]+)", text)
    assert match is not None
    assert math.isfinite(float(match.group(1)))
    assert "(1, 12)" in text
    assert (output / "config.json").is_file()
