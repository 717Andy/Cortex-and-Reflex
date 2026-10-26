from pathlib import Path

import pytest

from reflex.cli import main
from reflex.distill import (
    TARGET_VRAM_BYTES,
    StudentPolicy,
    benchmark_student,
    count_parameters,
    estimate_inference_vram_bytes,
    estimate_student_parameters,
    in_parameter_budget,
    load_distill_config,
)

ROOT = Path(__file__).resolve().parents[1]
CPU_CONFIG = ROOT / "configs" / "policies" / "student_towel_cpu.yaml"
GPU_CONFIG = ROOT / "configs" / "policies" / "student_towel_gpu.yaml"


def test_parameter_formula_matches_the_module() -> None:
    cpu = load_distill_config(CPU_CONFIG)
    gpu = load_distill_config(GPU_CONFIG)
    tiny = StudentPolicy(cpu.student)
    assert count_parameters(tiny) == estimate_student_parameters(cpu.student)
    assert estimate_student_parameters(gpu.student) == count_parameters(StudentPolicy(gpu.student))
    assert in_parameter_budget(estimate_student_parameters(gpu.student))
    assert estimate_inference_vram_bytes(gpu.student) < TARGET_VRAM_BYTES
    assert gpu.task_prompt == "Fold towel"
    assert gpu.teacher_kind == "checkpoint"


def test_cpu_distill_resumes_and_bench_hits_the_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "student"
    args = ["distill", "--config", str(CPU_CONFIG), "--output", str(output)]
    assert main(args) == 0
    first = capsys.readouterr().out
    assert "resumed=false" in first
    assert "step=1" in first
    assert main(args) == 0
    second = capsys.readouterr().out
    assert "resumed=true" in second
    assert "step=2" in second
    bench_code = main(["bench", "--config", str(CPU_CONFIG), "--steps", "10", "--strict"])
    bench = capsys.readouterr().out
    assert bench_code == 0
    assert "meets_hz=true" in bench
    assert "meets_vram=true" in bench
    measured = benchmark_student(load_distill_config(CPU_CONFIG).student, steps=5, warmup=1)
    assert measured.hertz >= 30
    assert measured.parameters == estimate_student_parameters(
        load_distill_config(CPU_CONFIG).student
    )
