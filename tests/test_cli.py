from pathlib import Path

from cortex.cli import main as cortex_main
from cortex.labeling import MinerConfig, miner_config_from_mapping
from reflex.cli import _load_config
from reflex.registry import PolicyStatus, ready_policies

ROOT = Path(__file__).resolve().parents[1]


def test_label_demo_exits_clean() -> None:
    assert cortex_main(["label-demo"]) == 0


def test_local_ingest_cli(tmp_path: Path) -> None:
    from tests.support import write_local_dataset

    write_local_dataset(tmp_path, [12])
    assert cortex_main(["ingest", "--root", str(tmp_path), "--repo", "local/towel"]) == 0


def test_checked_in_configs_match_defaults() -> None:
    import yaml

    miner = yaml.safe_load((ROOT / "configs" / "labeling" / "stub_miner.yaml").read_text())
    assert miner_config_from_mapping(miner) == MinerConfig()
    smoke = _load_config(ROOT / "configs" / "policies" / "act_smoke.yaml")
    assert smoke.device == "cpu"
    assert smoke.use_vae is False
    assert smoke.steps == 1
    assert ready_policies()[0].name == "act"
    assert ready_policies()[0].status == PolicyStatus.READY
