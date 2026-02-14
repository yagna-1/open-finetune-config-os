from pathlib import Path

from ft_config_engine.recommender import _resolve_dataset_paths


def test_resolve_dataset_paths_auto_discovers_supported_auxiliary_jsonl(tmp_path: Path):
    primary = tmp_path / "finetuning_configs_final.jsonl"
    primary.write_text("{}\n", encoding="utf-8")

    real_world = tmp_path / "real_world_additions.jsonl"
    real_world.write_text("{}\n", encoding="utf-8")

    fixed_lower = tmp_path / "fixed.jsonl"
    fixed_lower.write_text("{}\n", encoding="utf-8")

    fixed_upper = tmp_path / "Fixed_extra.jsonl"
    fixed_upper.write_text("{}\n", encoding="utf-8")

    unrelated = tmp_path / "other_configs.jsonl"
    unrelated.write_text("{}\n", encoding="utf-8")

    resolved = _resolve_dataset_paths(primary)

    assert primary.resolve() in resolved
    assert real_world.resolve() in resolved
    assert fixed_lower.resolve() in resolved
    assert fixed_upper.resolve() in resolved
    assert unrelated.resolve() not in resolved

