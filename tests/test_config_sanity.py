from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with path.open() as f:
        return yaml.safe_load(f)


def test_direct_horizon_config_selects_the_new_prediction_mode():
    cfg = load_yaml(ROOT / "config/train/lewm_mh.yaml")

    assert cfg["wm"]["type"] == "lewm_direct_h"
    assert cfg["wm"]["prediction_mode"] == "direct_horizon"
    assert cfg["wm"]["history_size"] == 3
    assert cfg["wm"]["horizon"] == 5
    assert cfg["wm"]["num_preds"] == "${wm.horizon}"
    assert cfg["loss"]["horizon_weights"]["schedule"] == "uniform"


def test_default_training_config_keeps_original_autoregressive_mode():
    cfg = load_yaml(ROOT / "config/train/lewm.yaml")

    assert cfg["wm"]["type"] == "lewm"
    assert cfg["wm"]["prediction_mode"] == "autoregressive"
    assert cfg["wm"]["history_size"] == 3
    assert cfg["wm"]["horizon"] == 1
    assert cfg["wm"]["num_preds"] == 1
    assert cfg["wandb"]["enabled"] == "auto"
    assert cfg["resume"]["mode"] == "auto"
    assert cfg["checkpoint"]["save_best"] is True
    assert cfg["checkpoint"]["enabled"] is True
    assert cfg["early_stopping"]["enabled"] is True
    assert cfg["trainer"]["max_epochs"] == 10
    assert cfg["runtime"]["gpu_image_preprocess"] is True
    assert cfg["scheduler"]["type"] == "LinearWarmupCosineAnnealingLR"
    assert cfg["scheduler"]["interval"] == "step"


def test_speed_training_config_is_bounded_for_short_gpu_probes():
    cfg = load_yaml(ROOT / "config/train/lewm_speed.yaml")

    assert cfg["trainer"]["max_steps"] == 300
    assert cfg["checkpoint"]["enabled"] is False
    assert cfg["resume"]["mode"] == "never"
    assert cfg["wandb"]["enabled"] == "auto"
    assert cfg["wandb"]["config"]["name"] == "${output_model_name}_speed"


def test_prediction_visualization_has_no_frame_retrieval_config():
    cfg = load_yaml(ROOT / "config/visualize/predictions.yaml")

    assert "retrieval" not in cfg
    assert cfg["decoder"]["path"] is None


def test_reacher_training_configs_use_reported_dataset_name():
    for name in ("dmc", "reacher"):
        cfg = load_yaml(ROOT / "config/train/data" / f"{name}.yaml")

        assert cfg["dataset"]["name"] == "dmc/reacher_random"


def test_tworoom_training_config_uses_reported_dataset_columns():
    cfg = load_yaml(ROOT / "config/train/data/tworoom.yaml")

    assert cfg["dataset"]["name"] == "tworoom"
    assert cfg["dataset"]["frameskip"] == 5
    assert cfg["dataset"]["keys_to_load"] == ["pixels", "action", "proprio"]
    assert cfg["dataset"]["keys_to_cache"] == ["action", "proprio"]


def test_scene_training_config_uses_visual_ogbench_scene_dataset():
    cfg = load_yaml(ROOT / "config/train/data/scene.yaml")

    assert cfg["dataset"]["name"] == "ogbench/visual_scene_play"
    assert cfg["dataset"]["frameskip"] == 5
    assert cfg["dataset"]["keys_to_load"] == ["pixels", "action"]
    assert cfg["dataset"]["keys_to_cache"] == ["action"]


def test_scene_eval_config_sets_scene_state_and_goal_targets():
    cfg = load_yaml(ROOT / "config/eval/scene.yaml")

    assert cfg["world"]["env_name"] == "swm/OGBScene-v0"
    assert cfg["world"]["width"] == 64
    assert cfg["world"]["height"] == 64
    assert cfg["eval"]["dataset_name"] == "ogbench/visual_scene_play"
    methods = [item["method"] for item in cfg["eval"]["callables"]]
    assert methods == [
        "set_state",
        "set_cube_target_pos",
        "set_target_button_state",
        "set_target_button_state",
        "set_target_drawer_pos",
        "set_target_window_pos",
    ]
    set_state_args = cfg["eval"]["callables"][0]["args"]
    assert set_state_args["button_state_0"]["value"] == "button_state_0"
    assert set_state_args["button_state_1"]["value"] == "button_state_1"
    assert cfg["eval"]["callables"][1]["args"]["target_pos"]["value"] == (
        "goal_privileged_block_0_pos"
    )
    assert cfg["output"]["save_video"] is False


def test_scene_official_eval_config_uses_fixed_ogbench_tasks():
    cfg = load_yaml(ROOT / "config/eval/scene_official.yaml")

    assert cfg["eval"]["protocol"] == "official_scene"
    assert cfg["eval"]["env_template"] == "visual-scene-singletask-task{task_id}-v0"
    assert cfg["eval"]["task_ids"] == [1, 2, 3, 4, 5]
    assert cfg["eval"]["episode_start"] == 0
    assert cfg["eval"]["max_episode_steps"] == 750
    assert cfg["eval"]["dataset_name"] == "ogbench/visual_scene_play"
    assert cfg["dataset"]["keys_to_cache"] == ["action"]
