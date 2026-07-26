from pathlib import Path

from industrial_defect.config import load_project_config


def test_project_config_contract() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "project.yaml"
    config = load_project_config(config_path)

    assert config.name == "industrial-defect-inference-service"
    assert config.seed == 42
    assert config.dataset.name == "Severstal Steel Defect Detection"
    assert config.dataset.task == "semantic_segmentation"
    assert config.dataset.splits == ("train", "val", "test")
    assert len(config.dataset.classes) == 4
    assert config.dataset.expected_annotation_rows == 7095
    assert sum(config.dataset.split_ratio) == 1.0
    assert config.training.model == "unet_resnet18"
    assert config.training.image_size == (1024, 256)
    assert config.training.batch == 8
    assert config.training.learning_rate == 0.001
    assert config.training.weight_decay == 0.0001
    assert config.training.num_workers == 4
    assert config.training.amp is True
    assert config.training.threshold == 0.5
    assert config.deployment.checkpoint.endswith("class_aware_p075_e05_best_unet_resnet18.pt")
    assert config.deployment.onnx_model.endswith("unet_resnet18_severstal.onnx")
    assert config.deployment.onnx_opset == 18
    assert config.deployment.threshold == 0.8
    assert config.deployment.input_name == "images"
    assert config.deployment.output_name == "logits"
    assert config.deployment.dynamic_batch is True
