from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_dir: str = "../data/raw"
    model_name: str = "resnet50"
    num_classes: int = 3
    batch_size: int = 16
    num_epochs: int = 10
    lr: float = 1e-4
    seed: int = 42
    save_path: str = "model_best.onnx"
