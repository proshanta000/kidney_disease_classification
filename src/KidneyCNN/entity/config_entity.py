from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL: str
    local_data_file:Path
    unzip_dir:Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_split_dir: Path
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    split_seed: int


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for the model training stage.
    """
    root_dir: Path  # Root directory for training artifacts
    trained_model_path: Path  # Path to save the final trained model
    updated_base_model_path: Path  # Path to the base model from the preparation stage
    training_data: Path  # Path to the training dataset
    validation_data: Path # Path to the validation dataset
    params_epochs: int  # Number of training epochs
    params_batch_size: int  # Number of samples per batch
    params_is_augmentation: bool  # Flag to enable/disable data augmentation
    params_image_size: list  # Image dimensions for training
    params_learning_rate: float  # Learning rate for the optimizer
    params_callbacks: Dict[str, Any]  # Dictionary of callbacks (e.g., EarlyStopping, ReduceLROnPlateau)


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Configuration for the model evaluation stage.
    """
    path_of_model: Path  # Path to the trained model to be evaluated
    validation_data: Path  # Path to the validation dataset
    mlflow_uri: str  # MLflow tracking server URI
    all_params: dict  # A dictionary containing all project parameters for logging
    params_image_size: list  # Image dimensions for evaluation
    params_batch_size: int  # Batch size for evaluation