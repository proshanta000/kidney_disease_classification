from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for the Data Ingestion stage."""
    root_dir: Path  # Root directory where data ingestion artifacts are stored
    source_URL: str # URL to download the raw data from
    local_data_file: Path # Local path to save the downloaded data file (e.g., a zip file)
    unzip_dir: Path # Directory where the downloaded data zip file will be extracted

@dataclass(frozen=True)
class DataTransformationConfig:
    """Configuration for the Data Transformation (splitting) stage."""
    root_dir: Path # Root directory for transformation artifacts
    data_split_dir: Path # Output directory where 'train', 'val', and 'test' folders will be created
    train_ratio: float # Ratio of data to be used for the training set
    validation_ratio: float # Ratio of data to be used for the validation set
    test_ratio: float # Ratio of data to be used for the test set
    split_seed: int # Random seed to ensure reproducible data split


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    """Configuration for preparing the pre-trained base model (e.g., VGG16)."""
    root_dir: Path # Root directory for base model preparation artifacts
    base_model_path: Path # Path to save the downloaded VGG16 model without the top layer
    updated_base_model_path: Path # Path to save the model after adding the custom classification head
    params_image_size: list # Target image size [height, width, channels] for the model input
    params_learning_rate: float # Learning rate for the new layers/fine-tuning
    params_include_top: bool # Flag indicating whether to include the top classification layer (should be False)
    params_weights: str # Pre-trained weights to use (e.g., "imagenet")
    params_classes: int # Number of output classes for the new classification head

@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for the model training stage.
    """
    root_dir: Path  # Root directory for training artifacts
    trained_model_path: Path  # Path to save the final trained model
    updated_base_model_path: Path  # Path to the base model from the preparation stage
    training_data: Path  # Path to the training dataset folder (e.g., artifacts/data_transformation/train)
    validation_data: Path # Path to the validation dataset folder (e.g., artifacts/data_transformation/val)
    params_epochs: int  # Number of training epochs
    params_batch_size: int  # Number of samples per batch
    params_is_augmentation: bool  # Flag to enable/disable data augmentation
    params_image_size: list  # Image dimensions for training
    params_learning_rate: float  # Learning rate for the optimizer
    params_callbacks: Dict[str, Any]  # Dictionary of instantiated callbacks (e.g., EarlyStopping, ModelCheckpoint)


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Configuration for the model evaluation stage.
    """
    path_of_model: Path  # Path to the trained model to be evaluated
    validation_data: Path  # Path to the validation dataset folder
    mlflow_uri: str  # MLflow tracking server URI to log metrics and model
    all_params: dict  # A dictionary containing all project parameters for logging to MLflow
    params_image_size: list  # Image dimensions for evaluation
    params_batch_size: int  # Batch size for evaluation
