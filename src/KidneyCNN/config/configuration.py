import os
from KidneyCNN.constants import *
from KidneyCNN.utils.common import read_yaml, create_directories
from KidneyCNN.entity.config_entity import (DataIngestionConfig,
                                            DataTransformationConfig,
                                            PrepareBaseModelConfig,
                                            TrainingConfig,
                                            EvaluationConfig)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path


class ConfigurationManager:
    """
    Manages reading configuration and parameters from YAML files, creating necessary 
    directories, and returning configuration objects for each pipeline stage.
    """
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        # Read configuration settings from config.yaml and params.yaml
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Create the root directory for all artifacts (usually 'artifacts')
        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves and prepares configuration for the data ingestion stage.
        """
        config = self.config.data_ingestion

        # Create the root directory specific to the data ingestion artifact (e.g., 'artifacts/data_ingestion')
        create_directories([config.root_dir])

        # Map configuration values to the DataIngestionConfig data class
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Retrieves and prepares configuration for the data transformation (splitting) stage.
        """
        config = self.config.data_transformation
        params = self.params

        # Create the root directory specific to the data transformation artifact
        create_directories([config.root_dir])

        # Map configuration and parameters to the DataTransformationConfig data class
        data_transformation_config = DataTransformationConfig(
            # Paths (from config.yaml)
            root_dir = config.root_dir,
            # Output directory where train/val/test folders will be created
            data_split_dir = config.data_split_dir, 

            # Parameters (from params.yaml)
            train_ratio = params.TRAIN_RATIO,
            validation_ratio = params.VALIDATION_RATIO,
            test_ratio = params.TEST_RATIO,
            split_seed = params.SPLIT_SEED ,
        )

        return data_transformation_config
    

    def get_prepear_base_model_config(self) -> PrepareBaseModelConfig:
        """
        Retrieves and prepares configuration for the base model preparation stage.
        """
        config = self.config.prepare_base_model

        # Create the root directory specific to the base model preparation artifact
        create_directories([config.root_dir])

        # Map configuration and parameters to the PrepareBaseModelConfig data class
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir= Path(config.root_dir),
            base_model_path= Path(config.base_model_path),
            updated_base_model_path= Path(config.updated_base_model_path),
            # Model parameters (VGG16 configuration)
            params_image_size= self.params.IMAGE_SIZE,
            params_learning_rate= self.params.LEARNING_RATE,
            params_include_top= self.params.INCLUDE_TOP,
            params_weights= self.params.WEIGHTS,
            params_classes= self.params.CLASSES
        )

        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        """
        Provides the configuration for the model training stage, including paths 
        to data and model, hyperparameters, and instantiated Keras callbacks.
        """
        # Get settings from main config and params files
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        
        # Define the path to the training data directory (inside the split directory)
        training_data = os.path.join(self.config.data_transformation.data_split_dir, "train")
        # Define the path to the validation data directory (inside the split directory)
        validation_data = os.path.join(self.config.data_transformation.data_split_dir, "val")
        
        # Create the training artifacts directory
        create_directories([Path(training.root_dir)])

        # Instantiate the callbacks here using parameters from params.yaml
        callbacks = {
            "early_stopping": EarlyStopping(
                # Monitors the metric specified in params.yaml
                monitor=params.CALLBACKS.early_stopping.monitor, 
                # Number of epochs with no improvement after which training will be stopped.
                patience=params.CALLBACKS.early_stopping.patience,
                verbose=params.CALLBACKS.early_stopping.verbose
            ),
            "model_checkpoint": ModelCheckpoint(
                # Save the model to this path
                filepath=Path(training.trained_model_path), 
                # Only save the model when the monitored metric improves
                save_best_only=params.CALLBACKS.model_checkpoint.save_best_only,
                monitor=params.CALLBACKS.model_checkpoint.monitor
            ),
            "reduce_lr_on_plateau": ReduceLROnPlateau(
                monitor=params.CALLBACKS.reduce_lr_on_plateau.monitor,
                # Factor by which the learning rate will be reduced
                factor=params.CALLBACKS.reduce_lr_on_plateau.factor,
                # Number of epochs with no improvement after which learning rate will be reduced.
                patience=params.CALLBACKS.reduce_lr_on_plateau.patience
            )
        }
        
        # Create a TrainingConfig object with all necessary parameters
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            validation_data=Path(validation_data),
            params_epochs=params.EPOCHS, 
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION, 
            params_image_size=params.IMAGE_SIZE, 
            params_learning_rate=params.LEARNING_RATE, 
            # Pass the dictionary of instantiated callback objects
            params_callbacks=callbacks 
        )

        return training_config
    

    def get_evalution_config(self) -> EvaluationConfig:
        """
        Provides the configuration for the model evaluation stage, primarily for generating 
        metrics and logging to MLflow.
        """
        # Define the path to the validation data directory (used as evaluation data)
        validation_data = os.path.join(self.config.data_transformation.data_split_dir, "val")
        
        # Create an EvaluationConfig object with the relevant settings
        eval_config = EvaluationConfig(
            # Hardcoded path to the trained model artifact
            path_of_model="artifacts/training/model.h5", 
            validation_data=Path(validation_data), 
            # MLflow tracking server URI
            mlflow_uri="https://dagshub.com/proshanta000/kidney_disease_classification.mlflow", 
            # Pass all parameters for logging to MLflow
            all_params= self.params, 
            params_image_size= self.params.IMAGE_SIZE, 
            params_batch_size= self.params.BATCH_SIZE 
        )
        return eval_config
