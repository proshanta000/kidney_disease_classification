from KidneyCNN.constants import *
from KidneyCNN.utils.common import read_yaml, create_directories
from KidneyCNN.entity.config_entity import (DataIngestionConfig,
                                            DataTransformationConfig,
                                            PrepareBaseModelConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            # Paths (from config.yaml)
            # Input: Directory containing the 4 class subfolders
            root_dir = config.root_dir,
            data_split_dir = config.data_split_dir,

            # Output: Directory where the train/val/,
            
            # Parameters (from params.yaml)
            train_ratio = params.TRAIN_RATIO,
            validation_ratio = params.VALIDATION_RATIO,
            test_ratio = params.TEST_RATIO,
            split_seed = params.SPLIT_SEED ,
        )

        return data_transformation_config
    

    def get_prepear_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir= Path(config.root_dir),
            base_model_path= Path(config.base_model_path),
            updated_base_model_path= Path(config.updated_base_model_path),
            params_image_size= self.params.IMAGE_SIZE,
            params_learning_rate= self.params.LEARNING_RATE,
            params_include_top= self.params.INCLUDE_TOP,
            params_weights= self.params.WEIGHTS,
            params_classes= self.params.CLASSES
        )

        return prepare_base_model_config