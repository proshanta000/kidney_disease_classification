import os
import shutil
import random
from pathlib import Path
import splitfolders

from KidneyCNN.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def perform_split(self):
        """
        Executes the stratified split using the split-folders library.
        """
        
        # 1. Prepare input and output paths
        # Assuming the 4 class folders are directly inside the 'raw_data_dir' 
        
        input_path = Path(self.config.root_dir) / "dataset" 
        
        # Check if the input directory exists and has subfolders (classes)
        if not input_path.exists():
             # Using the correct Path object for checking and printing
             print(f"Error: Raw data directory not found at {input_path}. Ensure data ingestion ran successfully.")
             return
             
        # Ensure the output directory is created (splitfolders does this, but good practice)
        os.makedirs(self.config.data_split_dir, exist_ok=True)
        
        # 2. Define ratios
        ratios = (
            self.config.train_ratio, 
            self.config.validation_ratio, 
            self.config.test_ratio
        )

        # 3. Perform the split
        print(f"Starting stratified data split (Train:{ratios[0]}, Validation:{ratios[1]}, Test:{ratios[2]})")
        
        try:
            splitfolders.ratio(
                str(input_path), 
                output=str(self.config.data_split_dir),
                seed=self.config.split_seed, 
                ratio=ratios,
                group_prefix=None,
                move=False # Set to True if you want to move files instead of copying
            )
            print(f"Data split successful. New structure created in: {self.config.data_split_dir}")

        except Exception as e:
            print(f"An error occurred during data splitting: {e}")