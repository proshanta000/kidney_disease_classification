import os
import shutil
import random
from pathlib import Path
import splitfolders # Library used for stratified data splitting

# Import the configuration entity for type hinting and structure definition
from KidneyCNN.entity.config_entity import DataTransformationConfig 


class DataTransformation:
    """
    Component class responsible for splitting the raw dataset into 
    train, validation, and test subsets using a stratified approach.
    """
    def __init__(self, config: DataTransformationConfig):
        # Initialize the component with the configuration object containing 
        # all necessary paths and parameters (ratios, seed).
        self.config = config


    def perform_split(self):
        """
        Executes the stratified split using the split-folders library.
        The data is split based on the ratios defined in the configuration.
        """
        
        # 1. Prepare input and output paths
        # The 'root_dir' (e.g., 'artifacts/data_ingestion') is configured to be the parent.
        # We append "dataset" to point to the actual directory containing the class folders (cyst, normal, etc.)
        input_path = Path(self.config.root_dir) / "dataset" 
        
        # Check if the input directory (containing the classes) exists before splitting.
        if not input_path.exists():
             # Using the correct Path object for checking and printing
             print(f"Error: Raw data directory not found at {input_path}. Ensure data ingestion ran successfully.")
             return
             
        # Ensure the output directory for the split data exists.
        os.makedirs(self.config.data_split_dir, exist_ok=True)
        
        # 2. Define ratios
        # Tuple containing (train_ratio, validation_ratio, test_ratio)
        ratios = (
            self.config.train_ratio, 
            self.config.validation_ratio, 
            self.config.test_ratio
        )

        # 3. Perform the split
        print(f"Starting stratified data split (Train:{ratios[0]}, Validation:{ratios[1]}, Test:{ratios[2]})")
        
        try:
            # Use splitfolders.ratio to perform the stratified split
            splitfolders.ratio(
                str(input_path), # The source directory containing the class folders
                output=str(self.config.data_split_dir), # The destination directory
                seed=self.config.split_seed, # Random seed for reproducibility
                ratio=ratios, # The split ratios defined above
                group_prefix=None,
                move=False # Files are copied, not moved
            )
            print(f"Data split successful. New structure created in: {self.config.data_split_dir}")

        except Exception as e:
            print(f"An error occurred during data splitting: {e}")
