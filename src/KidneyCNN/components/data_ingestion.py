import os
import zipfile
import gdown
from pathlib import Path
from KidneyCNN import logger
from KidneyCNN.utils.common import get_size
from KidneyCNN.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self)-> str:
        # The file will only be download if it dosn't already exist
        if not os.path.exists(self.config.local_data_file):
            logger.info("Starting file download...")
            '''
            Fetch data from url 
            '''
            try:
                dataset_url = self.config.source_URL
                Zip_download_dir = self.config.local_data_file
                os.makedirs("artifacts/data_ingestion", exist_ok=True)
                logger.info(f"Downloading data from {dataset_url} into file {Zip_download_dir}")

                file_id= dataset_url.split("/")[-2]
                prefix = "https://drive.google.com/uc?/export=download&id="
                gdown.download(prefix+file_id, Zip_download_dir)

                logger.info(f"Downloading data from {dataset_url} into file {Zip_download_dir}")

            except Exception as e:
                raise e
        

        else:
            # Original logic for file already existing
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
    
    def extract_zip_file(self):
        """
            Zip_file_path: str
            Extract the zip file in to data diractory
            Function return None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)