import os
import zipfile
import gdown
from pathlib import Path
from KidneyCNN import logger
from KidneyCNN.utils.common import get_size
from KidneyCNN.entity.config_entity import DataIngestionConfig

# Defines the DataIngestion class responsible for fetching and preparing the raw data.
class DataIngestion:
    """
    Handles the downloading of data from a source URL (typically Google Drive) 
    and the extraction of the downloaded zip file into the artifacts directory.
    """
    def __init__(self, config: DataIngestionConfig):
        # Initializes the DataIngestion component with configuration settings 
        # provided via a DataIngestionConfig object.
        self.config = config
    
    def download_file(self)-> str:
        # Checks if the local data file already exists to prevent redundant downloads.
        if not os.path.exists(self.config.local_data_file):
            logger.info("Starting file download...")
            '''
            Fetches data from the specified Google Drive URL.
            '''
            try:
                dataset_url = self.config.source_URL
                Zip_download_dir = self.config.local_data_file
                
                # Ensures the local directory for data ingestion artifacts is created.
                os.makedirs("artifacts/data_ingestion", exist_ok=True)
                logger.info(f"Downloading data from {dataset_url} into file {Zip_download_dir}")

                # Extracts the unique Google Drive file ID from the URL structure.
                file_id= dataset_url.split("/")[-2]
                # Constructs the necessary prefix for gdown to force the download from Google Drive.
                prefix = "https://drive.google.com/uc?/export=download&id="
                
                # Executes the download using gdown.
                gdown.download(prefix+file_id, Zip_download_dir)

                # Logs successful completion of the download process.
                logger.info(f"Successfully downloaded data to {Zip_download_dir}") 

            except Exception as e:
                # Propagates any exceptions encountered during the download process.
                raise e
        
        else:
            # If the file exists, log its size to confirm its presence.
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
    
    def extract_zip_file(self):
        """
        Extracts the zip file contents to the configured unzip directory.
        
        Zip_file_path: str (Implicitly self.config.local_data_file)
        Extract the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        # Ensures the destination directory for the unzipped files exists.
        os.makedirs(unzip_path, exist_ok=True)
        
        # Opens the zip file and extracts all contents.
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
