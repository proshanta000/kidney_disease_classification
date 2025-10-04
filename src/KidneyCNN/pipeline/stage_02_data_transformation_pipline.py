# Import necessary modules
from KidneyCNN.config.configuration import ConfigurationManager # Imports the class responsible for reading configurations
from KidneyCNN.components.data_transformation import DataTransformation # Imports the component class that handles data splitting
from KidneyCNN import logger # Imports the custom logger for logging events and progress


# Define a constant for the stage name, used in logging
STAGE_NAME = "Data Transformation model"

# Define the main pipeline class for this stage
class DataTransformationPipeline:
    """
    Pipeline class dedicated to the Data Transformation stage, which involves 
    splitting the raw data into train, validation, and test sets.
    """
    def __init__(self):
        # Constructor for the pipeline class.
        pass

    def main(self):
        """
        Executes the main logic for the 'Data Transformation' pipeline stage.
        
        This method gets the configuration, creates the component instance, and runs its
        primary function to perform the stratified data split.
        """
        # 1. Initialize the Configuration Manager
        config = ConfigurationManager()
        
        # 2. Get the specific configuration for Data Transformation
        data_transformation_config = config.get_data_transformation_config()
        
        # 3. Initialize the Data Transformation component with the configuration
        data_transformation = DataTransformation(config=data_transformation_config)
        
        # 4. Execute the primary function of the component (data splitting)
        data_transformation.perform_split()

# Entry point of the script
if __name__ == '__main__':
    try:
        # Log the start of the current stage
        logger.info(f"**********************")
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<<<")
        
        # Create and run the pipeline object
        obj = DataTransformationPipeline()
        obj.main()
        
        # Log the completion of the stage
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<<<\n\nX============X")

    except Exception as e:
        # Log any exceptions that occur during the process and re-raise them
        logger.exception(e)
        raise e
