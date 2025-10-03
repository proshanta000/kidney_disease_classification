# Import necessary modules
from KidneyCNN.config.configuration import ConfigurationManager
from KidneyCNN.components.data_transformation import DataTransformation
from KidneyCNN import logger


# Define a constant for the stage name, used in logging
STAGE_NAME = "Data Transformation model"

# Define the main pipeline class for this stage
class DataTransformationPipeline:
    def __init__(self):
        # Constructor for the pipeline class.
        pass

    def main(self):
        """
        Executes the main logic for the 'Prepare base model' pipeline stage.
        
        This method gets the configuration, creates the component, and runs its
        primary functions to prepare the base model for training.
        """
        # Create an instance of the ConfigurationManager
        config = ConfigurationManager()
        
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
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