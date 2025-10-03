# Import necessary modules
from KidneyCNN.config.configuration import ConfigurationManager
from KidneyCNN.components.data_ingestion import DataIngestion
from KidneyCNN.components.prepare_base_model import PrepareBaseModel
from KidneyCNN import logger


# Define a constant for the stage name, used in logging
STAGE_NAME = "Prepare base model"

# Define the main pipeline class for this stage
class PrepareBaseModelPipeline:
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
        
        # Get the specific configuration for the 'PrepareBaseModel' component
        prepare_base_model_config = config.get_prepear_base_model_config()
        
        # Instantiate the PrepareBaseModel component with its configuration
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        
        # Call the method to load the pre-trained base model (e.g., VGG16)
        prepare_base_model.get_base_model()
        
        # Call the method to add custom layers and freeze the base model
        prepare_base_model.update_base_model()

# Entry point of the script
if __name__ == '__main__':
    try:
        # Log the start of the current stage
        logger.info(f"**********************")
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<<<")
        
        # Create and run the pipeline object
        obj = PrepareBaseModelPipeline()
        obj.main()
        
        # Log the completion of the stage
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<<<\n\nX============X")

    except Exception as e:
        # Log any exceptions that occur during the process and re-raise them
        logger.exception(e)
        raise e