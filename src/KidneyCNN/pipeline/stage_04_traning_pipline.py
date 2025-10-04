# Import necessary modules
from KidneyCNN.config.configuration import ConfigurationManager
from KidneyCNN.components.traning import Training
from KidneyCNN import logger

# Define a constant for the stage name, used for logging
STAGE_NAME = "Training"




# Define the main pipeline class for the model training stage
class ModelTraningPipeline:
    def __init__(self):
        # The constructor can be used for initial setup, though it is empty here.
        pass

    def main(self):
        """
        Orchestrates the model training process.
        
        This method executes the sequential steps of model training:
        1.  Getting the training configuration.
        2.  Loading the pre-trained base model.
        3.  Compiling the model for training.
        4.  Setting up data generators for training and validation data.
        5.  Initiating the training process.
        """
        # Create an instance of the ConfigurationManager to get configurations
        config = ConfigurationManager()
        
        # Get the specific configuration for the 'Training' component
        training_config = config.get_training_config()
        
        # Instantiate the Training component with the fetched configuration
        training = Training(config=training_config)
        
        # 1. Load the model's architecture and weights. This model has been
        #    prepared in a previous stage (PrepareBaseModel).
        training.get_base_model() 
        
        # 2. Re-compile the model. This is the crucial step because the optimizer's
        #    state is not saved with the model architecture. Compiling here
        #    ensures the model is ready for training.
        training.compile_model() 
        
        # 3. Set up the data generators. This handles data loading, preprocessing,
        #    and augmentation for both the training and validation sets.
        training.train_valid_generator()
        
        # 4. Start the training process. The model will now be trained on the
        #    prepared data using the specified epochs, batch size, and callbacks.
        training.train()

# Entry point of the script
if __name__ == '__main__':
    try:
        # Log the start of the current stage
        logger.info(f"**********************")
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<<<")
        
        # Create an instance of the pipeline and run its main method
        obj = ModelTraningPipeline()
        obj.main()
        
        # Log the successful completion of the stage
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<<<\n\nX============X")

    except Exception as e:
        # If any exception occurs, log it with a traceback and re-raise it
        logger.exception(e)
        raise e