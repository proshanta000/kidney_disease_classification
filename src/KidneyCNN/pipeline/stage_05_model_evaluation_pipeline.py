# Import necessary modules
from KidneyCNN.config.configuration import ConfigurationManager
from KidneyCNN.components.Model_evaluation import Evalution
from KidneyCNN import logger

# Define a constant for the stage name, used for logging
STAGE_NAME = "Model Evaluation"

# Define the main pipeline class for the model training stage
class ModelEvalutationPipeline:
    def __init__(self):
        # The constructor can be used for initial setup, though it is empty here.
        pass

    def main(self):
        
        # Create an instance of the ConfigurationManager to get configurations
        config = ConfigurationManager()
        eval_config = config.get_evalution_config()
        evaluation= Evalution(eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()
    

# Entry point of the script
if __name__ == '__main__':
    try:
        # Log the start of the current stage
        logger.info(f"**********************")
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<<<")
        
        # Create an instance of the pipeline and run its main method
        obj = ModelEvalutationPipeline()
        obj.main()
        
        # Log the successful completion of the stage
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<<<\n\nX============X")

    except Exception as e:
        # If any exception occurs, log it with a traceback and re-raise it
        logger.exception(e)
        raise e