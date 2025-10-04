# Import necessary modules
from KidneyCNN.config.configuration import ConfigurationManager # Component to manage and load configuration settings
from KidneyCNN.components.Model_evaluation import Evalution # The core component responsible for model evaluation
from KidneyCNN import logger # Custom logger for tracking pipeline progress and status

# Define a constant for the stage name, used for logging
STAGE_NAME = "Model Evaluation"

# Define the main pipeline class for the model evaluation stage
class ModelEvalutationPipeline:
    """
    Pipeline class dedicated to the Model Evaluation stage.
    It orchestrates loading the trained model, evaluating its performance, 
    and logging the results to MLflow.
    """
    def __init__(self):
        # The constructor can be used for initial setup, though it is empty here.
        pass

    def main(self):
        """
        Executes the main logic for the Model Evaluation pipeline stage.
        """
        
        # 1. Create an instance of the ConfigurationManager to get configurations
        config = ConfigurationManager()
        
        # 2. Retrieve the specific evaluation configuration
        eval_config = config.get_evalution_config()
        
        # 3. Initialize the Evaluation component with the configuration
        evaluation = Evalution(eval_config)
        
        # 4. Execute the evaluation process (loads model, runs prediction, saves scores)
        evaluation.evaluation()
        
        # 5. Log the results (parameters and metrics) and the model itself to MLflow
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
