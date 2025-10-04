import os
from KidneyCNN import logger
from KidneyCNN.pipeline.stage_01_ingestion_pipline import DataIngestionPipeline
from KidneyCNN.pipeline.stage_02_data_transformation_pipline import DataTransformationPipeline
from KidneyCNN.pipeline.stage_03_prepare_base_model import PrepareBaseModelPipeline
from KidneyCNN.pipeline.stage_04_traning_pipline import ModelTraningPipeline
from KidneyCNN.pipeline.stage_05_model_evaluation_pipeline import ModelEvalutationPipeline


os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/proshanta000/kidney_disease_classification.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="proshanta000"
os.environ["MLFLOW_TRACKING_PASSWORD"]="1d2afccec4c7566b7e8d9ed3f00a41e1e86ae8fd"



STAGE_NAME = "Data Ingestion Stage"


if __name__=='__main__':
    try:
        logger.info(f"**********************")
        logger.info(f">>>>>>>>>>  stage {STAGE_NAME} started <<<<<<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<\n\nX============X")
    except Exception as e:
        logger.exception(e)
        raise e
    

STAGE_NAME = "Data Transformation Stage"


if __name__=='__main__':
    try:
        logger.info(f"**********************")
        logger.info(f">>>>>>>>>>  stage {STAGE_NAME} started <<<<<<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<\n\nX============X")
    except Exception as e:
        logger.exception(e)
        raise e
    


STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f"**********************")
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<<<")
    obj= PrepareBaseModelPipeline()
    obj.main()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<<<\n\nX============X")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Training"

try:
    logger.info(f"**********************")
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<<<")
    obj= ModelTraningPipeline()
    obj.main()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<<<\n\nX============X")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Evaluation"

try:
    logger.info(f"**********************")
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<<<")
    obj= ModelEvalutationPipeline()
    obj.main()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<<<\n\nX============X")

except Exception as e:
    logger.exception(e)
    raise e