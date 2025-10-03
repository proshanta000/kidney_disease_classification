import os
from KidneyCNN import logger
from KidneyCNN.pipeline.stage_01_ingestion_pipline import DataIngestionPipeline
from KidneyCNN.pipeline.stage_02_data_transformation_pipline import DataTransformationPipeline
from KidneyCNN.pipeline.stage_03_prepare_base_model import PrepareBaseModelPipeline


"""os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/proshanta000/End_to_End_ml_project_chest_CT_scan.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="proshanta000"
os.environ["MLFLOW_TRACKING_PASSWORD"]="d856c7bfcbe6c5c979320b3160b26a5a3e1f4355"
"""


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

"""
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
    raise e"""