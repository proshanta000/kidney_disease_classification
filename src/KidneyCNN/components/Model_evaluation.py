import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import json
from KidneyCNN.utils.common import save_json
from KidneyCNN.entity.config_entity import EvaluationConfig

class Evalution:
    """
    Handles the final evaluation of the trained model, including setting up the 
    validation data flow, calculating metrics (loss, accuracy), saving the results, 
    and logging the entire run to MLflow.
    """
    def __init__(self, config: EvaluationConfig):
        # Initializes the component with the evaluation configuration 
        # (paths, MLflow URI, batch size, image size, etc.).
        self.config = config

    def _valid_generator(self):
        """
        Creates and configures an image data generator for the validation/test dataset.
        """
        # Dictionary of parameters for the ImageDataGenerator (preprocessing).
        datagenerator_kwargs = dict(
            # Normalize pixel values to the range [0, 1].
            rescale=1. / 255
        )
        
        # Dictionary of parameters for flow_from_directory (data loading behavior).
        dataflow_kwargs = dict(
            # Target size for all images (excluding the channel dimension).
            target_size=self.config.params_image_size[:-1],
            # Number of samples per batch.
            batch_size=self.config.params_batch_size,
            # Interpolation method for resizing images.
            interpolation="bilinear"
        )
        
        # Instantiate the Keras ImageDataGenerator.
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        # Create the data flow iterator from the specified directory.
        self.valid_generator = valid_datagenerator.flow_from_directory(
            # Path to the validation/test data directory.
            directory=self.config.validation_data,
            # Data should not be shuffled during evaluation.
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Loads a pre-trained Keras model from the given path.
        """
        return tf.keras.models.load_model(path)

    def evaluation(self):
        """
        Performs the model evaluation: loads the model, prepares the generator, 
        and calculates the score.
        """
        # Load the trained model.
        self.model = self.load_model(self.config.path_of_model)
        # Setup the data generator for the validation/test set.
        self._valid_generator()
        # Evaluate the model on the data generator and store the results (loss and accuracy).
        self.score = self.model.evaluate(self.valid_generator)
        # Save the resulting metrics to a JSON file.
        self.save_score()

    def save_score(self):
        """
        Saves the evaluation metrics (loss and accuracy) to a 'scores.json' file.
        """
        # Format the score into a dictionary.
        score = {"loss": self.score[0], "accuracy": self.score[1]}
        # Use the common utility function to save the dictionary as JSON.
        save_json(path=Path("scores.json"), data=score)

    def log_into_mlflow(self):
        """
        Configures and logs the experiment parameters, metrics, and the model to MLflow.
        """
        # Set the MLflow tracking server URI.
        mlflow.set_registry_uri(self.config.mlflow_uri)
        # Check the type of the tracking store to determine model registration capability.
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Log all parameters specified in the configuration (e.g., learning rate, epochs).
            mlflow.log_params(self.config.all_params)

            # Log each metric individually (loss).
            mlflow.log_metric(
                key="loss",
                value=self.score[0]
            )
            # Log each metric individually (accuracy).
            mlflow.log_metric(
                key="accuracy",
                value=self.score[1]
            )

            # Model registration logic: Only register model if the tracking store 
            # is not a local file system.
            if tracking_url_type_store != "file":
                # Log the Keras model and register it under a specific name.
                mlflow.keras.log_model(self.model, "model",
                                       registered_model_name="VGG16Model")
            else:
                # Log the model without registering if using a local file store.
                mlflow.keras.log_model(self.model, "model")
