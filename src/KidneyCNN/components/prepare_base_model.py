import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path

# Import the configuration class for PrepareBaseModel
from KidneyCNN.entity.config_entity import PrepareBaseModelConfig


# Define the class for preparing the base model for fine-tuning
class PrepareBaseModel:
    """
    Handles the preparation of a pre-trained base model for a new task.
    This includes loading the base model, adding custom layers, and freezing
    the base model's weights.
    """
    # Initialize the class with a configuration object
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    # Method to get the VGG16 base model from Keras applications
    def get_base_model(self):
        """
        Loads the VGG16 model with pre-trained weights and without the top
        classification layer. The model is saved to a file after loading.
        """
        # Load the VGG16 model with specified input shape, weights, and top exclusion
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        # Save the initial base model
        self.save_model(path=self.config.base_model_path, model=self.model)


    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepares the full model by adding a new classification head and compiling it.

        Args:
            model (tf.keras.Model): The base model to build upon.
            classes (int): The number of classes for the new classification head.
            freeze_all (bool): If True, freezes all layers of the base model.
            freeze_till (int): If not None, freezes all layers up to a specified index.
            learning_rate (float): The learning rate for the optimizer.
        """
        # Freeze layers of the base model based on the configuration
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        # Add a Flatten layer to prepare the output for the Dense layer
        flatten_in = tf.keras.layers.Flatten()(model.output)
        
        # Add the new classification layer with a 'softmax' activation for multi-class problems
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        # Create the full model by connecting the base model's input to the new output layer
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Compile the full model with an optimizer, loss function, and metrics
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        # Print a summary of the new full model's architecture
        full_model.summary()
        return full_model
    

    def update_base_model(self):
        """
        Adds the new classification head to the base model and saves the updated model.
        """
        # Call the helper method to build the full model
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        # Save the updated, compiled model
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves a Keras model to a specified file path.

        Args:
            path (Path): The file path to save the model to.
            model (tf.keras.Model): The model to be saved.
        """
        model.save(path)