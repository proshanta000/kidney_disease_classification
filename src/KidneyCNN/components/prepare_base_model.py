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
    Handles the preparation of a pre-trained base model for a new task (kidney classification).
    This includes loading the base model (VGG16), adding a custom classification head, 
    and freezing the weights of the base layers to prevent corruption during initial training.
    """
    # Initialize the class with a configuration object
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    # Method to get the VGG16 base model from Keras applications
    def get_base_model(self):
        """
        Loads the VGG16 model with pre-trained ImageNet weights and without the top
        classification layer (`include_top=False`), making it suitable as a feature extractor.
        The initial base model is saved to disk.
        """
        # Load the VGG16 model with specified parameters.
        self.model = tf.keras.applications.vgg16.VGG16(
            # Input shape must match the data generator output (e.g., 224x224x3).
            input_shape=self.config.params_image_size,
            # Use weights pre-trained on the ImageNet dataset.
            weights=self.config.params_weights,
            # Exclude the default 1000-class classification head.
            include_top=self.config.params_include_top 
        )

        # Save the initial base model to disk before modifications.
        self.save_model(path=self.config.base_model_path, model=self.model)


    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepares the full transfer learning model by adding a new classification head 
        (Flatten and Dense layers) and compiling it with an optimizer.

        Args:
            model (tf.keras.Model): The base model (VGG16 without the top) to build upon.
            classes (int): The number of classes for the new classification head (e.g., 4).
            freeze_all (bool): If True, freezes all layers of the base model.
            freeze_till (int): If not None, freezes all layers up to a specified index 
                                (for partial unfreezing/fine-tuning later).
            learning_rate (float): The learning rate for the optimizer.
        """
        # --- 1. Freezing Logic ---
        # Freeze layers of the base model based on the configuration to prevent updating 
        # pre-trained weights during initial training.
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        # The 'freeze_till' logic allows unfreezing a specific number of layers from the end.
        elif (freeze_till is not None) and (freeze_till > 0):
            # Freeze all layers EXCEPT the last 'freeze_till' layers.
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # --- 2. Classification Head ---
        # Add a Flatten layer to convert the 3D output of the VGG16 base into 1D feature vector.
        flatten_in = tf.keras.layers.Flatten()(model.output)
        
        # Add the new classification layer (Dense layer).
        prediction = tf.keras.layers.Dense(
            units=classes,
            # Use 'softmax' activation for multi-class classification to get probability distribution.
            activation="softmax" 
        )(flatten_in)

        # Create the full model by connecting the base model's input to the new output layer
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # --- 3. Model Compilation ---
        # Compile the full model with an optimizer, loss function, and metrics
        full_model.compile(
            # Using Stochastic Gradient Descent (SGD) with a defined learning rate.
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            # Categorical Crossentropy loss for multi-class classification.
            loss=tf.keras.losses.CategoricalCrossentropy(),
            # Track accuracy during training.
            metrics=["accuracy"]
        )

        # Print a summary of the new full model's architecture to check the added layers and trainable parameters.
        full_model.summary()
        return full_model
    

    def update_base_model(self):
        """
        Coordinates the preparation of the full model by calling the static helper 
        method and passing the required configuration parameters. 
        It then saves the final, compiled model to the updated path.
        """
        # Call the helper method to build the full model
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            # Use configuration value for freezing layers
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        # Save the updated, compiled model, ready for training
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
