import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from KidneyCNN.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config


    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    # This method is for compiling the model with an optimizer and loss function.
    def compile_model(self):
        # The .compile() method configures the model for training.
        self.model.compile(
            # The optimizer updates the model based on the data and the loss function.
            # Here, we use Stochastic Gradient Descent (SGD) with the learning rate from the config.
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
            # The loss function measures how accurate the model is.
            # CategoricalCrossentropy is suitable for multi-class classification problems.
            loss = tf.keras.losses.CategoricalCrossentropy(),
            # The metrics list is used to monitor the training and testing steps.
            # "accuracy" is a common metric to track model performance.
            metrics=["accuracy"]
        )

    def train_valid_generator(self):
        # This method prepares and configures data generators for training and validation.
        # Data generators load images from disk and apply transformations on the fly,
        # which is memory-efficient for large datasets.

        # Arguments for the ImageDataGenerator.
        # `rescale` normalizes pixel values to the range [0, 1].
        # `validation_split` reserves a portion of the data for validation.
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        # Arguments for the .flow_from_directory() method.
        # `target_size` resizes all images to the specified dimensions.
        # `batch_size` is the number of images to be yielded from the generator.
        # `interpolation` defines the method for resizing images.
        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            interpolation = "bilinear"
        )

        # Create an ImageDataGenerator instance with the defined arguments.
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Verify that the training and validation data directories exist before proceeding.
        if not os.path.exists(self.config.training_data):
            raise FileNotFoundError(f"Training data directory not found at: {self.config.training_data}")
        
        if not os.path.exists(self.config.validation_data):
            raise FileNotFoundError(f"Validation data directory not found at: {self.config.validation_data}")


        # The training data is assumed to be in the 'train' folder
        # The validation data is assumed to be in the 'valid' folder
        self.train_generator = valid_datagenerator.flow_from_directory(
            directory = self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        # Correctly pointing to the separate validation data directory
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.validation_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self):
        # Load the base model first
        self.get_base_model()
        # Compile the model with an optimizer, loss, and metrics
        self.compile_model()
        # Prepare the data generators
        self.train_valid_generator()
        
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            callbacks=list(self.config.params_callbacks.values())
        )
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )