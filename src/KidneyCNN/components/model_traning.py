import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from KidneyCNN.entity.config_entity import TrainingConfig


class Training:
    """
    Component class responsible for the model training process, including 
    loading the base model, compiling it, setting up data generators, and 
    executing the fit loop.
    """
    def __init__(self, config: TrainingConfig):
        # Initializes the component with the TrainingConfig object, which holds 
        # paths, hyperparameters (epochs, learning rate), and callbacks.
        self.config = config
        # self.model and self.train_generator/self.valid_generator will be set later.

    def get_base_model(self):
        """
        Loads the pre-prepared VGG16 model (with the classification head attached).
        """
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    # This method is for compiling the model with an optimizer and loss function.
    def compile_model(self):
        """
        Configures the model for training using the optimizer, loss function, and metrics 
        specified in the configuration (params.yaml).
        """
        # The .compile() method configures the model for training.
        self.model.compile(
            # The optimizer updates the model based on the data and the loss function.
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
            # The loss function measures how accurate the model is.
            # CategoricalCrossentropy is suitable for multi-class classification problems.
            loss = tf.keras.losses.CategoricalCrossentropy(),
            # The metrics list is used to monitor the training and testing steps.
            metrics=["accuracy"]
        )

    def train_valid_generator(self):
        """
        Prepares and configures two separate data generators: one for training and 
        one for validation. These generators load images from disk efficiently.
        """
        # Arguments for the ImageDataGenerator instance.
        datagenerator_kwargs = dict(
            # Normalizes pixel values from [0, 255] to [0, 1].
            rescale = 1./255,
            # Reserves 20% of the data found in the 'directory' for validation 
            # (though this may be redundant if separate directories are used).
            validation_split=0.20 
        )

        # Arguments for the .flow_from_directory() method.
        dataflow_kwargs = dict(
            # Resizes images to the required input size for the VGG16 model (e.g., 224x224).
            target_size = self.config.params_image_size[:-1],
            # Defines the number of samples per batch.
            batch_size = self.config.params_batch_size,
            # Defines the method for resizing images.
            interpolation = "bilinear"
        )

        # Create the ImageDataGenerator instance.
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Verify that the required data directories exist before proceeding to load data.
        if not os.path.exists(self.config.training_data):
            raise FileNotFoundError(f"Training data directory not found at: {self.config.training_data}")
        
        if not os.path.exists(self.config.validation_data):
            raise FileNotFoundError(f"Validation data directory not found at: {self.config.validation_data}")


        # Training Data Generator
        self.train_generator = valid_datagenerator.flow_from_directory(
            # Path to the training data split directory.
            directory = self.config.training_data, 
            # Uses the training subset defined by validation_split (if applicable).
            subset="training",
            # Shuffles the data order for better training convergence.
            shuffle=True,
            **dataflow_kwargs
        )

        # Validation Data Generator
        self.valid_generator = valid_datagenerator.flow_from_directory(
            # Path to the validation data split directory.
            directory=self.config.validation_data,
            # Uses the validation subset defined by validation_split.
            subset="validation", 
            # Data order does not need to be shuffled for validation.
            shuffle=False,
            **dataflow_kwargs
        )

        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the trained Keras model to the specified path.
        """
        model.save(path)


    def train(self):
        """
        The main training loop, orchestrating model loading, compilation, 
        data preparation, fitting, and final saving.
        """
        # Load the model with the updated head.
        self.get_base_model()
        # Configure the optimizer and loss function.
        self.compile_model()
        # Set up the data loading pipelines.
        self.train_valid_generator()
        
        # Execute the model training loop.
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            # Pass the list of callbacks (e.g., EarlyStopping, ModelCheckpoint)
            callbacks=list(self.config.params_callbacks.values())
        )
        # Save the fully trained model to the designated path.
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
