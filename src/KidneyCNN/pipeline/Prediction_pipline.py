import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        # Load the model only once when the class is initialized for efficiency
        self.model = load_model(os.path.join("artifacts", "training", "model.h5"))
        # Define the class names based on the alphabetical order of the folders
        # This mapping should ideally be saved from the training pipeline for robustness
        self.class_names = ['cyst', 'normal', 'stone', 'tumor']

    def predict(self):
        imagename = self.filename
        try:
            test_image = image.load_img(imagename, target_size=(224, 224))
        except FileNotFoundError:
            return [{"image": "Error: Image file not found."}]
        except Exception as e:
            return [{"image": f"Error loading image: {e}"}]

        test_image = image.img_to_array(test_image)
        # Normalize the image pixels to the range [0, 1] just like in the training
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        
        # Make the prediction
        predictions = self.model.predict(test_image)
        result_index = np.argmax(predictions, axis=1)[0]
        
        # Get the prediction using the class_names list
        prediction = self.class_names[result_index]
        
        print(f"Prediction result index: {result_index}")
        print(f"Predicted class: {prediction}")

        return [{"image": prediction}]