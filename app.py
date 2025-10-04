import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from KidneyCNN.utils.common import decodeImage
from KidneyCNN.pipeline.Prediction_pipline import PredictionPipeline


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)



@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/tran", methods=["GET", "POST"])
@cross_origin
def traningRoute():
    os.system("python main.py")
    #os.system("dvc repo")
    return"Traning done successfully!"

# CORRECTED: Added the leading slash to the URL rule
@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    image =request.json['image']
    decodeImage(image, c1APP.filename)
    result =c1APP.classifier.predict()
    # Corrected: Now returns the first item from the list, which is the dictionary
    return jsonify(result[0])


if __name__ == "__main__":
    c1APP = ClientApp()
    app.run(host='0.0.0.0', port=8080)