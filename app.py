import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("log_reg_classifier.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    
    if prediction == 0:
        prediction_text = "The patient is unlikely to discontinue the ART"
    else:
        prediction_text = "The patient is likely to discontinue the ART"
    
    return render_template("index.html", prediction_text=prediction_text)


if __name__ == "__main__":
    flask_app.run(debug=True)