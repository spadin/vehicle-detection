from feature_extraction import extract_features_from_list
from sklearn.externals import joblib
import glob
import numpy as np
import pickle

def save_model(classifier, scaler, filename="./model_data.pkl"):
    model_data = {
        "classifier": classifier,
        "scaler": scaler
    }

    joblib.dump(model_data, "model_data.pkl")

def load_model(filename="./model_data.pkl"):
    model_data = joblib.load(filename)
    return model_data["classifier"], model_data["scaler"]

def load_scaler(filename="./scaler.pkl"):
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

class Model:
    def __init__(self):
        self.model, self.scaler = load_model()

    def predict(self, features):
        features = self.scaler.transform([features])
        return self.model.predict(features)

    def score(self, features, labels):
        features = self.scaler.transform([features])
        return self.model.score(features, labels)
