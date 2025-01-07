# model.py
import pickle

# Load the trained model
def load_model():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Predict whether a number is even or odd
def predict(number):
    model = load_model()
    prediction = model.predict([[number]])[0]
    return "Even" if prediction == 1 else "Odd"
