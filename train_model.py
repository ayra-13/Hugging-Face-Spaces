# train_model.py
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

# Generate dataset: X is numbers, y is 1 if even, 0 if odd
X = np.arange(1, 101).reshape(-1, 1)  # Numbers 1 to 100
y = (X % 2 == 0).astype(int).ravel()  # 1 for even, 0 for odd

# Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Save the model to a file
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model/model.pkl")
