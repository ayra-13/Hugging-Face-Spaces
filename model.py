import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.model_selection import train_test_split

# Function to load and preprocess data from a single sheet
def load_sheet(file_path, sheet_name):
    data = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=2)  # Skip header rows
    return data

# Load data from all sheets
ict_sheets = ['ICT Morning', 'ICT Afternoon']
cc_sheets = ['cloud computing morning', 'cloud computing afternoon']

ict_data = pd.concat([load_sheet("ICT_Result.xlsx", sheet) for sheet in ict_sheets], ignore_index=True)
cc_data = pd.concat([load_sheet("CC_Result.xlsx", sheet) for sheet in cc_sheets], ignore_index=True)

# Combine ICT and CC data
combined_data = pd.concat([ict_data, cc_data], ignore_index=True)

# Clean column names
combined_data.columns = combined_data.columns.astype(str).str.strip()

# Print cleaned column names for debugging
print("Cleaned Column Names:")
print(combined_data.columns.tolist())

# Define feature and target columns
feature_columns = ['30', '49', '100', '30.1', '15', '35', '45', '100.1', '32', '24', '40']
target_column = '40.1'  # Final score

# Check if columns exist in the dataset
missing_columns = [col for col in feature_columns if col not in combined_data.columns]
if missing_columns:
    print(f"Missing columns in dataset: {missing_columns}")
    exit()

# Select features and target
X = combined_data[feature_columns]
y = combined_data[target_column]

# Handle any missing values
X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save the model
joblib.dump(model, 'student_score_predictor.pkl')
print("Model saved as 'student_score_predictor.pkl'")

# Example: Predict the final score for a new student
new_data = [[24, 34, 100, 29, 10, 32.38, 36, 100, 21, 9, 32]]  # Replace with actual input
predicted_final = model.predict(new_data)
print(f"Predicted Final Score: {predicted_final[0]}")
