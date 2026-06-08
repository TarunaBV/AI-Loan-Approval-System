import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset/train.csv")

# Fill missing values
for column in df.columns:

    if pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].median())

    else:
        df[column] = df[column].fillna(df[column].mode()[0])

# Drop Loan_ID
df.drop("Loan_ID", axis=1, inplace=True)

# Encode categorical columns
encoders = {}

for column in df.columns:
   if not pd.api.types.is_numeric_dtype(df[column]):
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        encoders[column] = le

# Features and Target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, "model/loan_model.pkl")

# Save encoders
joblib.dump(encoders, "model/encoders.pkl")

print("Model Saved Successfully")