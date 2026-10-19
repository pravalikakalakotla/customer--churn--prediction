
import pandas as pd
import os

# Load the original dataset
df = pd.read_csv("data/customer_churn.csv")

# Convert TotalCharges from text to numbers
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Inspect rows with missing TotalCharges
missing_rows = df[df["TotalCharges"].isnull()]

print("Rows with missing TotalCharges:")
print(missing_rows[["customerID", "tenure", "TotalCharges"]])

# Check whether missing values belong to new customers
print("\nTenure values of missing rows:")
print(missing_rows["tenure"].value_counts())

# Replace missing charges with 0 only for zero-tenure customers
mask = df["TotalCharges"].isnull() & (df["tenure"] == 0)
df.loc[mask, "TotalCharges"] = 0

# Check if any missing values remain
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Save the cleaned dataset
os.makedirs("data", exist_ok=True)
df.to_csv("data/customer_churn_clean.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("Dataset shape:", df.shape)
