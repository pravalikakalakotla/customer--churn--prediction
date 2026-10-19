
import pandas as pd

# Load the customer dataset
df = pd.read_csv("data/customer_churn.csv")

# Display the first 5 rows
print("First 5 customers:")
print(df.head())

# Display the number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Display all column names
print("\nColumn names:")
print(df.columns.tolist())

# Display information about the dataset
print("\nDataset information:")
df.info()

# Count customers who left and stayed
print("\nChurn distribution:")
print(df["Churn"].value_counts())
