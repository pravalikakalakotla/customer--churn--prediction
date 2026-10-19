
import pandas as pd

# Load the dataset
df = pd.read_csv("data/customer_churn.csv")

# Check for duplicate customer records
print("Duplicate rows:", df.duplicated().sum())

# Convert TotalCharges to numeric
# Invalid or blank values become NaN
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Check missing values in each column
print("\nMissing values:")
print(df.isnull().sum())

# Check the data types after conversion
print("\nUpdated data types:")
print(df.dtypes)

# Display basic statistics
print("\nNumerical summary:")
print(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe())


import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a folder for saving graphs
os.makedirs("outputs", exist_ok=True)

sns.set_theme(style="whitegrid")

# Create four graphs
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Graph 1: Customers who left vs stayed
sns.countplot(data=df, x="Churn", ax=axes[0, 0])
axes[0, 0].set_title("Customer Churn Distribution")

# Graph 2: Churn by contract type
sns.countplot(
    data=df, x="Contract", hue="Churn", ax=axes[0, 1]
)
axes[0, 1].set_title("Churn by Contract Type")
axes[0, 1].tick_params(axis="x", rotation=15)

# Graph 3: Customer tenure vs churn
sns.boxplot(
    data=df, x="Churn", y="tenure", ax=axes[1, 0]
)
axes[1, 0].set_title("Tenure vs Churn")

# Graph 4: Monthly charges vs churn
sns.boxplot(
    data=df, x="Churn", y="MonthlyCharges",
    ax=axes[1, 1]
)
axes[1, 1].set_title("Monthly Charges vs Churn")

plt.tight_layout()

# Save the graphs for your project report
plt.savefig("outputs/churn_analysis.png", dpi=300)

# Display the graphs
plt.show()
