import pandas as pd

# Load dataset
df = pd.read_csv("../data/students.csv")

# Basic inspection
print("Shape of data:", df.shape)
print("\nFirst 5 rows:\n", df.head())

print("\nColumn Info:")
print(df.info())

print("\nStatistical Summary:\n", df.describe())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())
