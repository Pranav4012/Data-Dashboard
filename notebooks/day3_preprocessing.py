import pandas as pd

df = pd.read_csv("../data/students.csv")
df["CareerReady"] = (df["FinalScore"] >= 70).astype(int)

# Features for ML
X = df[["StudyHours", "Attendance", "InternalMarks", "PracticeTime", "Backlogs"]]

# Targets
y_reg = df["FinalScore"]      # Regression target
y_clf = df["CareerReady"]     # Classification target

# Outputs for verification
print("Data with CareerReady:\n", df.head())
print("\nShape of X:", X.shape)
print("\nCareerReady distribution:\n", df["CareerReady"].value_counts())