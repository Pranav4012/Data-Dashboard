import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../data/students.csv")

X = df[["StudyHours", "Attendance", "InternalMarks", "PracticeTime", "Backlogs"]]
y_reg = df["FinalScore"]  

X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set features
predictions = model.predict(X_test)

print(f"Predictions: {predictions}")
print(f"Actual values (y_test): {y_test}")
actual = y_test.values

predictions = model.predict(X_test)

actual = y_test.values
predicted = predictions

error = actual - predicted

result_df = pd.DataFrame({
    "Actual_FinalScore": actual,
    "Predicted_FinalScore": predicted,
    "Error": error
})

print(result_df)