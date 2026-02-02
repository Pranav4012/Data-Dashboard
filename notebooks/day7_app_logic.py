import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

# Load data
df = pd.read_csv("../data/students.csv")

# Create CareerReady column
df["CareerReady"] = (df["FinalScore"] >= 70).astype(int)

# Features and targets
X = df[["StudyHours", "Attendance", "InternalMarks", "PracticeTime", "Backlogs"]]
y_reg = df["FinalScore"]
y_clf = df["CareerReady"]

# Split data
Xr_train, Xr_test, yr_train, yr_test = train_test_split( X, y_reg, test_size=0.3, random_state=42)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_clf, test_size=0.3, random_state=42)

# Train models
reg_model = LinearRegression()
reg_model.fit(Xr_train, yr_train)

clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(Xc_train, yc_train)

# Prediction Engine (for app)
def predict_student(study_hours, attendance, internal_marks, practice_time, backlogs):
    # Convert single input into DataFrame with same structure as X
    input_df = pd.DataFrame([{
        "StudyHours": study_hours,
        "Attendance": attendance,
        "InternalMarks": internal_marks,
        "PracticeTime": practice_time,
        "Backlogs": backlogs
    }])

    # Predict
    predicted_score = reg_model.predict(input_df)[0]
    career_ready = clf_model.predict(input_df)[0]

    status = "Career Ready" if career_ready == 1 else "Needs Improvement"

    return predicted_score, status


# Test the function
score, status = predict_student(4, 80, 70, 3, 1)
print("Predicted Score:", round(score, 2))
print("Career Status:", status)


