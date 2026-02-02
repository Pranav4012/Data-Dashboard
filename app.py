import pandas as pd 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st

# Load data
df = pd.read_csv("data/students.csv")

df["CareerReady"] = (df["FinalScore"] >= 70).astype(int)


# Features and targets
X = df[["StudyHours", "Attendance", "InternalMarks", "PracticeTime", "Backlogs"]]
y_reg = df["FinalScore"]
y_clf = df["CareerReady"]

# Split data
Xr_train, Xr_test, yr_train, yr_test = train_test_split( X, y_reg, test_size=0.3, random_state=42)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_clf, test_size=0.3, random_state=42)
# train model 
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
    predicted_score = reg_model.predict(input_df)[0]
    career_ready = clf_model.predict(input_df)[0]

    status = "Career Ready" if career_ready == 1 else "Needs Improvement"
    return predicted_score, status

st.title("Welcome To Dashboard !")
st.header("Student Information Input")
st.write("The student dashboard provides a quick overview of courses, assignments, progress, and important announcements in one place, helping students stay organized and on track.")

study_hours = st.number_input("Enter study hour :",min_value=1, max_value=100, value=25)
st.write(f"The study hour is {study_hours}")

Attendance = st.number_input("Enter the attandence", min_value=0, max_value=100, value=75)
st.write(f"The attendance :{Attendance}")

InternalMarks = st.number_input("Enter the internal marks :", min_value=0, max_value=100, value=60)
st.write(f"Internal marks is : {InternalMarks}")

PracticeTime = st.number_input("How much practice time of the student : ", min_value=0, max_value=10, value=2)
st.write(f"The practice time : {PracticeTime}")

Backlogs = st.number_input("Student with how many backlogs :", min_value=0, max_value=10, value=0)
st.write(f"The backlogs :{Backlogs}")

if st.button("Predict"):
    score, status = predict_student(
        study_hours,
        Attendance,
        InternalMarks,
        PracticeTime,
        Backlogs
    )

    st.success(f"Predicted Final Score: {round(score, 2)}")
    st.info(f"Career Status: {status}")

st.header("Data Overview")
col1 , col2 = st.columns(2)
col1.metric("Rows", df.shape[0])
col2.metric("Columns",df.shape[1])

#data preview 
st.subheader("Preview")
st.dataframe(df.head())
#As attendance increases, how scores behave , Clean, readable, zero extra code
st.subheader("Attendance vs Final Score")
st.line_chart(
    df.sort_values("Attendance")[["Attendance", "FinalScore"]].set_index("Attendance"))
#More practice generally leads to better performance
st.subheader("Practice Time vs Final Score")
st.line_chart(
    df.sort_values("PracticeTime")[["PracticeTime", "FinalScore"]].set_index("PracticeTime"))

st.markdown("---")
st.header("📂 Batch Prediction (Upload CSV)")
st.write("Upload a CSV file with student details to predict performance for multiple students.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

st.markdown("---")
st.header("📄 Download Sample CSV Template")

sample_df = pd.DataFrame({
    "StudyHours": [5],
    "Attendance": [75],
    "InternalMarks": [60],
    "PracticeTime": [2],
    "Backlogs": [0]
})

csv_template = sample_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Sample CSV Template",
    data=csv_template,
    file_name="sample_student_input.csv",
    mime="text/csv"
)


if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(batch_df.head())

    required_columns = [
        "StudyHours",
        "Attendance",
        "InternalMarks",
        "PracticeTime",
        "Backlogs"
    ]

    if not all(col in batch_df.columns for col in required_columns):
            st.error("❌ Invalid file format! Your CSV must contain exactly these columns:")
            for col in required_columns:
                st.write(f"• {col}")
    else:
        st.success("All required columns are present.")

        batch_X = batch_df[required_columns]

        batch_df["Predicted_FinalScore"] = reg_model.predict(batch_X)

        batch_df["Career_Status"] = clf_model.predict(batch_X)
        batch_df["Career_Status"] = batch_df["Career_Status"].map(
            {1: "Career Ready", 0: "Needs Improvement"}
        )

        st.subheader("Batch Prediction Results")
        st.dataframe(batch_df)

        csv = batch_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv,
            file_name="student_predictions.csv",
            mime="text/csv"
        )
st.markdown("---")
st.caption("Built by Pranav Choudhary | Python • Machine Learning • Streamlit")









