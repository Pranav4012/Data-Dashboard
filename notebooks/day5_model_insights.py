import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# have import the csv file 
df = pd.read_csv("../data/students.csv")

X = df[["StudyHours", "Attendance", "InternalMarks", "PracticeTime", "Backlogs"]]
y_reg = df["FinalScore"]  
# applying the logic for train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.3, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
# linear regression 
model = LinearRegression()
model.fit(X_train,y_train)

# we have to print model cofficient (by this we can get the coefficient  )
coefficient = model.coef_
print("cofficients : ", coefficient)

for feature, coef in zip(X.columns,coefficient):
    print(f"{feature} -> {coef:.2f}")



