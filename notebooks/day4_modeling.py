import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# we are going to train the ML 

df = pd.read_csv("../data/students.csv")
df["CareerReady"] = (df["FinalScore"] >= 70).astype(int)

X = df[["StudyHours", "Attendance", "InternalMarks", "PracticeTime", "Backlogs"]]
y_reg = df["FinalScore"]   
y_clf = df["CareerReady"] 
# split data into both traning and testing 
# test_size=0.3 means 30% of the data goes to testing, 70% to training.
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.3, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Initialize and train (fit) the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test features (X_test)
predictions = model.predict(X_test)

# Evaluate the model by comparing predictions to the actual test labels (y_test)
mse = mean_squared_error(y_test, predictions)
print(f"\nMean Squared Error on Test Data: {mse}")

# from sklearn.model_selection import train_test_split : = this splits your data for traning and testing purpose 
# from sklearn.linear_model import LinearRegression : = means it finds a relationship of a staright line y = mx + c
# from sklearn.linear_model import LinearRegression : used for predicting continous values
# from sklearn.metrics import mean_squared_error : measure the how wrong perdiction are , it calculate the mean square error 
# lower mse = means better model .(mse = mean square error)

# for classification 
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_clf, test_size=0.3, random_state=42)
# here this is for train classification model
clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(Xc_train, yc_train)

# Predict on test data (NOT training data)
clf_predictions = clf_model.predict(Xc_test)

# Evaluate
accuracy = accuracy_score(yc_test, clf_predictions)
print("Classification Accuracy:", accuracy)









