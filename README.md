# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.Load the dataset, drop unnecessary columns, and encode categorical variables.
 2.Define the features (X) and target variable (y).
 3.Split the data into training and testing sets.
 4 .Train the logistic regression model, make predictions, and evaluate using accuracy and other

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: AJAYPRABU A
RegisterNumber:  212225220005

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
```
## Output:
HEAD
<img width="1039" height="271" alt="image" src="https://github.com/user-attachments/assets/d88cdb49-1cd1-4343-b573-2737dddbe21b" />
COPY
<img width="1044" height="310" alt="image" src="https://github.com/user-attachments/assets/01f124e1-20ae-4f3c-881a-ea6a8a3d5647" />
FIT TRASFORM
<img width="1037" height="650" alt="image" src="https://github.com/user-attachments/assets/0d5605a3-f84f-4086-8c73-1197f0202acc" />
LOGISTIC REGRESSION
<img width="1031" height="246" alt="image" src="https://github.com/user-attachments/assets/355f1e86-a95e-4c44-adbe-7a8f5ea02a20" />
ACCURACY SCORE
<img width="996" height="140" alt="image" src="https://github.com/user-attachments/assets/e4c96f8a-2617-416c-b4e2-5515ad10d5fa" />
CONFUSION MATRIX
<img width="1019" height="159" alt="image" src="https://github.com/user-attachments/assets/ad750854-ed42-4b7c-8d4d-27acefd70d3b" />
CLASSIFICATION REPORT & PREDICTION
<img width="1042" height="448" alt="image" src="https://github.com/user-attachments/assets/ef5cec29-8b7e-42f7-a5ff-3743922ee89b" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
