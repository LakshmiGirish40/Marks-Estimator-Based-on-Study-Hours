#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
data = pd.read_csv(r'D:\Data_Science&AI\Spyder\Student_Marks_prediction\student_info.csv')
df = data.copy()
#----------------------------
df.head()
descriptive_stats = df.describe()
info = df.info()
#==============================
#Visualization
plt.scatter(x=df['study_hours'],y=df['student_marks'])
plt.xlabel("Student study hours")
plt.ylabel("Student marks")
plt.title("Scatter plot of Student study hours VS Student_marks")
plt.show()
#The plot show Linear regression Model(the data points are linear)
#====================================================================
#DataCleaning
nullvalues = df.isnull().sum() #student hours hasnull values

#find the mean value
df.mean()
df2 = df.fillna(df.mean())
print(df2.isnull().sum())
#nonull values and missing data in dataset
#====================================================================
#Split the data
X = df2.drop("student_marks", axis =1) #dependent
y = df2.drop("study_hours", axis = 1)#dependent

#check shape of X and y
print("Shape of X", X.shape)
print("Shape of y", y.shape)

#model training
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#fit the model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

#Find Coefficient (Linear Regression = y = mx+c )
coef = lin_reg.coef_
interceptc = lin_reg.intercept_ #c value
#prediction
y_pred = lin_reg.predict(X_test) #Prediction for X_test always
#========================================================
#passing Study hours and check
y_pred_4 =lin_reg.predict([[4]])[0][0].round(2) #(four hours study)
y_pred_4 =lin_reg.predict([[10]])[0][0].round(2) #(four hours study)
#===========================================================
student_data_prediction_table = pd.DataFrame(np.c_[X_test, y_test, y_pred], columns = ["study_hours", "student_marks_original","student_marks_predicted"])
#Fine tune your model or model evalution or performance 
score = lin_reg.score(X_test,y_test) 
#Visualization for trianed models
plt.scatter(X_train,y_train)

#==============================================
#Visualization for testing models 
plt.scatter(X_test, y_test)
plt.plot(X_train, lin_reg.predict(X_train), color = "r")

#=================================

# Check model performance
bias = lin_reg.score(X_train, y_train) # Training Score (Bias Approximation)
variance = lin_reg.score(X_test, y_test) #Test Score (Variance Approximation)
train_mse = mean_squared_error(y_train, lin_reg.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

#Save the ML model(Trained and test model)
import joblib
joblib.dump(lin_reg,"student_mark_predictor.pkl")
 #------------------------------------------------------
model = joblib.load("student_mark_predictor.pkl")


student_data_prediction_table .to_csv('student_data_prediction.csv', index=False)

import os 
# Get the current working directory
print(os.getcwd())
