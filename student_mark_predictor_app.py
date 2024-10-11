import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the saved model
# Load the model
model = joblib.load("student_mark_predictor.pkl")
data = pd.read_csv(r'D:\Data_Science&AI\Spyder\Student_Marks_prediction\student_info.csv')
student_data_predicted = pd.read_csv("D:\Data_Science&AI\Spyder\Student_Marks_prediction\student_data_prediction.csv")
#student_data = pd.read_(r"D:\Data_Science&AI\Spyder\Student_Marks_prediction\Student_marks.py")
# Load the saved linear regression model

st.title("Student Marks Prediction App")
st.write("<h5 style='text-align: left; color:blue;'>Predict student marks based on the number of hours they study in a day.</h5>", unsafe_allow_html=True) 

# Add input widget for user to enter hours of study
hours = st.number_input("Enter how many hours the student studies:", min_value=1.0, max_value=24.0, step=1.0)

if st.button('Predict Marks'):
    marks = np.array([[hours]])  # Ensure it's a 2D array
    prediction = model.predict(marks)
    st.write(f"The predicted marks for {hours} of study: {prediction}") 
st.write("The model was trained using a dataset of student marks and hours of study.")
#==================================================================

st.write("<h4 style='text-align: left; color:Red;'>Visualization</h4>", unsafe_allow_html=True) 
st.write("<h5 style='text-align: left; color:green;'>Original_Data</h5>", unsafe_allow_html=True) 
fig = plt.scatter(x=data['study_hours'],y=data['student_marks'])
plt.xlabel("Student study hours")
plt.ylabel("Student marks")
plt.title("Scatter plot of Student study hours VS Student_marks")
plt.show()

st.scatter_chart(data)
st.write("<h5 style='text-align: left; color:green;'>Scatter_Chart(Student_mark Vs Study_hours Vs Student_marks_predicttion)</h5>", unsafe_allow_html=True) 
st.scatter_chart(student_data_predicted,x ='study_hours',y='student_marks_predicted',color="#04f")
st.write("<h5 style='text-align: left; color:green;'>Bar_Chart(Student_mark Vs Study_hours Vs Student_marks_predicttion)</h5>", unsafe_allow_html=True) 
st.bar_chart(student_data_predicted)
st.write("<h5 style='text-align: left; color:green;'>line_Chart(Student_mark Vs Study_hours Vs Student_marks_predicttion)</h5>", unsafe_allow_html=True) 
st.line_chart(student_data_predicted)

st.bar_chart(student_data_predicted )
st.write("<h5 style='text-align: left; color:red;'>Original_DataSet</h5>", unsafe_allow_html=True) 
st.write(pd.DataFrame(data))
st.write("<h5 style='text-align: left; color:red;'>Predicton DataSet</h5>", unsafe_allow_html=True) 
st.write(pd.DataFrame(student_data_predicted))


st.bar_chart(student_data_predicted)
st.write("<h5 style='text-align: left; color:red;'>Study_hours Vs Student_marks_predicted </h5>", unsafe_allow_html=True) 
st.bar_chart(student_data_predicted,x ='study_hours',y='student_marks_predicted')

# Write the custom HTML header
st.write("<h5 style='text-align: left; color:red;'>Study_hours Vs Student_marks_predicted</h5>", unsafe_allow_html=True) 

import plotly.express as px
# Create a bar chart with Plotly
fig = px.bar(student_data_predicted , x='study_hours', y='student_marks_predicted', color='student_marks_predicted',
             color_continuous_scale='Plasma', title='')

# Update layout for cleaner look
fig.update_layout(xaxis_title='Study Hours', yaxis_title='Predicted Marks')
# Display the chart in Streamlit
st.plotly_chart(fig)

# Display the chart in Streamlit

#streamlit run student_mark_predictor_app.py
