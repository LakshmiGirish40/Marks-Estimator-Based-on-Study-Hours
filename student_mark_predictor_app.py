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
data = pd.read_csv('student_info.csv')
student_data_predicted = pd.read_csv('student_data_prediction.csv')
#student_data = pd.read_('Student_marks.py')
# Load the saved linear regression model
#=============================================================================
import streamlit as st
import base64

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {{
        background: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: repeat;
        color: white;
    }}
    .css-1g8v9l0 {{
        background: rgba(255, 252, 253, 0.4);
        padding: 20px;
        border-radius: 10px;
        text-align: center; /* Centering the text */
    }}
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: white;
    }}
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
     .stButton > button {{
        background-color: #4C4C6D;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 5px 10px;
        font-size: 16px;
    }}
    .stButton > button:hover {{
        background-color: #6A5ACD;
    }}
    .stSlider > div {{
        background-color: transparent;
    }}
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Correctly call the function with the actual file path
set_background('image3.jpg')


#======================================================

#st.title("Student Marks Prediction App")
st.markdown("<h2 style='text-align: left; color:White'>Student Marks Prediction App</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: left; color:White'>Predict student marks based on the number of hours they study in a day.</h4>", unsafe_allow_html=True) 

# Add input widget for user to enter hours of study
#hours = st.number_input("Enter how many hours the student studies:", min_value=1.0, max_value=24.0, step=1.0)
#st.markdown('<p style="color:white;">Enter how many hours the student studies:</p>', unsafe_allow_html=True)
st.write('<h6 style="color:white;">Enter how many hours the student studies:</h6>', unsafe_allow_html=True)
hours = st.number_input("", min_value=1.0, max_value=24.0, step=1.0)

if st.button('Predict Marks'):
    marks = np.array([[hours]])  # Ensure it's a 2D array
    prediction = model.predict(marks)
    st.write(f"The predicted marks for {hours} of study: {prediction}") 
st.write('<h6 style="color:white;">The model was trained using a dataset of student marks and hours of study.</h6>', unsafe_allow_html=True)
#st.write("The model was trained using a dataset of student marks and hours of study.")
#==================================================================

st.write("<h4 style='text-align: left; color:Red;'>Visualization</h4>", unsafe_allow_html=True) 
st.write("<h5 style='text-align: left; color:sky blue;'>Original_Data</h5>", unsafe_allow_html=True) 
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
             color_continuous_scale='Plasma', title='plotly Graph for study_hours vs student_marks_predicted')

# Update layout for cleaner look
fig.update_layout(xaxis_title='Study Hours', yaxis_title='Predicted Marks')
# Display the chart in Streamlit
st.plotly_chart(fig)

# Display the chart in Streamlit

#streamlit run student_mark_predictor_app.py
