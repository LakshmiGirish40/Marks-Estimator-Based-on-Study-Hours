# Student_Marks_Prediction
**Project Title: Marks Estimator Based on Study Hours Using Linear Regression**

**1. Project Overview:** 
This project builds a machine learning model using linear regression to predict students' exam marks based on the number of hours they study. The model is trained using historical data of student study hours and corresponding marks, and then used to predict marks for new study durations. The project also includes data visualization, model performance evaluation, and saving/loading the model for future use. 

**2. Objective:** 
 - The key objectives of the project are:
 - To understand the relationship between study hours and student marks.
 -  To predict student marks given a certain number of study hours using a linear regression model.
 -  To evaluate the performance of the model and visualize the results.
 -  To save the trained model for future predictions. 

**3. Project Workflow:** 
   - **Step 1: Data Collection**
     - The dataset used in this project is assumed to be in CSV format, where it contains two columns: 
    - **study hours:**  Number of hours studied. 
     - **student marks:**  Corresponding marks obtained by students. 
   -  **Step 2: Data Preprocessing**
    - **Data Exploration:** Display the first few rows, descriptive statistics, and check for null values. 
   - **Handling Missing Values:** If there are missing values in the dataset, they are filled using the mean value of the respective column. 

  - **Step 3: Visualization**
    - A scatter plot is created to visualize the relationship between study hours and student marks. 
    - The regression line is plotted on top of the data points to show the linear relationship. 

 - **Step 4: Model Training**
    - The Linear Regression model from the scikit-learn library is used. 
    - The dataset is split into training and testing sets using train_test_split (80% for training and 20% for testing). 
    - The model is trained on the training data (X_train, y_train). 

  - **Step 5: Model Evaluation** 
    -  The model's performance is evaluated using the mean_squared_error function from scikit-learn.
    -  The predicted values for the test set are compared against the actual marks. 

 -   **Step 6: Prediction** 
     - After training, the model is used to predict the marks for a student who studies for a certain number of hours, such as 4 or 8 hours. 
  - **Step 7: Model Deployment (Optional)**
    - The trained model is saved using joblib, allowing future predictions without retraining the model.
    - The model is then loaded for making predictions on new data. 

  - **5. Expected Output:**
    - Scatter plot showing the relationship between study hours and marks.
    - A linear regression model that predicts the marks based on input study hours.
    - Performance evaluation of the model, along with MSE values.
    - A trained model saved as a .pkl file for future use.
    - CSV file containing the original and predicted marks. 

 - **Tools and Libraries:** 
 **Python:** Programming language for data processing and model building. 
 **Pandas:** For data manipulation. 
 **NumPy:** For numerical operations. 
 **Matplotlib:** For visualizations. 
 **Scikit-learn:** For building and evaluating the regression model. 
 **Joblib:** For saving and loading the trained model. 

**7. Conclusion:**
This project successfully demonstrates how to apply linear regression to predict student marks based on the hours they study. By using machine learning techniques and data visualization, the project provides valuable insights into academic performance trends and allows students to estimate their potential exam scores based on their study habits. 

for output view: https://marks-estimator-based-on-study-hours.streamlit.app/

 

 

 

 

 
