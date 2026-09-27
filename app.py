
import streamlit as st
import pandas as pd
import joblib
import os

# Function to load a .pkl file with error handling
def load_pkl_file(filename):
    try:
        return joblib.load(filename)
    except FileNotFoundError:
        st.error(f"Error: The file '{filename}' was not found. Please ensure all .pkl files are in the same directory as app.py.")
        st.stop() # Stop the Streamlit app if files are missing
    except Exception as e:
        st.error(f"An unexpected error occurred while loading '{filename}': {e}")
        st.stop()

# Load the trained model and preprocessing objects
rf_model = load_pkl_file('random_forest_stroke_model.pkl')
imputer = load_pkl_file('median_imputer_stroke.pkl')
scaler = load_pkl_file('scaler_stroke.pkl')
feature_columns = load_pkl_file('feature_columns_stroke.pkl')

# Streamlit App Title
st.title('Stroke Prediction App')
st.write('Enter patient details to predict stroke risk.')

# Input fields for user data
st.header('Patient Information')

age = st.slider('Age', 0.08, 82.0, 40.0)
hypertension = st.radio('Hypertension', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
heart_disease = st.radio('Heart Disease', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
avg_glucose_level = st.slider('Average Glucose Level', 55.12, 271.74, 100.0)
bmi = st.slider('BMI', 10.3, 97.6, 25.0)

gender_options = {'Male': 1, 'Female': 0, 'Other': 2}
gender = st.selectbox('Gender', list(gender_options.keys()))

ever_married = st.radio('Ever Married', ['Yes', 'No'])
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked'])
residence_type = st.radio('Residence Type', ['Urban', 'Rural'])
smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Create a dictionary from user inputs
user_data = {
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
}

# Handle one-hot encoding for categorical features
# Initialize all OHE columns to 0 first
for col in feature_columns:
    if any(cat_prefix in col for cat_prefix in ['gender_', 'ever_married_', 'work_type_', 'Residence_type_', 'smoking_status_']):
        user_data[col] = 0

# Set the value for the selected gender
if gender == 'Male':
    user_data['gender_1'] = 1
elif gender == 'Other':
    user_data['gender_2'] = 1

# Set the value for ever_married
if ever_married == 'Yes':
    user_data['ever_married_Yes'] = 1

# Set the value for work_type
if work_type == 'Never_worked':
    user_data['work_type_Never_worked'] = 1
elif work_type == 'Private':
    user_data['work_type_Private'] = 1
elif work_type == 'Self-employed':
    user_data['work_type_Self-employed'] = 1
elif work_type == 'children':
    user_data['work_type_children'] = 1

# Set the value for residence_type
if residence_type == 'Urban':
    user_data['Residence_type_Urban'] = 1

# Set the value for smoking_status
if smoking_status == 'formerly smoked':
    user_data['smoking_status_formerly smoked'] = 1
elif smoking_status == 'never smoked':
    user_data['smoking_status_never smoked'] = 1
elif smoking_status == 'smokes':
    user_data['smoking_status_smokes'] = 1

# Convert to DataFrame, ensuring all feature columns are present and in the correct order
input_df = pd.DataFrame([user_data])
input_df = input_df[feature_columns] # Ensure correct column order and add missing OHE columns

# Apply the imputer to the entire input_df (it was fitted on all features).
# The imputer returns a numpy array.
input_imputed_array = imputer.transform(input_df)

# Convert the imputed array back to a DataFrame to maintain column names for the scaler.
input_imputed_df = pd.DataFrame(input_imputed_array, columns=feature_columns, index=input_df.index)

# Scale numerical features
input_scaled = scaler.transform(input_imputed_df)


if st.button('Predict Stroke Risk'):
    prediction = rf_model.predict(input_scaled)
    prediction_proba = rf_model.predict_proba(input_scaled)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error('Based on the provided information, the model predicts a HIGH risk of stroke.')
    else:
        st.success('Based on the provided information, the model predicts a LOW risk of stroke.')

    st.write(f"Probability of No Stroke: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Stroke: {prediction_proba[0][1]:.2f}")

    st.write('---')
    st.write('Disclaimer: This is a predictive model and should not be used as a substitute for professional medical advice.')
