import streamlit as st
import pickle
import numpy as np

# Set page config
st.set_page_config(
    page_title="Calories Burned Predictor",
    page_icon=":fire:",
    layout="wide",
    initial_sidebar_state="auto",
)

# Load the saved model
model = pickle.load(open("model.pkl", "rb"))

# Define the function to make predictions
def predict_calories(gender, age, height, weight, duration, heart_rate, body_temp):
    # Create a numpy array with the input values
    input_data = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])

    # Use the loaded model to make a prediction
    prediction = model.predict(input_data)[0]

    return prediction

# Set up the Streamlit app
st.title("Calories Burned Calculator")
st.image("header.jpg", use_column_width=True)

facts = {
    'Running': 'Running can reduce your risk of heart disease and improve your mental health.',
    'Cycling': 'Cycling can help reduce stress and improve your overall fitness level.',
    'Swimming': 'Swimming is a low-impact exercise that is easy on your joints and can improve your cardiovascular health.',
    'Weightlifting': 'Weightlifting can help improve bone density and muscle mass, leading to a stronger and healthier body.'
}

st.subheader("Enter your information")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age (years)", min_value=1, max_value=120)
height = st.number_input("Height (cm)", min_value=1, max_value=300)
activity = st.sidebar.selectbox('Activity Type', ['Running', 'Cycling', 'Swimming', 'Weightlifting'])
weight = st.number_input("Weight (kg)", min_value=1, max_value=500)
duration = st.number_input("Duration of Exercise (minutes)", min_value=1, max_value=1440)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=1, max_value=250)
body_temp = st.number_input("Body Temperature (°C)", min_value=20, max_value=50)

# When the user clicks the 'Predict' button, make a prediction and display the result
if st.button("Predict Calories Burned"):
    # Convert the gender to a numeric value
    gender_num = 0 if gender == "Male" else 1

    # Make a prediction
    prediction = predict_calories(gender_num, age, height, weight, duration, heart_rate, body_temp)
    # Display the result
    st.success(f"You burned {prediction:.2f} calories during your {activity.lower()} exercise.")
else:
    st.write("Enter the values and click on the 'Predict Calories Burned' button to get the result.")

# Display the interesting fact for the selected activity
st.sidebar.markdown(f"**Interesting fact about {activity}:**")
st.sidebar.write(facts[activity])