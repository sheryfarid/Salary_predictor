import streamlit as st
import pandas as pd
import pickle
from word2number import w2n

# Load the trained model
with open("salary_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Salary Prediction App 💰")

# User Input Fields
experience = st.text_input("Enter experience (e.g., zero, five, ten):")
test_score = st.number_input("Enter test score (out of 10):", min_value=0.0, max_value=10.0, step=0.1)
interview_score = st.number_input("Enter interview score (out of 10):", min_value=0.0, max_value=10.0, step=0.1)

# Convert experience word to number
if experience:
    try:
        experience = w2n.word_to_num(experience)
    except ValueError:
        st.error("Invalid experience input. Please enter a valid number word.")

# Predict Button
if st.button("Predict Salary"):
    if isinstance(experience, int):  # Ensure experience is a valid number
        input_data = pd.DataFrame([[experience, test_score, interview_score]], 
                                  columns=["experience", "test_score(out of 10)", "interview_score(out of 10)"])
        prediction = model.predict(input_data)
        st.success(f"Predicted Salary: ${prediction[0]:,.2f}")
    else:
        st.error("Please enter valid inputs!")
