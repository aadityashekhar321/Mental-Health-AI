import streamlit as st
from transformers import pipeline   

classifier = pipeline("sentiment-analysis")  

st.title("Student Mental Health AI")
user_input = st.text_area("How are you feeling today?")
if st.button("Analyze"):
    st.write("Your input:", user_input)

    # Sentiment Analysis
    result = classifier(user_input)[0]
    label = result['label']
    score = result['score']

    st.write("Mood Prediction:", label)
    st.write("Confidence:", round(score, 2))
