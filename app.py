import streamlit as st
import pickle
import nltk

nltk.download("stopwords")

# Load the saved model and vectorizer
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# App title and description
st.title("AI-Assisted Toxic Comment Detector")
st.write("This tool predicts whether a comment is toxic or non-toxic.")

# User input
user_input = st.text_area("Enter a comment below:")

# Prediction button
if st.button("Check Toxicity"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        # Preprocess input (same style as training)
        cleaned_input = user_input.lower()
        
        # Convert text to numerical features
        input_vector = vectorizer.transform([cleaned_input])
        
        # Make prediction
        prediction = model.predict(input_vector)[0]
        confidence = model.predict_proba(input_vector).max()

        # Display result
        if prediction == 1:
            st.error(f"⚠️ Toxic Comment Detected (Confidence: {confidence:.2%})")
        else:
            st.success(f"✅ Comment is Non-Toxic (Confidence: {confidence:.2%})")
