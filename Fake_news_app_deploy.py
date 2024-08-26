# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:50:15 2024

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:39:16 2024

@author: Administrator
"""

import streamlit as st
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the saved model
with open('random_forest_model.pkl', 'rb') as f:
    clf = pickle.load(f)

# Load the dataset
data = pd.read_csv('Fake_news_detection.csv')

# Create a Streamlit app
st.title("Fake News Detection App")
st.write("This app uses a Random Forest Classifier to predict whether a news article is fake or real based on its word count, number of sentences, unique words, and average word length.")

# Create input fields for the user to enter the text features
word_count = st.number_input("Word Count:", min_value=10, value=100)
num_sentence = st.number_input("Number of Sentences:", min_value=4, value=15)
unique_words = st.number_input("Unique Words:", min_value=5, value=50)
avg_word_length = st.number_input("Average Word Length:", min_value=0.0, value=6.999798916)

# Create a button to trigger the prediction
predict_button = st.button("Predict")

# Create a placeholder for the prediction result
result_placeholder = st.empty()

# Define a function to make predictions
def make_prediction(word_count, num_sentence, unique_words, avg_word_length):
    # Preprocess the input data
    X_new = pd.DataFrame({
        'Word_Count': [word_count],
        'Number_of_Sentence': [num_sentence],
        'Unique_Words': [unique_words],
        'Average_Word_Length': [avg_word_length]
    })
    
    # Make predictions using the loaded model
    y_pred = clf.predict(X_new)
    
    # Return the predicted label
    return y_pred[0]

# Define a function to display the prediction result
def display_result(label):
    if label == 0:
        result_placeholder.write("Predicted label: Real News")
    else:
        result_placeholder.write("Predicted label: Fake News")

# Trigger the prediction when the button is clicked
if predict_button:
    label = make_prediction(word_count, num_sentence, unique_words, avg_word_length)
    display_result(label)