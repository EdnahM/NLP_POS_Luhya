import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

from app.helper_functions_app import pos_tag_colors, pos_tag_meanings

# Your existing preprocessing code to create word2idx and tag2idx
data = pd.read_csv("cleaned_data_bukususection.csv", header=0)

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

words = list(set(data["WORD"].values))
n_words = len(words)
tags = list(set(data["SPEECH TAG"].values))
n_tags = len(tags)

word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1
word2idx["PAD"] = 0
tag2idx = {t: i for i, t in enumerate(tags)}

data["word_idx"] = data["WORD"].map(word2idx)
data["tag_idx"] = data["SPEECH TAG"].map(tag2idx)

# Save word2idx
with open("word2idx.json", "w") as f:
    json.dump(word2idx, f)

# Save tag2idx
with open("tag2idx.json", "w") as f:
    json.dump(tag2idx, f)

# Load the trained model and necessary data
model = load_model("pos_model_explicit_initializers_v1.h5")

# Load the word and tag dictionaries
with open("word2idx.json", "r") as f:
    word2idx = json.load(f)
with open("tag2idx.json", "r") as f:
    tag2idx = json.load(f)

idx2word = {v: k for k, v in word2idx.items()}
idx2tag = {v: k for k, v in tag2idx.items()}

pos_tag_colors =  pos_tag_colors
pos_tag_meaning =  pos_tag_meanings

# Streamlit application
st.markdown("""
    <style>
    .header {
        padding: 10px;
        text-align: center;
        font-size: 25px;
        font-weight: bold;
    }
    .footer {
        padding: 10px;
        text-align: center;
        font-size: 15px;
    }
    .pos-button {
        display: inline-block;
        padding: 10px 15px;
        margin: 5px;
        font-size: 16px;
        color: white;
        background-color: #4CAF50;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        text-decoration: none;
    }
    .pos-button:hover {
        background-color: #45a049;
    }
    </style>
    <div class="header">
        POS Tagging Application
    </div>
    """, unsafe_allow_html=True)

# Display POS tags and their meanings as buttons
st.subheader("POS Tags Representation")
pos_tags_html = "".join([f"<a href='#' class='pos-button' style='background-color:{color}'>{tag} = {meaning}</a>" for tag, color in pos_tag_colors.items() for key, meaning in pos_tag_meanings.items() if tag == key])
st.markdown(pos_tags_html, unsafe_allow_html=True)

# Display part of the visuals
st.subheader("Dataframe  Sample Preview")
st.dataframe(data.head(5))

# Display histogram for count distribution
st.subheader("Count Distribution of POS Tags")
tag_counts = data["SPEECH TAG"].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=tag_counts.index, y=tag_counts.values, color="purple", ax=ax)
ax.set_title("Count Distribution of POS Tags")
ax.set_xlabel("POS Tag")
ax.set_ylabel("Count")
st.pyplot(fig)

# TensorBoard visuals
st.subheader("TensorBoard Visuals")
st.markdown("View the TensorBoard logs for training metrics [here](logs).")


# Search functionality
st.subheader("Search Dataset for a Word")
search_word = st.text_input("Enter a word to search in the dataset:")
if st.button("Search"):
    if search_word:
        # Strip the search word of any leading/trailing whitespace
        search_word = search_word.strip()
        # Case insensitive search
        search_results = data[data["WORD"].fillna("").str.strip().str.contains(f"(?i)^{search_word}$")]
        if not search_results.empty:
            st.write(f"Search results for '{search_word}':")
            st.dataframe(search_results.head(1))  # Display only the first match
        else:
            st.write(f"No results found for '{search_word}'.")
    else:
        st.write("Please enter a word to search.")
        
st.subheader("POS Tag Prediction")
user_input = st.text_area("Enter a sentence for POS tagging:")
if st.button("Predict"):
    # Preprocess user input
    input_tokens = user_input.split()
    input_indices = [word2idx.get(word, word2idx["UNK"]) for word in input_tokens]
    max_len = 50  # Ensure this matches the max length the model was trained on
    input_padded = pad_sequences([input_indices], maxlen=max_len, padding="post")

    # Predict
    predictions = model.predict(input_padded)
    st.write(f"Predictions: {predictions}")
    
     # Print out the intermediate values for debugging
    st.write(f"Input Tokens: {input_tokens}")
    st.write(f"Input Indices: {input_indices}")
    st.write(f"Padded Input: {input_padded}")


    predicted_tags = np.argmax(predictions, axis=-1).flatten()
    st.write(f"Predicted Tags: {predicted_tags}")

    # Display predictions with confidence levels above 0.95
    prediction_html = ""
    for word, tag_idx, pred in zip(input_tokens, predicted_tags, predictions[0]):
        tag = idx2tag.get(tag_idx, "UNKNOWN")
        confidence = np.max(pred)
        color = pos_tag_colors.get(tag, "black")
        prediction_html += f" {word} <span style='color:{color}'> ({tag} - {confidence:.2f})</span> "

    if prediction_html:
        st.markdown(prediction_html, unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:red'>No predictions with confidence level above 0.95</span>", unsafe_allow_html=True)
           
        
        

# st.subheader("POS Tag Prediction")
# user_input = st.text_area("Enter a sentence for POS tagging:")
# if st.button("Predict"):
#     input_tokens = user_input.split()
#     input_indices = [word2idx.get(word, word2idx["UNK"]) for word in input_tokens]
#     input_padded = pad_sequences([input_indices], maxlen=50, padding="post")

#     # Predict
#     predictions = model.predict(input_padded)
#     predicted_tags = np.argmax(predictions, axis=-1).flatten()

#     prediction_html = ""
#     for word, tag_idx, pred in zip(input_tokens, predicted_tags, predictions[0]):
#         tag = idx2tag.get(tag_idx, "UNKNOWN")
#         confidence = np.max(pred)
#         if confidence > 0.95:
#             color = pos_tag_colors.get(tag, "black")
#             prediction_html += f" {word} <span style='color:{color}'> ({tag} - {confidence:.2f})</span> "

#     if prediction_html:
#         st.markdown(prediction_html, unsafe_allow_html=True)
#     else:
#         st.markdown("<span style='color:red'>No predictions with confidence level above 0.95</span>", unsafe_allow_html=True)




# # Section to add text and do prediction
# st.subheader("POS Tag Prediction")
# user_input = st.text_area("Enter a sentence for POS tagging:")
# if st.button("Predict"):
#     # Preprocess user input
#     input_tokens = user_input.split()
#     input_indices = [word2idx.get(word, word2idx["UNK"]) for word in input_tokens]
#     input_padded = pad_sequences([input_indices], maxlen=50, padding="post")
    
#     # Predict
#     predictions = model.predict(input_padded)
#     predicted_tags = np.argmax(predictions, axis=-1).flatten()
    
#     # Display predictions with confidence levels above 0.95
#     prediction_html = ""
#     for word, tag_idx, pred in zip(input_tokens, predicted_tags, predictions[0]):
#         tag = idx2tag.get(tag_idx, "UNKNOWN")
#         confidence = np.max(pred)
#         if confidence > 0.95:
#             color = pos_tag_colors.get(tag, "black")
#             prediction_html += f" {word} <span style='color:{color}'> ({tag} - {confidence:.2f})</span> "
    
#     if prediction_html:
#         st.markdown(prediction_html, unsafe_allow_html=True)
#     else:
#         st.markdown("<span style='color:red'>No predictions with confidence level above 0.95</span>", unsafe_allow_html=True)
