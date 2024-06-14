import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

import keras
import tensorflow
print(keras.__version__)
print(tensorflow.__version__)

# Custom LSTM layer to handle any custom configurations
class CustomLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        # Remove 'time_major' from kwargs if it exists
        if kwargs and 'time_major' in kwargs:
            kwargs.pop('time_major')
        super(CustomLSTM, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super(CustomLSTM, self).get_config()
        return config

# Function to load the model with custom objects
def load_trained_model():
    custom_objects = {'Orthogonal': Orthogonal, 'CustomLSTM': CustomLSTM}
    try:
        model = load_model('pos_model_explicit_initializers.h5', custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess input data
def preprocess_data(data, word2idx, max_len=50):
    sentences = []
    sentence = []
    for _, row in data.iterrows():
        word_idx = word2idx.get(row["WORD"], word2idx["UNK"])
        sentence.append(word_idx)
        if row["WORD"] == ".":
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append(sentence)  # Add the last sentence if it doesn't end with a period
    X = pad_sequences(sentences, maxlen=max_len, padding="post")
    return X, len(sentences)

# POS color mapping
pos_colors = {
    'N': 'blue',
    'V': 'green',
    'PRON': 'orange',
    'ADJ':'skyblue',
    'ADV':'yellow',
    'PREP':'purple',
    'CONJ': 'red',
    'NUM' :'salmon',
    'INTJ' : 'violet',
    'ADP': 'brown',
    'UNKOWN' : 'black'
}

# Load the model
model = load_trained_model()

# Check if the model loaded correctly
if model is None:
    st.error("Failed to load model. Please check the model path and custom objects.")
else:
    # Load the word2idx and tag2idx mappings
    # Replace these mappings with your actual mappings
    word2idx = {'UNK': 1, 'PAD': 0}  # Example, replace with your actual word2idx dictionary
    tag2idx = {'TAG1': 0, 'TAG2': 1}  # Example, replace with your actual tag2idx dictionary
    idx2tag = {v: k for k, v in tag2idx.items()}
    unknown_tag = 'UNKNOWN'  # Define a default tag for unknown indices

    # Streamlit interface
    st.title("Part of Speech Tagging")

    # Language selection
    language = st.selectbox("Select Language", ["Luhya", "English", "Other"])

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.write(data.head())

            # Preprocess the data for prediction
            X, num_sentences = preprocess_data(data, word2idx)
            
            # Make predictions
            predictions = model.predict(X)
            predicted_tags = np.argmax(predictions, axis=-1)
            confidence_levels = np.max(predictions, axis=-1)
            
            # Flatten predicted tags to match the number of words
            predicted_tags_flat = [idx for sentence in predicted_tags for idx in sentence][:len(data)]
            confidence_levels_flat = [conf for sentence in confidence_levels for conf in sentence][:len(data)]
            
            # Convert indices to tags and colors
            data['PREDICTED_TAG'] = [idx2tag.get(idx, unknown_tag) for idx in predicted_tags_flat]
            data['CONFIDENCE'] = confidence_levels_flat
            data['COLOR'] = data['PREDICTED_TAG'].map(pos_colors).fillna('grey')  # Default to grey for unknown tags
            
            st.write("Predictions:")
            st.write(data.head())

            # Basic analysis: Part of Speech count
            pos_counts = data['PREDICTED_TAG'].value_counts()
            st.write("Part of Speech Counts:")

            # Create a purple bar chart for POS counts
            fig, ax = plt.subplots()
            pos_counts.plot(kind='bar', color='purple', ax=ax)
            ax.set_xlabel('Part of Speech')
            ax.set_ylabel('Count')
            ax.set_title('Part of Speech Count')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

    # Section for prediction
    st.header("Make Predictions")
    user_input = st.text_area("Enter text here")

    if st.button("Predict"):
        try:
            # Preprocess user input
            user_data = pd.DataFrame(user_input.split(), columns=["WORD"])
            X_user, num_sentences_user = preprocess_data(user_data, word2idx)
            
            # Make predictions
            predictions_user = model.predict(X_user)
            predicted_tags_user = np.argmax(predictions_user, axis=-1)
            confidence_levels_user = np.max(predictions_user, axis=-1)
            
            # Flatten predicted tags to match the number of words
            predicted_tags_user_flat = [idx for sentence in predicted_tags_user for idx in sentence][:len(user_data)]
            confidence_levels_user_flat = [conf for sentence in confidence_levels_user for conf in sentence][:len(user_data)]
            
            # Convert indices to tags and colors
            user_data['PREDICTED_TAG'] = [idx2tag.get(idx, unknown_tag) for idx in predicted_tags_user_flat]
            user_data['CONFIDENCE'] = confidence_levels_user_flat
            user_data['COLOR'] = user_data['PREDICTED_TAG'].map(pos_colors).fillna('grey')  # Default to grey for unknown tags
            
            st.write("Predictions for user input:")
            st.write(user_data)
        except Exception as e:
            st.error(f"Error making predictions: {e}")
