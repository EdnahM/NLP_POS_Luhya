import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal, RandomUniform, GlorotUniform, Zeros
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, TimeDistributed, Dense
import pandas as pd
import json
from tensorflow.keras.models import Sequential

custom_objects = {
    'Sequential': Sequential,
    'Embedding': Embedding,
    'LSTM': LSTM,
    'Bidirectional': Bidirectional,
    'TimeDistributed': TimeDistributed,
    'Dense': Dense,
    'Orthogonal': Orthogonal,
    'RandomUniform': RandomUniform,
    'GlorotUniform': GlorotUniform,
    'Zeros': Zeros,
}

with open('version2_model/pos_model_architecture_version1.json', 'r') as f:
    model_json = json.load(f)

for layer in model_json['config']['layers']:
    if 'config' in layer and 'time_major' in layer['config']:
        del layer['config']['time_major']

model = tf.keras.models.model_from_json(json.dumps(model_json), custom_objects=custom_objects)

model.load_weights('version2_model/pos_model_weights_version2.h5')

# Load the dataset
data = pd.read_csv('cleaned_data.csv')  

words = list(set(data["WORD"].values))
n_words = len(words)
tags = list(set(data["SPEECH TAG"].values))
n_tags = len(tags)

word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1
word2idx["PAD"] = 0

tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for t, i in tag2idx.items()}


def predict_with_confidence(sentence, threshold=0.50):
    words = sentence.split()
    word_indices = [word2idx.get(word, word2idx["UNK"]) for word in words]
    word_indices_padded = pad_sequences([word_indices], maxlen=50, padding='post')
    
    predictions = model.predict(word_indices_padded)
    
    predicted_tags = []
    confidence_scores = []
    
    for pred in predictions[0]:
        max_prob = np.max(pred)
        tag_idx = np.argmax(pred)
        if max_prob >= threshold:
            predicted_tag = idx2tag[np.argmax(pred)]
        else:
            predicted_tag = 'UNK' 
            
        predicted_tags.append(predicted_tag)
        confidence_scores.append(pred[tag_idx])
    
    return predicted_tags[:len(words)] , confidence_scores[:len(words)]


pos_colors = {
    'N': 'blue',
    'V': 'green',
    'PRON': 'orange',
    'ADJ':'skyblue',
    'ADV':'yellow',
    'PREP':'purple',
    'CONJ': 'red',
    'NUM' :'salmon',
    'INTJ' : 'white',
    'ADP': 'brown',
    'xx' : 'violet',
    'DT' : 'maroon',
    'UNK': 'pink',
    'xx' : 'navy blue'
}


st.title("POS TAGGER BUKUSU LANGUAGE")
st.write("Enter a sentence or word to get POS tags:")

input_text = st.text_input("Input text:")


if st.button("Predict"):
    if input_text:
        predictions, confidence_scores = predict_with_confidence(input_text)
        if predictions:
            colored_text = ""
            words = input_text.split()
            for word, tag, conf in zip(words, predictions, confidence_scores):
                color = pos_colors.get(tag, 'black')
                colored_text += f'{word} <span style="color: {color};"> ({tag} - {conf:.2f}) </span> '
            st.markdown(colored_text, unsafe_allow_html=True)
        else:
            st.write("No predictions available for the input text.")
    else:
        st.write("Please enter some text to get POS tags.")