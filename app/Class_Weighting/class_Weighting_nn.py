import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, TimeDistributed, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


data =  pd.read_csv("/Dataset/Processed/cleaned_data.csv", header=0)

# Prepare word and tag mappings
words = list(set(data["WORD"].values))
n_words = len(words)
tags = list(set(data["SPEECH TAG"].values))
n_tags = len(tags)

word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1
word2idx["PAD"] = 0

tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for t, i in tag2idx.items()}

# Group words and tags into sentences
sentences = []
sentence_tags = []

temp_sentence = []
temp_tags = []

for index, row in data.iterrows():
    word = row['WORD']
    tag = row['SPEECH TAG']
    if word == ".":
        if temp_sentence:  # if the sentence is not empty
            sentences.append(temp_sentence)
            sentence_tags.append(temp_tags)
            temp_sentence = []
            temp_tags = []
    else:
        temp_sentence.append(word)
        temp_tags.append(tag)

# Add the last sentence if it's not empty
if temp_sentence:
    sentences.append(temp_sentence)
    sentence_tags.append(temp_tags)

# Convert words and tags to indices
X = [[word2idx.get(w, word2idx["UNK"]) for w in s] for s in sentences]
y = [[tag2idx[t] for t in s] for s in sentence_tags]

#pad sequences
maxlen = 50  
X = pad_sequences(X, maxlen=maxlen, padding='post')
y = pad_sequences(y, maxlen=maxlen, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# class weights
y_flat = [item for sublist in y_train for item in sublist] 
class_weights = compute_class_weight('balanced', classes=np.unique(y_flat), y=y_flat)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Categorical
y_train = np.array([to_categorical(i, num_classes=n_tags) for i in y_train])
y_test = np.array([to_categorical(i, num_classes=n_tags) for i in y_test])

model = Sequential()
model.add(Embedding(input_dim=n_words + 2, output_dim=50, input_length=maxlen, mask_zero=True))
model.add(Bidirectional(LSTM(units=100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(TimeDistributed(Dense(n_tags, activation="softmax")))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.1, class_weight=class_weights_dict)

model.save('Dataset/pos_model_improved.h5')
