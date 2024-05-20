import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from class_Weighting_nn import word2idx, maxlen, idx2tag

# Load the model
model = load_model('Dataset//pos_model_improved.h5')

def predict_with_confidence(sentence):
    words = sentence.split()
    word_indices = [word2idx.get(word, word2idx["UNK"]) for word in words]
    word_indices_padded = pad_sequences([word_indices], maxlen=maxlen, padding='post')
    predictions = model.predict(word_indices_padded)

    predicted_tags = []
    confidence_scores = []

    for pred in predictions[0]:
        tag_idx = np.argmax(pred)
        predicted_tags.append(idx2tag[tag_idx])
        confidence_scores.append(pred[tag_idx])

    return predicted_tags[:len(words)], confidence_scores[:len(words)]


sentence = input("Kindly input your Luhya Sentence!!!")
predicted_tags, confidence_scores = predict_with_confidence(sentence)
print("Predicted Tags:", predicted_tags)
print("Confidence Scores:", confidence_scores)
