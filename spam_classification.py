# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('combined_data.csv')

data.head()

data.shape

counts = data['label'].value_counts()
print(counts)

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Preprocess data
data['text'] = data['text'].apply(lambda x: x.lower())  # Lowercase text
data['text'] = data['text'].str.replace('[^\w\s]', '', regex=False)  # Remove punctuation

num_words_in_dataset = data['text'].str.split().explode().nunique()

print(f"Number of unique words in the dataset: {num_words_in_dataset}")

from gensim.models import Word2Vec
# Train the Word2Vec model
corpus = [doc.split() for doc in data['text']]
Word2Vecmodel = Word2Vec(sentences=corpus, vector_size=100, window=10, min_count=3, workers=6)
# Tokenize text data
tokenizer = Tokenizer(num_words=num_words_in_dataset, oov_token='<OOV>')
tokenizer.fit_on_texts(data['text'])

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_data['text'])
val_sequences = tokenizer.texts_to_sequences(val_data['text'])
test_sequences = tokenizer.texts_to_sequences(test_data['text'])

# Pad sequences
train_padded = pad_sequences(train_sequences, maxlen=256, truncating='post', padding='post')
val_padded = pad_sequences(val_sequences, maxlen=256, truncating='post', padding='post')
test_padded = pad_sequences(test_sequences, maxlen=256, truncating='post', padding='post')

# Define the vocabulary size and embedding matrix
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in word_index.items():
    if word in Word2Vecmodel.wv.key_to_index:
        embedding_matrix[i] = Word2Vecmodel.wv[word]

#define hypermaraters
embedding_dim = 200
max_length = 256

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, LSTM, GlobalMaxPooling1D, Dense, BatchNormalization
from tensorflow.keras.regularizers import l1

# define the model

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Dropout(0.2),
    Conv1D(128, 5, activation='relu'),
    BatchNormalization(),
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    LSTM(32, return_sequences=True),
    BatchNormalization(),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='BinaryCrossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(train_padded, train_data['label'], validation_data=(val_padded, val_data['label']), epochs=5, batch_size=64)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_padded, test_data['label'])
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

#Save the model
model.save('Spam_Classification_model.h5')

# Evaluate the model on custom text

sample_text = 'Nigeria on Tuesday destroyed 2.5 tonnes of seized elephant tusks valued at over 9.9 billion naira ($11.2 million) in a push to protect its dwindling elephant population from rampant wildlife traffickers. Over the past three decades, Nigeria’s elephant population has declined drastically from an estimated 1,500 to less than 400 due to poaching for ivory, habitat loss and human-elephant conflict, according to conservationists.'

# Convert the text to a padded sequence
sequence = tokenizer.texts_to_sequences([sample_text])
padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

# Get the probability of being spam
probability = model.predict(padded_sequence)[0][0]

# Determine spam based on the confidence level
confidence_level = round(probability * 100, 2)

if probability > 0.5:
    print(f"Spam Confidence: {confidence_level}%")
    print("Prediction: Negative (Spam)")
else:
    print(f"Not Spam Confidence: {100 - confidence_level}%")
    print("Prediction: Positive (Not Spam)")

sample_text_1 = 'Hey Trybers! Jojo is here...but Jojo is sad. Japa syndrome has caught up with me. See, what no one talks about is how japa syndrome is affecting friendships. You cannot even build long-term relationships because before you know it, they are leaving for Slovakia. On one hand, you are happy for them. Yunno, you can just slip in a conversation about how you talked to your friend in the US or disturb them for dollars. On the other hand, you are losing real friendship bonds. Damn. '

# Convert the text to a padded sequence
sequence = tokenizer.texts_to_sequences([sample_text_1])
padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

# Get the probability of being spam
probability = model.predict(padded_sequence)[0][0]

# Determine spam based on the confidence level
confidence_level = round(probability * 100, 2)

if probability > 0.5:
    print(f"Negative (Spam) with Confidence: {confidence_level}%")
else:
    print(f"Positive (Not Spam) with Confidence: {100 - confidence_level}%")

import gradio as gr

def classify_spam(text):
    # Convert the text to a padded sequence
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

    # Get the probability of being spam
    probability = model.predict(padded_sequence)[0][0]

    # Determine spam based on the confidence level
    confidence_level = round(probability * 100, 2)

    if probability > 0.5:
        result = f"Negative (Spam) with Confidence: {confidence_level}%"
    else:
        result = f"Positive (Not Spam) with Confidence: {100 - confidence_level}%"

    return result

# Create a Gradio interface
iface = gr.Interface(
    fn=classify_spam,
    inputs="text",
    outputs="text",
    live=True,
)

# Launch the Gradio interface
iface.launch(share=True)