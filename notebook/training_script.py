import pandas as pd
import numpy as np
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Download NLTK data (only need to run once) ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# --- 1. Data Loading and Preprocessing ---
print("Loading and preprocessing data...")
df = pd.read_csv('train.csv')
df = df[['text', 'target']].dropna()

# Define the text processing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', text) # Remove HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special characters and numbers
    tokens = word_tokenize(text)
    processed_text = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(processed_text)

df['processed_text'] = df['text'].apply(preprocess_text)

# --- 2. Data Splitting ---
X = df['processed_text']
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Train Classic ML Models (Naive Bayes & Random Forest) ---

# Create and train the TF-IDF Vectorizer on the training data
print("Training TF-IDF Vectorizer...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Naive Bayes
print("Training Naive Bayes model...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_accuracy = accuracy_score(y_test, nb_model.predict(X_test_tfidf))
print(f"Naive Bayes Accuracy: {nb_accuracy*100:.2f}%")

# Train Random Forest
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test_tfidf))
print(f"Random Forest Accuracy: {rf_accuracy*100:.2f}%")

# --- 4. Train Deep Learning Model (LSTM) ---

# Create and train the Keras Tokenizer on the training data
print("Training Keras Tokenizer...")
max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, oov_token="<unk>")
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# Build and Train the LSTM model
print("Building and training LSTM model...")
lstm_model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = lstm_model.fit(
    X_train_padded, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test_padded, y_test),
    verbose=2
)
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test_padded, y_test)
print(f"LSTM Accuracy: {lstm_accuracy*100:.2f}%")

# --- 5. Save All Artifacts ---
print("\nSaving all models and preprocessors...")
# Save classic models
with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Save the LSTM model
lstm_model.save('lstm_model.h5')

# Save the Keras Tokenizer
with open('keras_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("All artifacts saved successfully!")