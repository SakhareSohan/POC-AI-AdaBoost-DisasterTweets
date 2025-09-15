from flask import Flask, request, render_template
import pickle
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# --- Load all saved artifacts ---
print("Loading pre-trained models and preprocessors...")
# Load classic models and vectorizer
with open('naive_bayes_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load LSTM model and tokenizer
lstm_model = load_model('lstm_model.h5')
with open('keras_tokenizer.pkl', 'rb') as f:
    keras_tokenizer = pickle.load(f)
print("All artifacts loaded successfully.")

# --- Preprocessing function (must be identical to training) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    processed_text = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(processed_text)

# --- Prediction function ---
def predict_ensemble(text):
    # 1. Preprocess the input text
    processed_text = preprocess_text(text)

    # 2. Get predictions from classic models using TF-IDF
    text_tfidf = tfidf_vectorizer.transform([processed_text])
    nb_pred = nb_model.predict(text_tfidf)[0]
    rf_pred = rf_model.predict(text_tfidf)[0]

    # 3. Get prediction from LSTM model using Keras Tokenizer
    max_len = 100
    text_seq = keras_tokenizer.texts_to_sequences([processed_text])
    text_padded = pad_sequences(text_seq, maxlen=max_len, padding='post', truncating='post')
    lstm_prob = lstm_model.predict(text_padded)[0][0]
    lstm_pred = 1 if lstm_prob >= 0.5 else 0

    # 4. Implement Majority Vote Ensemble
    all_predictions = [nb_pred, rf_pred, lstm_pred]
    final_prediction = 1 if sum(all_predictions) >= 2 else 0 # Majority vote (2 or 3 '1's)
    
    if final_prediction == 1:
        return "Real Disaster"
    else:
        return "Not a Real Disaster"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = ""
    text_input = ""
    
    if request.method == 'POST':
        text_input = request.form['text']
        if text_input:
            prediction_result = predict_ensemble(text_input)

    return render_template('index.html', 
                           prediction=prediction_result, 
                           text=text_input)

if __name__ == '__main__':
    app.run(debug=True)