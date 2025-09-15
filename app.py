from flask import Flask, request, render_template, jsonify
import joblib
from keras.models import load_model
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)

predefined_disasters = ["earthquake", "flood", "fire", "hurricane", "tornado", "explosion", "accident", "tsunami", "volcano", "drought", "pandemic", "wildfire", "avalanche", "cyclone", "landslide", "heatwave", "blizzard", "terrorist attack", "nuclear meltdown", "chemical spill"]

# Load the saved models
naive_bayes_model = joblib.load('models/naive_bayes_model.pkl')
random_forest_model = joblib.load('models/random_forest_model.pkl')
lstm_model = load_model('models/lstm_model.h5')

# Load the TF-IDF vectorizer if you have previously saved it
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    matched_disasters = []
    is_disaster = "Not a Disaster"
    text = "Not Entered any Comment"
    if request.method == 'POST':
        data = request.form['text']
        words = data.split()
        text = data

        # Tokenize the text data
        max_sequence_length = 100
        max_words = 1000  # You can adjust this based on your data
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts([text])

        # Use the TF-IDF vectorizer to transform the input
        custom_input_tfidf = tfidf_vectorizer.transform([text])

        # Tokenize and pad the input for the LSTM model
        custom_input_seq = tokenizer.texts_to_sequences([text])
        custom_input_padded = pad_sequences(custom_input_seq, maxlen=max_sequence_length)

        # Make predictions using individual models
        lstm_prediction = lstm_model.predict(custom_input_padded)
        naive_bayes_prediction = naive_bayes_model.predict(custom_input_tfidf)
        random_forest_prediction = random_forest_model.predict(custom_input_tfidf)

        # Create an ensemble prediction
        ensemble_prediction = (lstm_prediction + naive_bayes_prediction + random_forest_prediction) / 3

        # Determine if the prediction is for a disaster or not
        is_disaster = "Disaster" if ensemble_prediction[0] >= 0.5 else "Not a Disaster"
        if ensemble_prediction[0] >= 0.5:
             matched_disasters = [word for word in words if word.lower() in predefined_disasters]

    return render_template('index.html', is_disaster = is_disaster, text = text, matched_disasters = matched_disasters)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, debug=True)
