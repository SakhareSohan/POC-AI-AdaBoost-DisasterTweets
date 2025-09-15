# Real-Time Disaster Tweet Classification with an Ensemble ML Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20TensorFlow-orange.svg)
![Libraries](https://img.shields.io/badge/Libraries-Scikit--learn%20%7C%20NLTK%20%7C%20Pandas-green.svg)

## 📌 Project Motivation & Real-World Impact

In the chaotic aftermath of a disaster, social media platforms like Twitter become a critical source of real-time information. However, the sheer volume of posts makes it nearly impossible for humanitarian organizations, news agencies, and first responders to separate actionable alerts from noise.

This project addresses this critical challenge by building a robust Natural Language Processing (NLP) pipeline and a machine learning model to automatically classify tweets, identifying those that are reporting on a real disaster. The goal is to create a tool that can help filter crucial information, enabling faster and more effective response efforts during emergencies.

---

## ✨ Project Summary & The Journey

This project showcases a complete end-to-end machine learning workflow, from raw text data to a deployed web application. The development followed a strategic journey:

1.  **Baseline Modeling:** I began by training and evaluating three distinct types of classification models to establish performance baselines:
    * **Naive Bayes:** A fast, probabilistic model.
    * **Random Forest:** A powerful tree-based model.
    * **LSTM (Long Short-Term Memory):** A deep learning model capable of understanding text sequence and context.

2.  **Ensemble Implementation:** Recognizing that no single model is perfect, I engineered a **majority vote ensemble**. This approach combines the predictions from all three models, leveraging their diverse strengths to create a final classifier that is more accurate, robust, and reliable than any of its individual components.

3.  **Deployment:** The final ensemble model was integrated into a user-friendly web application using **Flask**, allowing for real-time predictions on new, unseen text.



---

## 🛠️ Technical Stack & Skills Demonstrated

This project demonstrates a wide range of skills essential for a data science and machine learning role:

* **Languages & Frameworks:** Python, Flask, TensorFlow/Keras, Scikit-learn, NLTK, Pandas, NumPy
* **Data Preprocessing for NLP:**
    * Advanced text cleaning using regular expressions (regex).
    * Tokenization, stop-word removal, and lemmatization with NLTK.
* **Feature Engineering:**
    * Converting raw text into numerical features using TF-IDF vectorization.
    * Preparing text for deep learning models with Keras Tokenizer and sequence padding.
* **Machine Learning Modeling:**
    * Training and evaluating multiple model architectures (Probabilistic, Tree-based, and Neural Network).
    * Implementing a **Majority Vote Ensemble** to improve prediction accuracy and robustness.
* **Deep Learning for NLP:**
    * Building, compiling, and training a Recurrent Neural Network (LSTM) for sequence classification using Keras.
* **Model Deployment & Persistence:**
    * Building a RESTful API and web interface with Flask.
    * Saving and loading all necessary artifacts (models, vectorizers, tokenizers) using `pickle` and `HDF5` for a seamless prediction pipeline.

---

## 📂 Project Structure

```
.
├── models/
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── lstm_model.h5
│   └── keras_tokenizer.pkl
├── templates/
│   └── index.html
├── app.py                      # Main Flask application file
├── notebook                    # Notebook for the full training pipeline
│   └── Adaboost(Comment-Detection).ipynb
│   └── training_script.py
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📈 Results & Evaluation (Example)

Each model was evaluated on a hold-out test set. The ensemble model demonstrated a clear improvement over the individual baseline models.

| Model           | Test Set Accuracy |
| --------------- | ----------------- |
| Naive Bayes     | ~79.5%            |
| Random Forest   | ~78.8%            |
| LSTM            | ~79.0%            |
| **Ensemble (Vote)** | **~80.5%**    |

*(Note: You should replace these with your actual accuracy scores from the training notebook.)*

---

## 🚀 Setup and Usage

### Prerequisites
-   Python 3.9+
-   `pip` package manager

### Installation
1.  Clone the repository.
2.  Create and activate a virtual environment.
3.  Install dependencies: `pip install -r requirements.txt`
4.  Download NLTK data (run once in Python):
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

### Running the Project
1.  **Train the Models:** Run the `train_ensemble_models.py` notebook to train all models and generate the saved artifact files.
2.  **Launch the Web App:** Run the Flask application from your terminal:
    ```bash
    flask run
    ```
3.  Open your browser to `http://127.0.0.1:5000` to use the prediction system.