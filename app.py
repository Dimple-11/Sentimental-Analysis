import pickle
import re
import nltk

from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download stopwords (only first time)
nltk.download("stopwords", quiet=True)

# Initialize Flask app
app = Flask(__name__)

# Load trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Load tokenizer
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# -----------------------------
# Text Preprocessing
# -----------------------------
def clean_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()
    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# -----------------------------
# Prediction Function
# -----------------------------
def predict_sentiment(text):
    text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)

    prediction = model.predict(padded_sequence, verbose=0)

    confidence = float(prediction[0][0])

    if confidence >= 0.5:
        sentiment = "😊 Positive"
    else:
        sentiment = "😡 Negative"

    return sentiment, round(confidence, 4)


# -----------------------------
# Home Route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None

    if request.method == "POST":
        text = request.form.get("text", "").strip()

        if text:
            result, confidence = predict_sentiment(text)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )


# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)