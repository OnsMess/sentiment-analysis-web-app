from flask import Flask, render_template, request
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words("english")]
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    input_text = None
    if request.method == "POST":
        input_text = request.form["text"]
        cleaned_text = preprocess_text(input_text)
        text_tfidf = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_tfidf)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return render_template("index.html", sentiment=sentiment, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
