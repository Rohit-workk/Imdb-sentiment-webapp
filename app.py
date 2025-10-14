# app.py
from flask import Flask, render_template, request
import pickle
from preprocess import preprocess_text

# Initialize app
app = Flask(__name__)

# Load model and vectorizer
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('models/sentiment_logisticmodel.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    text = None
    if request.method == 'POST':
        text = request.form['text']
        cleaned_text = preprocess_text(text)
        text_vec = tfidf.transform([cleaned_text])
        prediction = model.predict(text_vec)[0]
        sentiment = prediction.capitalize()
    return render_template('index.html', text=text, sentiment=sentiment)





if __name__ == '__main__':
    app.run(debug=True)
