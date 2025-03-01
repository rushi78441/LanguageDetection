from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('language_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')  

@app.route('/', methods=['GET'])  
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  
def predict():
    text = request.form.get("text")
    text_vectorized = vectorizer.transform([text])
    predicted_language = model.predict(text_vectorized)[0]

    return render_template('index.html', language=predicted_language)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == '__main__':
    app.run(debug=True)
