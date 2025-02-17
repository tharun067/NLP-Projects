from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


def fake_news_det(news):
    # Transform the text using the saved vectorizer
    text_transformed = vectorizer.transform([news])
    
    # Get prediction from the model
    prediction = model.predict(text_transformed)[0]
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        prediction = fake_news_det(message)
        # Map prediction to a human-readable label (e.g., 'Fake' or 'Real')
        result = 'Fake' if prediction == 1 else 'Real'
        return render_template('index.html', prediction=result, news_text=message)
    else:
        return render_template('index.html', prediction="Ops, Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)
