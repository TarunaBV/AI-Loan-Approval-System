from flask import Flask, render_template
import joblib

app = Flask(__name__)

model = joblib.load('model/loan_model.pkl')

encoders = joblib.load('model/encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)