from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "AI Loan Approval System Running Successfully!"

if __name__ == '__main__':
    app.run(debug=True)