from flask import Flask, render_template, request, jsonify

import chatbot

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = chatbot.chat(text)[0]
    message = {"answer": response}
    return jsonify(message)

@app.post("/lenth")
def predict():
    text = request.get_json().get("message")
    lenth = len(chatbot.chat(text)[1])
    return lenth

@app.post("/message_tree")
def predict():
    text = request.get_json().get("message")
    response = chatbot.chat(text)[1]
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run()