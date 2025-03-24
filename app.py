from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import torch
import pickle
from transformers import BertTokenizer, BertModel
import numpy as np
import os

MODEL_PATH = r"A:\Visual Files\Machine Learning Project\Resume Screening System\resume_classifier_bert.h5"
ENCODER_PATH = r"A:\Visual Files\Machine Learning Project\Resume Screening System\label_encoder.pkl"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

try:
    with open(ENCODER_PATH, "rb") as file:
        label_encoder = pickle.load(file)
except Exception as e:
    print(f"Error loading label encoder: {e}")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

def get_bert_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  

    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  
    return embedding

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if "resume" not in data:
            return jsonify({"error": "Missing 'resume' key in JSON"}), 400
        resume_text = data["resume"]

        resume_embedding = get_bert_embedding(resume_text)

        resume_embedding = resume_embedding.reshape(1, -1)

        prediction = model.predict(resume_embedding)
        predicted_class = np.argmax(prediction)

        
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]

        return jsonify({"predicted_category": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
