from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(script_dir, "my_model.h5")
label_encoder_path = os.path.join(script_dir, "label_encoder.pkl")

model = load_model(model_path, compile=False)
with open(label_encoder_path, "rb") as f:  
    label_encoder = pickle.load(f)

disease_info = {
    "akiec": "Actinic Keratoses — precancerous lesions caused by sun exposure.",
    "bcc": "Basal Cell Carcinoma — slow-growing cancer, rarely spreads.",
    "bkl": "Benign Keratosis-like lesions — non-cancerous skin growths.",
    "df": "Dermatofibroma — benign fibrous skin nodule.",
    "mel": "Melanoma — dangerous and aggressive skin cancer.",
    "nv": "Melanocytic Nevi — common mole, usually benign.",
    "vasc": "Vascular Lesion — benign blood vessel growths."
}

@app.route('/', methods=['GET'])
def home():
    """
    Sunucunun çalıştığını test etmek için ana sayfa.
    Tarayıcıdan http://127.0.0.1:5000/ adresine gidince burası çalışır.
    """
    return "Skin AI API başarıyla çalışıyor! Tahmin için /predict adresine POST isteği gönderin."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Görüntüleri alıp tahmin yapan ana API endpoint'i.
    """
    if 'image' not in request.files:
        return jsonify({"error": "Görüntü 'image' anahtarıyla gönderilmedi."}), 400

    file = request.files['image']
    
    try:
        IMG_SIZE = (96, 96)  
        img = Image.open(file.stream).convert("RGB").resize(IMG_SIZE)
        
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  

        preds = model.predict(img_array)
        pred_class_index = np.argmax(preds)
        confidence = float(np.max(preds)) 

        class_name = label_encoder[pred_class_index]

        return jsonify({
            "prediction": class_name,
            "confidence": confidence * 100,  
            "description": disease_info.get(class_name, "Bu hastalık için açıklama bulunamadı."),
            "all_probabilities": {label: float(prob) for label, prob in zip(label_encoder, preds[0])}
        })

    except Exception as e:
        return jsonify({"error": f"Bir hata oluştu: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)