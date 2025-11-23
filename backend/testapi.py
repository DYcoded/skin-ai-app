from flask import Flask, request, jsonify
from tensorflow.keras.models import Model 
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras import layers
from PIL import Image
import numpy as np
import json
import os
import tensorflow as tf
from flask_cors import CORS

# --- FLASK SETUP ---
app = Flask(__name__)
CORS(app)

# --- KONFİGÜRASYON ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# LÜTFEN İNDİRDİĞİNİZ DOSYA ADLARINI KONTROL EDİN
weights_path = os.path.join(script_dir, "my_skin_model.h5") 
label_encoder_path = os.path.join(script_dir, "class_indices.json")

# 1. MODEL MİMARİSİ (Functional API)
IMG_SIZE = (224, 224)
NUM_CLASSES = 7 

base_model = MobileNetV2(weights=None, include_top=False, input_shape=IMG_SIZE + (3,))

# KRİTİK: Eğitimdeki (frozen) durumu eşleştiriyoruz
base_model.trainable = False 

# Functional API ile custom başlığı oluşturma
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(NUM_CLASSES, activation='softmax')(x)

# Modeli oluşturma
model = Model(inputs=base_model.input, outputs=predictions)

# 2. AĞIRLIKLARI YÜKLEME (Son Hata Çözümü: by_name=True)
print("Model mimarisi oluşturuldu. Ağırlıklar yükleniyor...")
try:
    # by_name=True ile Keras bug'ını atlayıp ağırlıkları yüklüyoruz
    model.load_weights(weights_path, by_name=True) 
    print("Model başarıyla yüklendi!")
except Exception as e:
    print(f"HATA: Ağırlıklar yüklenemedi! Dosya {weights_path} konumunda mı? {e}")
    raise e

# 3. JSON YÜKLEME (Sınıf İsimlerini Düzgün Listeye Çevirme)
with open(label_encoder_path, "r") as f:
    loaded_data = json.load(f)

label_encoder = []

if isinstance(loaded_data, dict):
    # KRİTİK DÜZELTME: Anahtarları (key="0", "1", "2") sayıya çevirip sırala 
    # ve değerleri (sınıf isimlerini) al. Bu, çoğu online eğitimin kullandığı formattır.
    try:
        sorted_items = sorted(loaded_data.items(), key=lambda item: int(item[0]))
        label_encoder = [item[1] for item in sorted_items]
    except ValueError:
        # Eğer anahtarlar sayıya çevrilemezse, önceki mantığı kullan
        sorted_items = sorted(loaded_data.items(), key=lambda item: item[1])
        label_encoder = [item[0] for item in sorted_items]
    
elif isinstance(loaded_data, list):
    label_encoder = loaded_data
else:
    label_encoder = list(loaded_data.values())

# --- DİĞER BİLGİLER ---
disease_info = {
    "akiec": "Actinic Keratoses — precancerous lesions caused by sun exposure.",
    "bcc": "Basal Cell Carcinoma — slow-growing cancer, rarely spreads.",
    "bkl": "Benign Keratosis-like lesions — non-cancerous skin growths.",
    "df": "Dermatofibroma — benign fibrous skin nodule.",
    "mel": "Melanoma — dangerous and aggressive skin cancer.",
    "nv": "Melanocytic Nevi — common mole, usually benign.",
    "vasc": "Vascular Lesion — benign blood vessel growths."
}

# --- FLASK ROUTES ---

@app.route('/', methods=['GET'])
def home():
    return "Skin Cancer Prediction API is running (Functional API - MobileNetV2)!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Image not provided with key 'image'."}), 400

    file = request.files['image']
    
    try:
        # Preprocessing: Load, Resize, Convert to array
        IMG_SIZE = (224, 224)
        img = Image.open(file.stream).convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img)
        
        # KRİTİK: MobileNetV2'nin özel ön işleme fonksiyonu (Normalizasyon -1'den 1'e)
        img_array = preprocess_input(img_array) 
        
        img_array = np.expand_dims(img_array, axis=0) # Batch boyutunu ekle

        # Tahmin yapma
        preds = model.predict(img_array)
        pred_class_index = int(np.argmax(preds))
        confidence = float(np.max(preds))

        # Hata çözümü: Class adının string olduğundan emin oluyoruz
        class_name = str(label_encoder[pred_class_index])

        return jsonify({
            "prediction": class_name,
            "confidence": confidence * 100,
            "description": disease_info.get(class_name, "No description available for this disease."),
            "all_probabilities": {label: float(prob) for label, prob in zip(label_encoder, preds[0])}
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)