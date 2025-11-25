from flask import Flask, request, jsonify
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess_input
from tensorflow.keras import layers
from PIL import Image
import numpy as np
import json
import os
import tensorflow as tf
from flask_cors import CORS

# initialize Flask app
app = Flask(__name__)
CORS(app)

# define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, "final_resnet_oversampled_weights.weights.h5")
label_encoder_path = os.path.join(script_dir, "class_indices.json")

# define model architecture
IMG_SIZE = (380, 380)
NUM_CLASSES = 7

base_model = ResNet50V2(
    weights=None,
    include_top=False,
    input_shape=IMG_SIZE + (3,),
)

base_model.trainable = False
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)

predictions = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.load_weights(weights_path)
model.compile(optimizer='adam', loss='categorical_crossentropy')

print("Model architecture created. Weights are loading...")

# load weights
try:
    model.load_weights(weights_path)
    print("Model succesfully loaded!")
except Exception as e:
    print(f"FATAL ERROR: Weight loading failed! The error is: {e}")
    print("The weights and architecture DO NOT match. Check Dropout and Dense layer sizes.")
    raise e 

# load label encoder
with open(label_encoder_path, "r") as f:
    loaded_data = json.load(f)

label_encoder = [item[0] for item in sorted(loaded_data.items(), key=lambda item: item[1])]

if len(label_encoder) != NUM_CLASSES:
    print(f"WARNING: Num classes ({NUM_CLASSES}) does not match the number of classes in JSON ({len(label_encoder)}).")
    NUM_CLASSES = len(label_encoder)

# disease information dictionary
disease_info = {
    "akiec": "Actinic Keratoses — precancerous lesions caused by sun exposure. May develop into squamous cell carcinoma. Please consult a dermatologist for monitoring and treatment options.",
    "bcc": "Basal Cell Carcinoma — slow-growing cancer, rarely spreads. Usually appears as a pearly bump. Early treatment is important to prevent tissue damage. Consult a dermatologist for options.",
    "bkl": "Benign Keratosis-like lesions — non-cancerous skin growths. Generally harmless but should be monitored for changes. If you notice any changes in size, color, or shape, please see a dermatologist.",
    "df": "Dermatofibroma — benign fibrous skin nodule. Usually harmless but can be removed if bothersome. Consult a dermatologist for evaluation and removal options.",
    "mel": "Melanoma — dangerous and aggressive skin cancer. Early detection is crucial. If you notice changes in size, shape, or color of a mole, seek immediate medical attention from a dermatologist.",
    "nv": "Melanocytic Nevi — common mole, usually benign. Monitor for changes in size, color, or shape. If changes occur, consult a dermatologist for evaluation.",
    "vasc": "Vascular Lesion — benign blood vessel growths. Usually harmless but should be monitored. If you notice changes or symptoms, please see a dermatologist for evaluation."
}

FULL_NAME_MAP = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesion"
}

# define routes
@app.route('/', methods=['GET'])
def home():
    return "Skin Cancer Prediction API is running (Functional API - MobileNetV2)!"

# prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Image not provided with key 'image'."}), 400
    file = request.files['image']

    try:
        # preprocess image
        img = Image.open(file.stream).convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img)        
        processed_img = resnet_preprocess_input(img_array)
        processed_img = np.expand_dims(processed_img, axis=0)
        preds = model.predict(processed_img)
        probabilities = preds[0]

        top_indices = np.argsort(probabilities)[::-1][:2] 
        top_2_predictions = []

        
        # compile top 2 predictions with descriptions
        for i in top_indices:
            short_name = str(label_encoder[i])
            full_name = FULL_NAME_MAP.get(short_name, short_name)
            confidence = float(probabilities[i])
            
            description = disease_info.get(short_name, "No description available for this disease.")
            
            top_2_predictions.append({
                "class": full_name,
                "confidence": confidence * 100,
                "description": description if i == top_indices[0] else None 
            })

        highest_prediction = top_2_predictions[0]['class']
        highest_confidence = top_2_predictions[0]['confidence']
        
        CONFIDENCE_THRESHOLD = 10.0 
        if highest_confidence < CONFIDENCE_THRESHOLD:
            final_prediction = "Undetected"
            final_description = "The accuracy is below the confidence threshold (10%). A lesion couldn't be detected. Please upload a clearer image or consult a dermatologist for accurate diagnosis."
        else:
            final_prediction = highest_prediction
            final_description = top_2_predictions[0]['description']
        
        return jsonify({
            "final_prediction": final_prediction,
            "final_confidence": highest_confidence,
            "description": final_description,
            "top_2_predictions": top_2_predictions, 
            "all_probabilities": {label: float(prob) for label, prob in zip(label_encoder, probabilities)}
        })

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred during prediction: {str(e)}", "type": type(e).__name__}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000) 