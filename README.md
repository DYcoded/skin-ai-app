# DetectDisease: AI-Powered Skin Disease Detection Application

DetectDisease is an innovative web application designed to assist in the early detection of seven different types of skin lesions using advanced Artificial Intelligence (AI) technology. Our mission is to empower individuals and healthcare professionals with accurate and timely information to promote better skin health.

## Live Demo

* Experience the application live: [LIVE DEMO](https://detect-disease.vercel.app/)
---

## Important Notice (Medical Disclaimer)

**This application is NOT a medical diagnosis tool.** It is developed solely for awareness and educational purposes. AI predictions can never replace a consultation with a doctor. **Always consult a dermatologist for accurate results.**

## ✨ Features

* **7 Disease Detection:** Capable of identifying seven major types of skin lesions, including melanoma.
* **Deep Learning Model:** Utilizes a ResNet50V2-based model trained on thousands of dermatoscopic images.
* **Instant Results:** Provides the predicted disease name, confidence percentage, and a brief description for uploaded images.
* **Responsive Design:** Ensures an accessible and user-friendly experience across all devices (desktop, tablet, mobile).
* **Side Menu Navigation:** Features a side-sliding (sidebar) hamburger menu for easy navigation on mobile devices.

##  Technologies

### Frontend
* **HTML5**
* **CSS3 (Flexbox/Grid, Responsive Media Queries)**
* **JavaScript (For DOM manipulation and API interaction)**

### Backend & AI Service
* **Python / Flask:** Used as the Web API service.
* **TensorFlow / Keras:** Used for loading the deep learning model and making predictions.
    * **Model Architecture:** ResNet50V2
* **PIL (Pillow):** For image processing and resizing.
* **Numpy:** For data manipulation.
* **CORS:** To allow cross-origin access (enabling the API to be called from the web application).

##  Detected Conditions

The 7 main types of skin lesions the application can detect, along with their descriptions (Source: `testapi.py`):

| Abbreviation | Full Name | Description |
| :--- | :--- | :--- |
| **akiec** | Actinic Keratoses | Precancerous lesions caused by sun exposure. May develop into squamous cell carcinoma. |
| **bcc** | Basal Cell Carcinoma | Slow-growing cancer, rarely spreads. Usually appears as a pearly bump. |
| **bkl** | Benign Keratosis | Non-cancerous skin growths. Generally harmless but should be monitored for changes. |
| **df** | Dermatofibroma | Benign fibrous skin nodule. Usually harmless but can be removed if bothersome. |
| **mel** | Melanoma | Dangerous and aggressive skin cancer. Early detection is crucial. |
| **nv** | Melanocytic Nevi | Common mole, usually benign. Monitor for changes in size, color, or shape. |
| **vasc** | Vascular Lesion | Benign blood vessel growths. Usually harmless but should be monitored. |

## ⚙️ Setup and Running

### A. Prerequisites

* Python 3.x
* pip (Python package manager)
* Git (optional, for cloning the repository)

### B. Backend (API) Setup

1.  Clone the repository and navigate to the project directory:
    ```bash
    git clone [REPO_URL]
    cd DetectDisease
    ```
2.  Install the required Python packages:
    ```bash
    pip install Flask tensorflow numpy Pillow flask-cors
    ```
3.  Obtain the model weights and class labels. (Files: `final_resnet_oversampled_weights.weights.h5` and `class_indices.json`.) Ensure these files are in the same directory as `testapi.py`.
4.  Run the Flask application:
    ```bash
    python testapi.py
    ```
    The API will run on `http://0.0.0.0:5000` by default.

### C. Frontend (Web Application) Running

The frontend consists of static HTML, CSS, and JavaScript files.

1.  While the API is running, open the `index.html` file using any modern web browser.
2.  Alternatively, serve the files using a local HTTP server (e.g., `python -m http.server`).

