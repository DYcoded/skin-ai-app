# DetectDisease: AI-Powered Skin Lesion Classification
**DetectDisease** is a web-based application designed to provide preliminary screening for skin conditions using Deep Learning. The system analyzes user-uploaded images of skin lesions and predicts the likelihood of 7 different types of skin diseases, helping to raise awareness and encourage early diagnosis.

⚠️ This tool is for educational and informational purposes only. It is **not** a medical diagnostic device and does not replace professional medical advice.


> **Current State:** Under Active Development

This project is currently in the **prototyping phase** and operates on a local environment. We are actively refining the AI model architecture and optimizing the user interface.

**Deployment Roadmap:**
We are preparing to deploy the full-stack application for public access in the upcoming weeks:
* **Backend (API):** Will be deployed to **Render**.
* **Frontend (UI):** Will be deployed to **Vercel**.

*Until the deployment is finalized, please refer to the Installation & Setup section below to run the project locally on your machine.*


## Screenshots

| Home Page | Disease Information | Detection Page |
|:---:|:---:|:---:|
| ![Home Page](./images/home.png) | ![Diseases](./images/diseases.png) | ![Result](./images/detection.png) |

*(Note: Screenshots represent the current interface designed in Figma.)*


## Features

* **User-Friendly Interface:** Custom-designed, multi-page web application (Home, Detection, About, Team).
* **AI Analysis:** Powered by a Convolutional Neural Network (CNN) trained on the HAM10000 dataset.
* **Real-Time Prediction:** Instant classification of skin lesions with confidence scores.
* **7 Disease Classes:** Can identify Melanoma, Nevi, Dermatofibroma, and more.
* **Secure Backend:** Robust Flask API handling image processing and model inference.


## Tech Stack

### Frontend
* **HTML5 & CSS3:** Custom styling (no external heavy frameworks).
* **JavaScript:** Asynchronous API communication (AJAX/Fetch).
* **Figma:** Used for UI/UX prototyping.

### Backend
* **Python:** Core logic.
* **Flask:** Lightweight WSGI web application framework.
* **Flask-CORS:** Handling Cross-Origin Resource Sharing.

### Machine Learning
* **TensorFlow / Keras:** Model training and inference.
* **NumPy & Pillow:** Image preprocessing.
* **Dataset:** HAM10000 (Human Against Machine with 10,000 Training Images).

## Model Details (Current Status)

* **Architecture:** Convolutional Neural Network (CNN)
* **Input Resolution:** 96x96 pixels (Intermediate Prototype)
* **Training Accuracy:** ~96% (on validation set)
* **Status:** The model is currently in the optimization phase. We are working on scaling up the input resolution to 300x300 for higher real-world accuracy.


## Future Roadmap

- [ ] **High-Res Retraining:** Upgrade model input to 300x300px for better detail extraction.
- [ ] **Cloud Deployment:** Deploy Backend to Render and Frontend to Vercel.
- [ ] **Mobile Responsiveness:** Optimize CSS for mobile devices.
- [ ] **Advanced Augmentation:** Improve model robustness against lighting variations.


## The Team

This project was developed by students of **Istanbul Arel University, Computer Engineering Department**.

| Team Member | Role | Socials |
| :--- | :--- | :--- |
| **Yaren Çanakçıoğlu** | Full-Stack Development & Design | [LinkedIn](https://www.linkedin.com/in/yarencnkc/) |
| **Dila Demirhan** | Full-Stack Development & Design | [LinkedIn](https://www.linkedin.com/in/dilademrhn/) |
| **Berzat Babur** | Machine Learning | [LinkedIn](https://www.linkedin.com/in/berzat-babur-b61277235/) |



## Installation & Setup

Follow these steps to run the project locally:

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/detect-disease.git](https://github.com/yourusername/detect-disease.git)
cd detect-disease
```
2. Install Dependencies
Make sure you have Python installed. Then run:
```bash
pip install -r requirements.txt
```
3. Run the Backend Server
Start the Flask API:
```bash
python testapi.py
```
The server will start running at http://127.0.0.1:5000/
5. Launch the Frontend
Simply open the index.html file in your web browser.


