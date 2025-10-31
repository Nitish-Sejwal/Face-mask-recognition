from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model("face_mask_detection.keras")


# {'mask_weared_incorrect': 0, 'with_mask': 1, 'without_mask': 2}
CLASS_NAMES = ['mask_weared_incorrect', 'with_mask', 'without_mask']

IMG_SIZE = 224

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert BGR to RGB 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Crop
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]

    else:
        face_img = img

    # Resize
    resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)

    # Predict
    prediction = model.predict(input_tensor)[0]
    idx = np.argmax(prediction)
    label = f"{CLASS_NAMES[idx]}: {prediction[idx]:.2f}"

    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(debug=True)
