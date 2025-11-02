import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import tensorflow as tf
import threading
import time

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
model = None
classes = None
model_loaded = False

def load_model_and_classes():
    """Load model and classes in background"""
    global model, classes, model_loaded
    try:
        print("[INFO] Loading model...")
        start_time = time.time()
        
        # Load with optimizations
        model = load_model('mask_detector.keras', compile=False)
        classes = np.load('classes.npy')
        
        # Warm up the model with a dummy prediction
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        model.predict(dummy_input, verbose=0)
        
        load_time = time.time() - start_time
        print(f"[INFO] Model loaded in {load_time:.2f} seconds. Classes: {list(classes)}")
        model_loaded = True
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")

def preprocess_image(image):
    """Optimized image preprocessing"""
    # Resize image
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype="float32")
    
    # Preprocess for MobileNetV2
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_mask(image):
    """Optimized prediction"""
    if not model_loaded:
        return None, None, None
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        start_time = time.time()
        predictions = model.predict(processed_image, verbose=0)
        prediction_time = time.time() - start_time
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = classes[predicted_class_idx]
        
        class_probabilities = {
            classes[i]: float(predictions[0][i]) for i in range(len(classes))
        }
        
        print(f"[INFO] Prediction took {prediction_time:.3f} seconds")
        return predicted_class, confidence, class_probabilities
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return None, None, None

# Load model at startup in a separate thread
@app.before_first_request
def initialize():
    """Initialize model before first request"""
    thread = threading.Thread(target=load_model_and_classes)
    thread.daemon = True
    thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model_loaded,
        'classes': list(classes) if classes else None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model still loading, please try again in a moment'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read image directly from memory (no file saving)
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'error': 'Could not decode image'}), 400
            
            # Make prediction
            predicted_class, confidence, class_probabilities = predict_mask(image)
            
            if predicted_class is None:
                return jsonify({'error': 'Could not process image'}), 400
            
            result = {
                'prediction': predicted_class,
                'confidence': round(confidence * 100, 2),
                'class_probabilities': class_probabilities,
                'prediction_time': 'fast'  # Add timing info
            }
            
            return jsonify(result)
        
        except Exception as e:
            print(f"[ERROR] Prediction endpoint error: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    # Start model loading immediately
    load_model_and_classes()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
