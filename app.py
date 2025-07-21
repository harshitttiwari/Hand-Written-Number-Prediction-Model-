from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
import cv2
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DigitRecognition')

app = Flask(__name__, static_folder='static')
app.secret_key = os.urandom(24)  # Secure random key
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Model loading with enhanced error handling
try:
    logger.info("Attempting to load TFLite model...")
    if not os.path.exists("digit_model.tflite"):
        raise FileNotFoundError("digit_model.tflite not found in application directory")
    
    interpreter = tf.lite.Interpreter(model_path="digit_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    logger.info(f"Model loaded successfully. Input details: {input_details}")
    logger.info(f"Output details: {output_details}")

    # Validate model expectations
    expected_shape = tuple(input_details[0]['shape'])
    if expected_shape != (1, 784):
        logger.warning(f"Model expects shape {expected_shape}, but our preprocessing targets (1, 784)")

except Exception as e:
    logger.error(f"Critical model loading error: {str(e)}")
    raise SystemExit("Failed to initialize model - check logs for details")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_single_digit(image_path):
    """Enhanced validation with detailed diagnostics"""
    try:
        logger.info(f"Validating image: {image_path}")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error("Could not read image file")
            return False, "Invalid image file - please upload a valid PNG, JPG, or JPEG"
            
        # Check image quality
        if img.size == 0:
            logger.error("Empty image detected")
            return False, "Empty image detected"
            
        if img.mean() > 240:  # Mostly white
            logger.warning("Image appears mostly blank")
            return False, "Image appears blank - please ensure digit is visible"
            
        # Enhanced thresholding
        thresh = cv2.adaptiveThreshold(
            img, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Contour analysis
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = sorted(
            [c for c in contours if cv2.contourArea(c) > 50],
            key=cv2.contourArea, 
            reverse=True
        )
        
        if not valid_contours:
            logger.warning("No valid contours found")
            return False, "No digit detected - please ensure clear handwriting"
            
        if len(valid_contours) > 1:
            logger.warning(f"Multiple contours found: {len(valid_contours)}")
            return False, "Multiple digits detected - please upload single digit only"
            
        return True, "Validation passed"
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}", exc_info=True)
        return False, f"Image validation failed: {str(e)}"

def preprocess_image(image_path):
    """Robust preprocessing with detailed error handling"""
    try:
        logger.info(f"Preprocessing image: {image_path}")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("OpenCV could not read the image")
        
        # Debug original image
        debug_img = img.copy()
        
        # Resize with aspect ratio preservation
        h, w = img.shape
        if h != 28 or w != 28:
            scale = 28 / max(h, w)
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            delta_w = 28 - img.shape[1]
            delta_h = 28 - img.shape[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
        
        # Invert and normalize
        img = cv2.bitwise_not(img)
        img = img.astype(np.float32) / 255.0
        
        # Flatten for model
        img = img.reshape(1, 28 * 28)
        
        # Debug output
        logger.debug(f"Preprocessed image shape: {img.shape}")
        return img
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
        raise Exception(f"Could not process image: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate request
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('home'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('home'))
        
        if not allowed_file(file.filename):
            flash('Allowed file types: PNG, JPG, JPEG (Max 5MB)', 'error')
            return redirect(url_for('home'))
        
        # Secure file handling
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = secure_filename(file.filename)
        filename = f"{timestamp}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save file
        file.save(filepath)
        logger.info(f"File saved to: {filepath}")
        
        # Validate image
        is_valid, validation_msg = validate_single_digit(filepath)
        if not is_valid:
            os.remove(filepath)
            flash(validation_msg, 'error')
            return redirect(url_for('home'))
        
        # Preprocess
        img_array = preprocess_image(filepath)
        
        # Validate input shape
        if img_array.shape != tuple(input_details[0]['shape']):
            error_msg = f"Input shape mismatch. Expected {input_details[0]['shape']}, got {img_array.shape}"
            logger.error(error_msg)
            os.remove(filepath)
            flash("System error - please try a different image", 'error')
            return redirect(url_for('home'))
        
        # Make prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Process results
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        confidence_percent = f"{confidence*100:.1f}%"
        
        logger.info(f"Prediction: {predicted_digit} with confidence {confidence_percent}")
        
        # Quality control
        if confidence < 0.5:
            os.remove(filepath)
            flash(f'Unclear prediction (confidence: {confidence_percent}) - please try a clearer image', 'warning')
            return redirect(url_for('home'))
        
        return render_template('result.html',
                           digit=predicted_digit,
                           confidence=confidence_percent,
                           image_path=f"uploads/{filename}")
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        flash('An unexpected error occurred - please try again', 'error')
        return redirect(url_for('home'))

if __name__ == '__main__':
    # Create upload directory if not exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Start application
    logger.info("Starting Flask application")
    app.run(host='0.0.0.0', port=5000, debug=True)  