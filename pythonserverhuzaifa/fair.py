from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os  # Import the os module for file operations
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def preprocess_image(image_path):
    # Read the image using OpenCV
    src = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to improve object detection
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return thresholded

def detect_objects(image_path):
    # Load the pre-trained object detection model (replace this with your own model if available)
    # For example, you can use Haar cascades or deep learning-based object detection models
    # Here, we'll use a simple Haar cascade for face detection as an example
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read the image using OpenCV
    src = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return faces

@app.route('/api/process-image', methods=['POST'])
def process_image():
    # Get image file from request
    image_file = request.files['image']
    # Save the image to a temporary file
    temp_image_path = 'temp_image.png'
    image_file.save(temp_image_path)

    try:
        # Preprocess the image
        thresholded = preprocess_image(temp_image_path)

        # Encode the processed image to base64
        _, encoded_image = cv2.imencode('.png', thresholded)
        sketch_base64 = base64.b64encode(encoded_image).decode('utf-8')
        sketch_url = f'data:image/png;base64,{sketch_base64}'

        # Clean up temporary image file
        os.remove(temp_image_path)

        return jsonify({'sketchUrl': sketch_url}), 200
    except Exception as e:
        print('Error processing image:', e)
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
