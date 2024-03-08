import cv2
import numpy as np
import base64
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def preprocess_image(image_path):
    src = cv2.imread(image_path)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur and Canny edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)
    
    # Convert edges to uint8 data type
    edges = edges.astype(np.uint8)
    
    return edges

def adjust_opacity(image, opacity):
    return cv2.addWeighted(image, opacity / 100, np.zeros(image.shape, dtype=np.uint8), 0, 0)

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    brightness_matrix = np.ones(image.shape, dtype="uint8") * brightness
    brightness_contrast_image = cv2.add(image, brightness_matrix)
    return cv2.addWeighted(brightness_contrast_image, 1 + contrast / 100, np.zeros(image.shape, dtype=np.uint8), 0, 0)

def adjust_sharpness(image, sharpness):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel * sharpness)

@app.route('/api/process-image', methods=['POST'])
def process_image():
    image_file = request.files['image']
    properties = {
        'opacity': int(request.form.get('opacity', 100)),
        'brightness': int(request.form.get('brightness', 0)),
        'contrast': int(request.form.get('contrast', 0)),
        'sharpness': int(request.form.get('sharpness', 0))
    }
    temp_image_path = 'temp_image.png'
    image_file.save(temp_image_path)

    try:
        # Preprocess the image
        edges = preprocess_image(temp_image_path)

        # Apply properties
        edges_with_properties = adjust_opacity(edges, properties['opacity'])
        edges_with_properties = adjust_brightness_contrast(edges_with_properties, properties['brightness'], properties['contrast'])
        edges_with_properties = adjust_sharpness(edges_with_properties, properties['sharpness'])

        # Encode the processed image to base64
        _, encoded_image = cv2.imencode('.png', edges_with_properties)
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
