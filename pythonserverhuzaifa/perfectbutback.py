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
    
    # Invert the edges to get the object as white and background as black
    edges = cv2.bitwise_not(edges)
    
    # Create a mask of the edges
    mask = np.zeros_like(edges)
    mask[edges != 0] = 255
    
    # Apply the mask to the original image to make the background white
    result = np.zeros_like(src)
    result[mask == 0] = 255  # Set background to white
    result[mask != 0] = src[mask != 0]  # Keep the object as it is
    
    return result


@app.route('/api/process-image', methods=['POST'])
def process_image():
    image_file = request.files['image']
    temp_image_path = 'temp_image.png'
    image_file.save(temp_image_path)

    try:
        # Preprocess the image
        edges = preprocess_image(temp_image_path)

        # Encode the processed image to base64
        _, encoded_image = cv2.imencode('.png', edges)
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
