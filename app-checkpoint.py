import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model with error handling
model_path = "final_model.h5"
try:
    generator = load_model(model_path)
    input_shape = generator.input_shape[1:3]  # (height, width)
    print(f"Model loaded. Input shape: {input_shape}")
except Exception as e:
    print(f"Error loading model: {e}")
    generator = None
    input_shape = (256, 256)

# Convert numpy image to base64
def image_to_base64(image):
    try:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        if image.ndim == 3 and image.shape[2] == 1:
            image = np.squeeze(image, axis=2)  # Convert (H, W, 1) → (H, W)
        img = Image.fromarray(image)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str
    except Exception as e:
        print(f"Base64 encoding error: {e}")
        return None


# Image preprocessing
def process_lr_image(img_file):
    try:
        img = Image.open(img_file)
        img = img.convert('RGB')  # Ensure 3 channels
        img = img.resize(input_shape)  # Resize to model input
        img_array = np.array(img).astype(np.float32) / 255.0
        return img_array
    except Exception as e:
        print(f"Image processing error: {e}")
        return None

# Route: home page
@app.route('/', methods=['GET', 'POST'])
def index():
    lr_image_base64 = None
    hr_image_base64 = None

    if request.method == 'POST':
        print("POST request received.")
        if not generator:
            print("Model not loaded.")
        uploaded_file = request.files.get('lr_image')
        if uploaded_file:
            print(f"File received: {uploaded_file.filename}")
            lr_image = process_lr_image(uploaded_file)
            if lr_image is not None:
                try:
                    print("Running prediction...")
                    hr_image = generator.predict(np.expand_dims(lr_image, axis=0))[0]
                    print("Prediction complete.")
                    hr_image_base64 = image_to_base64(hr_image)
                    lr_image_base64 = image_to_base64(lr_image)
                except Exception as e:
                    print(f"Prediction error: {e}")
            else:
                print("Image preprocessing failed.")
        else:
            print("No file uploaded.")

    return render_template('index.html', lr_image=lr_image_base64, hr_image=hr_image_base64)

if __name__ == '__main__':
    app.run(debug=True)
