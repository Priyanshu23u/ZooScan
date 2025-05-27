from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model('model.h5')  # Your VGG16 model

# Ensure uploads folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Image size must match model input
IMG_SIZE = (224, 224)

# Modify these labels to match your model's output
labels = ['Cat', 'Dog', 'Snake']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, img.filename)
    img.save(filepath)

    image = load_img(filepath, target_size=IMG_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    predictions = model.predict(image)
    predicted_class = labels[np.argmax(predictions)]

    return jsonify({'prediction': predicted_class, 'image_url': filepath})

if __name__ == '__main__':
    app.run(debug=True)
