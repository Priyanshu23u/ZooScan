from flask import Flask, render_template, request
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
import os
from tensorflow.keras import Input

app = Flask(__name__)

# === Rebuild model architecture ===
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 2. Add custom head exactly like you used during training
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', name='dense')(x)
predictions = Dense(3, activation='softmax', name='dense_1')(x)

# 3. Rebuild the full model
model = Model(inputs=base_model.input, outputs=predictions)

# 4. Then load the weights
model.load_weights("model/model_vgg16_gap2d.weights.h5")
# === Classes (adjust to your actual order) ===
CLASS_NAMES = ['Cat', 'Dog', 'Snake']

def prepare_image(img):
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file.stream).convert('RGB')
            processed = prepare_image(image)
            preds = model.predict(processed)
            predicted_class = CLASS_NAMES[np.argmax(preds)]
            prediction = f"Prediction: {predicted_class}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
