from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("brain_tumor_efficientnetb0.keras")

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image=preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file).convert('RGB')


    processed = preprocess_image(image)
    predictions = model.predict(processed)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index])
    print(predictions)
    return jsonify({
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2),
        "all_predictions": predictions.tolist()
    })
    


if __name__ == '__main__':
    app.run(debug=True)
