from flask import Flask, request, jsonify
import tensorflow as tf 
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Add this line

# load pre-trained model
def load_model(path): 
    try:
        model = tf.keras.models.load_model(path)
        print("Loaded model")
        return model
    except Exception as e:
        print("Error loading model: ", e)
        raise

model = load_model("model.keras")

def process_image(image_bytes): 
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((224, 224))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image
    except Exception as e:
        print("Error processing image: ", e)
        raise

@app.route('/predict', methods=['POST'])
def predict():
    print("Request received")
    print("Files in request:", request.files)
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        image_file = request.files['image']
        print("Image filename:", image_file.filename)
        image_bytes = image_file.read()
        image = process_image(image_bytes)
        predictions = model.predict(image)
        class_names = ['good', 'over_exposed', 'blurry', 'under_exposed', 'light_exposure']
        confidence = np.max(predictions[0])
        predicted_class = class_names[np.argmax(predictions[0])]
        return jsonify({"class": predicted_class, "confidence": float(confidence)})
    except Exception as e:
        print("Error predicting image: ", e)
        return jsonify({"error": "Error predicting image"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Healthy"}), 200 


@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Test successful'}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)






