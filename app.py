from flask import Flask, request, jsonify, Response
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import os
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

session = ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
print("Loaded model")

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
ALLOWED_ORIGIN    = 'https://ndellamaria.github.io'

def process_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image, dtype=np.float32)
        image = np.expand_dims(image, axis=0)
        image = image[..., ::-1]
        image -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
        return image
    except Exception as e:
        print("Error processing image:", e)
        raise

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    try:
        image_bytes = request.files['image'].read()
        image = process_image(image_bytes)
        predictions = session.run(None, {input_name: image})[0]
        class_names = ['good', 'over_exposed', 'blurry', 'under_exposed', 'light_exposure']
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        return jsonify({"class": predicted_class, "confidence": confidence})
    except Exception as e:
        print("Error predicting image:", e)
        return jsonify({"error": "Error predicting image"}), 500

@app.route('/anthropic/messages', methods=['POST'])
def proxy_anthropic():
    origin = request.headers.get('Origin', '')
    print(f"Proxy request from origin: '{origin}'")
    if ALLOWED_ORIGIN not in origin and 'localhost' not in origin:
        print(f"Blocked origin: '{origin}'")
        return jsonify({'error': 'Forbidden'}), 403
    if not ANTHROPIC_API_KEY:
        return jsonify({'error': 'API key not configured on server'}), 500
    try:
        resp = requests.post(
            'https://api.anthropic.com/v1/messages',
            json=request.json,
            headers={
                'Content-Type': 'application/json',
                'x-api-key': ANTHROPIC_API_KEY,
                'anthropic-version': '2023-06-01',
            },
            timeout=60,
        )
        return Response(resp.content, status=resp.status_code, mimetype='application/json')
    except Exception as e:
        print("Proxy error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "Healthy"}), 200

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Test successful'}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
