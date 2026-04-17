from flask import Flask, request, jsonify, Response
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import os
import base64
import json
import time as time_module
import requests
from flask_cors import CORS
from runwayml import RunwayML

app = Flask(__name__)
CORS(app)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    return response

session = ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
print("Loaded model")

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
RUNWAY_API_KEY    = os.environ.get('RUNWAY_API_KEY', '')
GITHUB_TOKEN      = os.environ.get('GITHUB_TOKEN', '')
GITHUB_REPO       = 'ndellamaria/ndellamaria.github.io'
GITHUB_API        = 'https://api.github.com'

def gh():
    return {'Authorization': f'token {GITHUB_TOKEN}', 'Accept': 'application/vnd.github.v3+json'}

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

@app.route('/runway/generate', methods=['POST'])
def runway_generate():
    if not RUNWAY_API_KEY:
        return jsonify({'error': 'Runway API key not configured'}), 500
    try:
        data   = request.json
        client = RunwayML(api_key=RUNWAY_API_KEY)
        task   = client.image_to_video.create(
            model='gen3a_turbo',
            prompt_image=data.get('prompt_image'),
            prompt_text=data.get('prompt_text', ''),
        )
        return jsonify({'task_id': task.id})
    except Exception as e:
        print("Runway generate error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/runway/task/<task_id>', methods=['GET'])
def runway_task_status(task_id):
    if not RUNWAY_API_KEY:
        return jsonify({'error': 'Runway API key not configured'}), 500
    try:
        client = RunwayML(api_key=RUNWAY_API_KEY)
        task   = client.tasks.retrieve(task_id)
        result = {'status': task.status}
        if task.status == 'SUCCEEDED' and task.output:
            result['video_url'] = task.output[0]
        elif task.status == 'FAILED':
            result['error'] = getattr(task, 'failure_reason', 'Generation failed')
        return jsonify(result)
    except Exception as e:
        print("Runway task error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/github/add-photo', methods=['POST', 'OPTIONS'])
def github_add_photo():
    if request.method == 'OPTIONS':
        return '', 204
    if not GITHUB_TOKEN:
        return jsonify({'error': 'GitHub token not configured'}), 500
    try:
        data        = request.json
        photo_list  = data.get('photos', [])
        if not photo_list:
            return jsonify({'error': 'No photos provided'}), 400

        # 1. Get main branch SHA
        r = requests.get(f'{GITHUB_API}/repos/{GITHUB_REPO}/git/ref/heads/main', headers=gh())
        r.raise_for_status()
        main_sha = r.json()['object']['sha']

        # 2. Create a single branch for all photos
        first_safe = photo_list[0]['filename'].replace(' ', '-').rsplit('.', 1)[0][:30]
        branch = f'film-lab/{first_safe}-{int(time_module.time())}'
        r = requests.post(f'{GITHUB_API}/repos/{GITHUB_REPO}/git/refs', headers=gh(),
                          json={'ref': f'refs/heads/{branch}', 'sha': main_sha})
        r.raise_for_status()

        # 3. Upload each image to pics/, each video to videos/
        for p in photo_list:
            r = requests.put(
                f'{GITHUB_API}/repos/{GITHUB_REPO}/contents/pics/{p["filename"]}',
                headers=gh(),
                json={'message': f'Add {p["filename"]}', 'content': p['image_base64'], 'branch': branch}
            )
            r.raise_for_status()

            if p.get('video_base64') and p.get('video_filename'):
                r = requests.put(
                    f'{GITHUB_API}/repos/{GITHUB_REPO}/contents/videos/{p["video_filename"]}',
                    headers=gh(),
                    json={'message': f'Add video {p["video_filename"]}', 'content': p['video_base64'], 'branch': branch}
                )
                r.raise_for_status()

        # 4. Get portfolio-photos.json + SHA
        r = requests.get(f'{GITHUB_API}/repos/{GITHUB_REPO}/contents/portfolio-photos.json', headers=gh())
        r.raise_for_status()
        jf       = r.json()
        existing = json.loads(base64.b64decode(jf['content'].replace('\n', '')).decode('utf-8'))
        json_sha = jf['sha']

        # 5. Append all new entries
        for p in photo_list:
            meta = p.get('meta', {})
            existing.append({
                'filename': p['filename'],
                'alt':      meta.get('title', ''),
                'video':    p.get('video_filename'),
            })

        # 6. Update portfolio-photos.json in one commit
        updated = base64.b64encode(json.dumps(existing, indent=2).encode()).decode()
        r = requests.put(f'{GITHUB_API}/repos/{GITHUB_REPO}/contents/portfolio-photos.json', headers=gh(),
                         json={'message': f'Add {len(photo_list)} photo(s) to portfolio manifest',
                               'content': updated, 'sha': json_sha, 'branch': branch})
        r.raise_for_status()

        # 7. Open one PR describing all photos
        pr_title = (f'Add to portfolio: {photo_list[0]["meta"].get("title") or photo_list[0]["filename"]}'
                    if len(photo_list) == 1
                    else f'Add {len(photo_list)} photos to portfolio')

        body_sections = []
        for p in photo_list:
            meta  = p.get('meta', {})
            lines = [
                f'**{meta["title"]}**' if meta.get('title') else f'**{p["filename"]}**',
                '🎞 Animated'          if p.get('video_filename') else None,
            ]
            body_sections.append('\n'.join(l for l in lines if l))

        pr_body = '\n\n---\n\n'.join(body_sections) or 'New photos added via Film Lab.'
        r = requests.post(f'{GITHUB_API}/repos/{GITHUB_REPO}/pulls', headers=gh(),
                          json={'title': pr_title, 'body': pr_body, 'head': branch, 'base': 'main'})
        r.raise_for_status()
        pr = r.json()
        return jsonify({'pr_url': pr['html_url'], 'pr_number': pr['number'], 'branch': branch})

    except Exception as e:
        print('GitHub PR error:', e)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return '', 204
    return jsonify({"status": "Healthy"}), 200

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Test successful'}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
