import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import base64
import cv2
import mediapipe as mp

app = Flask(__name__)
CORS(app)  # Allow requests from frontend (e.g., React)

# Load model and setup
model = load_model('sign_language_model.h5')
actions = np.array(['hi', 'i love you', 'thank you'])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Hand detection
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            # Ensure consistent input shape: (30, 63)
            sequence = np.array([keypoints] * 30)  # 30 identical frames
            input_data = np.expand_dims(sequence, axis=0)
            input_data = input_data / np.max(input_data + 1e-6)  # Avoid divide-by-zero

            prediction = model.predict(input_data)[0]
            confidence = float(np.max(prediction))
            label = actions[np.argmax(prediction)]

            return jsonify({
                'prediction': label,
                'confidence': round(confidence, 2)
            }), 200
        else:
            return jsonify({'prediction': 'No hand detected', 'confidence': 0.0}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


