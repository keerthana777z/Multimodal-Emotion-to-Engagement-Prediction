# backend/api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import base64
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Phase 1.2: Rule-Based Chatbot ---
# This is the chatbot's response logic.
CHATBOT_RESPONSES = {
    "happy": ["That's great! Tell me more.", "Your joy is contagious!", "I'm glad to hear that."],
    "sad": ["I'm sorry to hear that. How can I help?", "It's okay to feel sad.", "I'm here for you."],
    "angry": ["Take a deep breath. Let's talk about it.", "I understand you're upset. What's on your mind?", "It's important to stay calm."],
    "neutral": ["What's on your mind?", "Is there anything you'd like to discuss?", "Okay, I'm listening."],
    "fearful": ["Everything is alright. Don't worry.", "It's going to be okay.", "I'm here to help you feel safe."]
}

@app.route('/chatbot_reply', methods=['POST'])
def chatbot_reply():
    data = request.json
    emotion = data.get('emotion', 'neutral')
    
    response_list = CHATBOT_RESPONSES.get(emotion, CHATBOT_RESPONSES["neutral"])
    reply = random.choice(response_list)
    
    return jsonify({"reply": reply})

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    try:
        data = request.json
        image_data_base64 = data.get('image_data')

        # --- Process Audio Data ---
        # For now, we'll use a placeholder since the speech emotion function
        # requires live recording. In a full implementation, you'd need to
        # modify the speech emotion function to accept audio data directly.
        speech_emotion_result = "neutral" # Placeholder for now

        # --- Process Image Data ---
        image_bytes = base64.b64decode(image_data_base64.split(',')[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        facial_emotion = get_facial_emotion_from_frame(frame)
        
        # --- Fusion Logic (simplified) ---
        fused_emotion = facial_emotion # Simplified for this demo

        # --- Get top 3 predictions from speech model (if needed) ---
        # This requires a more complex implementation on the backend.
        top3_speech = ["Not implemented", "Not implemented", "Not implemented"]
        
        return jsonify({
            "fused_emotion": fused_emotion,
            "top3_speech": top3_speech,
            "speech_emotion": speech_emotion_result,
            "facial_emotion": facial_emotion
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)

# Helper function to get facial emotion from a single frame
def get_facial_emotion_from_frame(frame):
    try:
        from deepface import DeepFace
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    except Exception:
        return "neutral"