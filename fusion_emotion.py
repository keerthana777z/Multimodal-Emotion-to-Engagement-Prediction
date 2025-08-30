import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
from deepface import DeepFace
import sounddevice as sd

# ------------------------------
# --- Speech Model Setup (CPU) ---
# ------------------------------
MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
device = torch.device("cpu")
print("Loading feature extractor...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
print("Loading model on CPU...")
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(device)
print("Model loaded successfully ✅")

# Simplified emotion mapping
emotion_mapping = {
    "angry": "angry",
    "calm": "neutral",
    "disgust": "angry",
    "fearful": "sad",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprised": "happy"
}

# ------------------------------
# --- Functions ---
# ------------------------------
def record_audio(duration=5, samplerate=16000):
    print(f"\n🎙️ Recording {duration}s audio...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("Audio recorded ✅")
    return np.squeeze(audio)

def get_speech_emotion(duration=5, topk=3):
    audio = record_audio(duration)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    # Top K predictions
    sorted_idx = np.argsort(probs)[::-1]
    top_predictions = [(model.config.id2label[i], probs[i]*100) for i in sorted_idx[:topk]]

    # Map the top one to simplified emotion
    best_label = top_predictions[0][0]
    mapped = emotion_mapping.get(best_label, "neutral")

    return mapped, top_predictions

def get_facial_emotion(duration=5):
    print("\n📸 Accessing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return "Webcam error"

    start_time = time.time()
    dominant_emotion = "No face detected"

    print("--- Live Camera Feed ---")
    print("Press 'q' to stop early.")
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']
            # Put emotion text on the live frame
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except:
            pass
        
        # Display the frame
        cv2.imshow("Live Facial Emotion", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released.")
    print("Facial capture done ✅")
    return dominant_emotion

def fuse_emotions(facial, speech):
    # Simple fusion: if facial is neutral, trust speech
    if facial.lower() in ["neutral", "no face detected"]:
        return speech
    return facial

# ------------------------------
# --- Main ---
# ------------------------------
if __name__ == "__main__":
    print("\n--- Observing for 5 seconds ---")
    facial_emotion = get_facial_emotion(duration=5)
    speech_emotion, top3_speech = get_speech_emotion(duration=5)
    fused = fuse_emotions(facial_emotion, speech_emotion)

    # Print results
    print("\n--- Final Results ---")
    print(f"Facial Emotion: {facial_emotion}")
    print(f"Speech Emotion: {speech_emotion}")
    print(f"Fused Emotion : {fused}")
    print("\nTop 3 Speech Predictions:")
    for i, (label, score) in enumerate(top3_speech, 1):
        print(f"{i}. {label} ({score:.1f}%)")
    print("------------------------------\n")