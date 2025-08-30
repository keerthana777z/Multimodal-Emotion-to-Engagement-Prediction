# demo_speech_emotion.py (CPU-only, clean)
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # suppress tokenizer warnings

import sounddevice as sd
import torch
import numpy as np
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification

# CPU-only
device = torch.device("cpu")

MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

# Suppress HF warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

print("Loading feature extractor...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
print("Loading model on CPU...")
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(device)
print("Model loaded successfully ✅")

# Map 8 emotions to simplified 4 categories
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

def record_audio(duration=5, samplerate=16000):
    """Record audio and return numpy array"""
    print(f"\n🎙️ Recording {duration}s audio...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("Audio recorded ✅")
    return np.squeeze(audio)

def get_speech_emotion(duration=5, top_k=3):
    """Record audio and return top_k predictions and mapped emotion"""
    audio = record_audio(duration)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    # Top-k predictions
    sorted_idx = np.argsort(probs)[::-1][:top_k]
    top_preds = [(model.config.id2label[i], probs[i]*100) for i in sorted_idx]

    # Simplified final prediction
    best_label = top_preds[0][0]
    mapped = emotion_mapping[best_label]

    return top_preds, mapped
