import sounddevice as sd
import numpy as np

print("Testing microphone access...")
try:
    samplerate = 44100
    duration = 3  # seconds
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("Recording successful!")
    print("Audio data shape:", audio.shape)
except Exception as e:
    print(f"An error occurred: {e}")
    print("Possible issues: microphone not connected, or permissions denied.")