# facial_emotion.py
import cv2
import time
from deepface import DeepFace

def get_facial_emotion():
    """Capture frames for a duration and return detected facial emotion."""
    print("\nAccessing webcam... 📸")
    # Open a connection to the default webcam (index 0)
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return "Webcam error"

    # Set a start time for the loop
    start_time = time.time()
    duration = 5  # Capture frames for up to 5 seconds

    result = "No face detected"
    print("Looking for a face for 5 seconds...")

    try:
        while True:
            # Capture a single frame
            ret, frame = cap.read()
            
            # Check if the frame was captured successfully
            if not ret:
                print("Error: Could not read frame from webcam.")
                result = "Capture error"
                break
                
            try:
                # Try to analyze the frame for an emotion
                # We set enforce_detection=True so it raises a ValueError if no face is found
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True)
                
                # If a face is found, get the dominant emotion and break the loop
                result = analysis[0]['dominant_emotion']
                print("Face detected successfully! 👍")
                break
            except ValueError:
                # No face was found in the current frame, continue to the next frame
                pass
            
            # Check if the time limit has been reached
            if time.time() - start_time > duration:
                print("Time limit reached. No face detected.")
                break

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        result = "Analysis error"

    finally:
        # Always release the webcam
        cap.release()
        print("Webcam released.")
        
    return result


if __name__ == "__main__":
    print("\n--- Live Facial Emotion Recognition ---")
    print("The program will capture a single photo from your webcam to analyze.")
    
    while True:
        emotion = get_facial_emotion()
        
        # Print the final result in a formatted way
        print("\n" + "-"*35)
        print(f"Detected Emotion: {emotion.upper()}")
        print("-"*35)
        
        # Ask the user for input to continue or quit
        if input("Press Enter to try again or 'q' to quit: ").lower() == 'q':
            break

    print("Exiting program. Goodbye! 👋")