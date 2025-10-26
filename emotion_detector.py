import cv2
from deepface import DeepFace
import time

cap = cv2.VideoCapture(0)

def get_classroom_emotion():
    """
    Captures video for 5 seconds, finds the dominant emotion, and returns it.
    This version is "headless" and does NOT display a window.
    """
    emotions_detected = []
    start_time = time.time()
    
   
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            continue
            
        try:
            
            result = DeepFace.analyze(frame, 
                                     actions=['emotion'], 
                                     enforce_detection=False, 
                                     silent=True)
            
            emotion = result[0]['dominant_emotion']
            emotions_detected.append(emotion)
            
        except Exception:
           
            pass

   
    if emotions_detected:
        dominant_emotion = max(set(emotions_detected), key=emotions_detected.count)
        return dominant_emotion
        
    return "neutral" # Default if no faces were detected

if __name__ == "__main__":
    print("Starting emotion detection test for 5 seconds...")
    emotion = get_classroom_emotion()
    print(f"Dominant classroom emotion: {emotion}")
   
    cap.release()
