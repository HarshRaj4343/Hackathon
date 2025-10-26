import sounddevice as sd
import numpy as np

def check_classroom_audio(duration=5):
   
    
    print("🎤 Listening for", duration, "seconds...")
    

    sample_rate = 44100
    recording = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1)
    sd.wait()
    

    volume = np.abs(recording).mean()
    
    print(f"Volume level: {volume}")
    
 
    if volume < 0.01:  # Very quiet
        return "silent", "⚠️ Dead silence - students may be confused"
    elif volume < 0.05:  # Normal quiet
        return "quiet", "✅ Normal classroom sound"
    else:  # Active discussion
        return "active", "💬 Active discussion happening"

if __name__ == "__main__":
    state, message = check_classroom_audio()
    print(message)
