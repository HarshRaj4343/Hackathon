import speech_recognition as sr
import librosa
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import tempfile
import os

class SpeechAnalyzer:
    """Analyzes teacher's speech for pace and tone"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.sample_rate = 16000  # Changed to 16kHz for better compatibility
        
    def analyze_teacher_speech(self, duration=10):
        """
        Analyze teacher's speech for pace and tone
        
        Args:
            duration: How long to record in seconds (default 10)
            
        Returns:
            dict: Contains pace, tone, and nudge message
        """
        
        print(f"🎤 Recording teacher for {duration} seconds...")
        
        try:
            # Record audio
            audio_file = self._record_audio(duration)
            
            # Analyze speech pace (words per minute)
            pace_result = self._analyze_pace(audio_file)
            
            # Analyze tone (energy and pitch variation)
            tone_result = self._analyze_tone(audio_file)
            
            # Generate nudge based on analysis
            nudge = self._generate_speech_nudge(pace_result, tone_result)
            
            # Clean up temp file
            try:
                os.unlink(audio_file)
            except:
                pass
            
            return {
                'pace': pace_result['status'],
                'wpm': pace_result['wpm'],
                'tone': tone_result['status'],
                'energy': tone_result['energy'],
                'pitch_variation': tone_result['pitch_variation'],
                'nudge': nudge
            }
            
        except Exception as e:
            print(f"❌ Error analyzing speech: {e}")
            import traceback
            traceback.print_exc()
            return {
                'pace': 'unknown',
                'wpm': 0,
                'tone': 'unknown',
                'energy': 0,
                'pitch_variation': 0,
                'nudge': 'Could not analyze speech. Please check your microphone.'
            }
    
    def _record_audio(self, duration):
        """Record audio from microphone and save as WAV"""
        
        print("🔴 Recording... Speak naturally!")
        
        try:
            # Record audio
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            print("✅ Recording complete!")
            
            # Convert to int16 format (required for proper WAV)
            recording_int = np.int16(recording * 32767)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav', 
                delete=False,
                mode='wb'
            )
            temp_file.close()
            
            # Write WAV file
            wavfile.write(
                temp_file.name, 
                self.sample_rate, 
                recording_int
            )
            
            print(f"📁 Audio saved to: {temp_file.name}")
            
            return temp_file.name
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            raise
    
    def _analyze_pace(self, audio_file):
        """Analyze speech pace (words per minute)"""
        
        try:
            print("📝 Converting speech to text...")
            
            # Use speech_recognition to transcribe
            with sr.AudioFile(audio_file) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Record the audio
                audio = self.recognizer.record(source)
                
                # Transcribe
                text = self.recognizer.recognize_google(audio)
            
            print(f"📝 Transcribed: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            
            # Calculate words per minute
            words = text.split()
            word_count = len(words)
            
            # Get audio duration
            y, sample_rate = librosa.load(audio_file, sr=None)
            duration_minutes = len(y) / sample_rate / 60
            
            wpm = word_count / duration_minutes if duration_minutes > 0 else 0
            
            print(f"⏱️ Speech rate: {wpm:.0f} words per minute")
            print(f"📊 Word count: {word_count}, Duration: {duration_minutes:.2f} minutes")
            
            # Categorize pace
            if wpm > 180:
                status = 'too_fast'
            elif wpm > 160:
                status = 'fast'
            elif wpm < 100:
                status = 'too_slow'
            elif wpm < 120:
                status = 'slow'
            else:
                status = 'good'
            
            return {
                'status': status,
                'wpm': int(wpm),
                'word_count': word_count
            }
            
        except sr.UnknownValueError:
            print("⚠️ Could not understand audio - speech unclear or too quiet")
            print("💡 Try: Speaking louder, reducing background noise, or moving closer to mic")
            return {'status': 'unclear', 'wpm': 0, 'word_count': 0}
            
        except sr.RequestError as e:
            print(f"⚠️ Speech recognition service error: {e}")
            print("💡 Check your internet connection (Google Speech API requires internet)")
            return {'status': 'error', 'wpm': 0, 'word_count': 0}
            
        except Exception as e:
            print(f"⚠️ Unexpected error in pace analysis: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'wpm': 0, 'word_count': 0}
    
    def _analyze_tone(self, audio_file):
        """Analyze tone for energy and monotony"""
        
        try:
            print("🎵 Analyzing tone and energy...")
            
            # Load audio
            y, sample_rate = librosa.load(audio_file, sr=None)
            
            # 1. Energy Analysis (volume/intensity)
            rms = librosa.feature.rms(y=y)[0]
            energy = np.mean(rms)
            energy_variation = np.std(rms)
            
            # 2. Pitch Analysis (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sample_rate)
            
            # Extract pitch values (ignore zeros/silence)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:  # Ignore silence
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 0:
                mean_pitch = np.mean(pitch_values)
                std_pitch = np.std(pitch_values)
                pitch_variation = std_pitch / (mean_pitch + 1e-6)
            else:
                pitch_variation = 0
            
            print(f"🎵 Energy level: {energy:.4f}")
            print(f"🎶 Pitch variation: {pitch_variation:.4f}")
            
            # Categorize energy
            if energy < 0.02:
                energy_status = 'too_quiet'
            elif energy < 0.05:
                energy_status = 'quiet'
            elif energy > 0.15:
                energy_status = 'too_loud'
            else:
                energy_status = 'good'
            
            # Check for monotone (low pitch variation)
            if pitch_variation < 0.1:
                monotone = True
                tone_status = 'monotone'
            else:
                monotone = False
                tone_status = 'engaging'
            
            return {
                'status': tone_status,
                'energy': float(energy),
                'energy_status': energy_status,
                'pitch_variation': float(pitch_variation),
                'monotone': monotone
            }
            
        except Exception as e:
            print(f"⚠️ Tone analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'unknown',
                'energy': 0,
                'energy_status': 'unknown',
                'pitch_variation': 0,
                'monotone': False
            }
    
    def _generate_speech_nudge(self, pace_result, tone_result):
        """Generate specific nudge based on pace and tone analysis"""
        
        nudges = []
        
        # Pace-based nudges
        if pace_result['status'] == 'too_fast':
            nudges.append(f"🐇 You're speaking too fast ({pace_result['wpm']} WPM). Slow down to help students absorb the material.")
        elif pace_result['status'] == 'fast':
            nudges.append(f"⚡ Speaking a bit quickly ({pace_result['wpm']} WPM). Consider slowing down slightly.")
        elif pace_result['status'] == 'too_slow':
            nudges.append(f"🐢 Speaking quite slowly ({pace_result['wpm']} WPM). You can speed up a bit to maintain engagement.")
        elif pace_result['status'] == 'slow':
            nudges.append(f"😴 Pace is a bit slow ({pace_result['wpm']} WPM). Try adding more energy.")
        elif pace_result['status'] == 'unclear':
            nudges.append("⚠️ Could not analyze speech - please speak more clearly or check your microphone.")
        
        # Tone-based nudges
        if tone_result.get('monotone', False):
            nudges.append("📉 Your tone sounds monotonous. Try varying your pitch and enthusiasm!")
        
        if tone_result.get('energy_status') == 'too_quiet':
            nudges.append("🔇 You're speaking too quietly. Increase your volume or check your microphone.")
        elif tone_result.get('energy_status') == 'too_loud':
            nudges.append("🔊 Volume is quite high. Lower it slightly for comfort.")
        
        # If everything is good
        if not nudges:
            return "✅ Great speech pace and tone! Keep it up!"
        
        # Combine nudges
        return " | ".join(nudges)


def test_speech_analyzer():
    """Test function"""
    
    print("=" * 60)
    print("🎤 Speech Analyzer Test")
    print("=" * 60)
    print("\nThis will record you speaking for 10 seconds.")
    print("Try speaking at different speeds and tones!")
    print("\n💡 Tips:")
    print("  - Speak clearly and naturally")
    print("  - Try the test scripts provided")
    print("  - Make sure you have internet (for speech recognition)")
    print()
    
    input("Press Enter to start recording...")
    
    analyzer = SpeechAnalyzer()
    result = analyzer.analyze_teacher_speech(duration=10)
    
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print(f"Pace Status: {result['pace']}")
    print(f"Words Per Minute: {result['wpm']} WPM")
    print(f"Tone Status: {result['tone']}")
    print(f"Energy Level: {result['energy']:.4f}")
    print(f"Pitch Variation: {result['pitch_variation']:.4f}")
    print(f"\n💡 NUDGE:")
    print(f"{result['nudge']}")
    print("=" * 60)


if __name__ == "__main__":
    test_speech_analyzer()
