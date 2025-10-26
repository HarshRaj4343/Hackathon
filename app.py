from flask import Flask, render_template, jsonify, request
from emotion_detector import get_classroom_emotion
from audio_detector import check_classroom_audio
from Speech_analyzer import SpeechAnalyzer
import threading
import time
import os
import google.generativeai as genai

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    api_key = "AIzaSyA2E3ZnzXPHHK6p58XZA3Bs8J-Exduu6ww" 

    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.0-pro')
    print("✅ Gemini API configured successfully.")
except Exception as e:
    print(f"⚠️ Gemini API key not found or invalid: {e}")
    model = None


app = Flask(__name__)


speech_analyzer = SpeechAnalyzer()


dashboard_data = {
    'engagement_score': 50,
    'emotion': 'neutral',
    'audio_state': 'quiet',
    'teacher_pace': 'unknown',
    'teacher_wpm': 0,
    'teacher_tone': 'unknown',
    'nudge': 'Click "Analyze My Teaching" to get started!',
    'speech_nudge': '',
    'timestamp': time.time(),
    'status': 'Ready',
    'analyzing_speech': False 
}

data_lock = threading.Lock()

def update_student_data():
    """Background thread that only updates STUDENT metrics (emotion + audio)"""
    global dashboard_data
    
    time.sleep(5)
    cycle = 0
    
    while True:
        try:
            with data_lock:
                if dashboard_data['analyzing_speech']:
                    time.sleep(5)
                    continue
            
            cycle += 1
            print(f"\n{'='*60}")
            print(f"🔄 Student analysis cycle {cycle}")
            print(f"{'='*60}")

            with data_lock:
                dashboard_data['status'] = 'Analyzing students...'

            print("📸 Analyzing student emotions...")
            emotion = get_classroom_emotion()
            print(f"✅ Emotion: {emotion}")
 
            print("🎧 Analyzing classroom audio...")
            audio_state, audio_message = check_classroom_audio(duration=3)
            print(f"✅ Audio: {audio_state}")
            

            score = 50  
            
    
            if emotion in ['happy', 'neutral', 'surprise']:
                score += 20
            elif emotion in ['sad', 'angry', 'fear']:
                score -= 20
            
          
            if audio_state == 'silent':
                score -= 20
            elif audio_state == 'active':
                score += 10
            
       
            with data_lock:
                teacher_pace = dashboard_data.get('teacher_pace', 'unknown')
                teacher_wpm = dashboard_data.get('teacher_wpm', 0)
                teacher_tone = dashboard_data.get('teacher_tone', 'unknown')
                speech_nudge = dashboard_data.get('speech_nudge', '')
        
            if teacher_pace == 'too_fast':
                score -= 15
            elif teacher_pace == 'too_slow':
                score -= 10
            elif teacher_pace == 'good':
                score += 5
            

            if teacher_tone == 'monotone':
                score -= 15
            elif teacher_tone == 'engaging':
                score += 10
            
            score = max(0, min(100, score))
            
 
            nudge = "✅ All good! Keep going."
            if score < 40:
                nudge = "⚠️ Low engagement! Try asking a question or showing an example."
            elif score < 60:
                nudge = "⚡ Engagement dropping. Consider a quick activity or recap."
            

            with data_lock:
                dashboard_data['engagement_score'] = score
                dashboard_data['emotion'] = emotion
                dashboard_data['audio_state'] = audio_state
                dashboard_data['nudge'] = nudge
                dashboard_data['timestamp'] = time.time()
                dashboard_data['status'] = 'Active'
            
            print(f"\n📊 Updated: Score={score}, Emotion={emotion}, Audio={audio_state}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"❌ Error in student analysis: {e}")
            import traceback
            traceback.print_exc()
            with data_lock:
                dashboard_data['status'] = f'Error: {str(e)}'
        
 
        time.sleep(10)

@app.route('/gpt-search', methods=['POST'])
def gpt_search():
    if not model:
        return jsonify({
            'error': 'Gemini API is not configured on the server. Please check the API key.'
        }), 500

    query = request.json.get('query')
    if not query:
        return jsonify({'error': 'No query was provided.'}), 400

    try:
        print(f"🔍 Sending query to Gemini: '{query}'")
        
       
        prompt = f"As a helpful teaching assistant, answer the following question clearly and concisely for a teacher. Question: {query}"
        
        response = model.generate_content(prompt)
        
        print("✅ Received response from Gemini.")
        return jsonify({'answer': response.text})

    except Exception as e:
        print(f"❌ An error occurred with the Gemini API: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/data')
def get_data():
    """Return current dashboard data"""
    with data_lock:
        return jsonify(dashboard_data)


@app.route('/analyze-speech', methods=['POST'])
def analyze_speech():
    """Triggered when teacher clicks the button"""
    
    global dashboard_data
    

    with data_lock:
        if dashboard_data['analyzing_speech']:
            return jsonify({
                'status': 'error',
                'message': 'Already analyzing speech. Please wait.'
            })
    
        dashboard_data['analyzing_speech'] = True
        dashboard_data['status'] = 'Get ready to speak...'
        dashboard_data['speech_nudge'] = 'Preparing to record...'

    def analyze():
        global dashboard_data
        
        try:
            print("\n" + "="*60)
            print("🎤 TEACHER SPEECH ANALYSIS STARTED")
            print("="*60)
            
      
            with data_lock:
                dashboard_data['status'] = 'Recording your speech...'
                dashboard_data['speech_nudge'] = '🔴 Recording for 10 seconds... Speak naturally!'
            
      
            time.sleep(2)
            
    
            result = speech_analyzer.analyze_teacher_speech(duration=10)
            
            print(f"✅ Analysis complete!")
            print(f"   Pace: {result['pace']} ({result['wpm']} WPM)")
            print(f"   Tone: {result['tone']}")
            print(f"   Nudge: {result['nudge']}")
            
         
            with data_lock:
                dashboard_data['teacher_pace'] = result['pace']
                dashboard_data['teacher_wpm'] = result['wpm']
                dashboard_data['teacher_tone'] = result['tone']
                dashboard_data['speech_nudge'] = result['nudge']
                dashboard_data['status'] = 'Analysis complete!'
                dashboard_data['analyzing_speech'] = False
            
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"❌ Speech analysis error: {e}")
            import traceback
            traceback.print_exc()
            
            with data_lock:
                dashboard_data['speech_nudge'] = f'Error analyzing speech: {str(e)}'
                dashboard_data['status'] = 'Error occurred'
                dashboard_data['analyzing_speech'] = False
    

    thread = threading.Thread(target=analyze, daemon=True)
    thread.start()
    
    return jsonify({
        'status': 'success',
        'message': 'Speech analysis started. Recording will begin in 2 seconds...'
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})




if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Shiksha-Pulse Dashboard")
    print("=" * 60)
    print("📊 Dashboard: http://localhost:5000")
    print("⚠️  Grant camera/mic permissions if prompted")
    print("🎤 Click 'Analyze My Teaching' button to analyze your speech")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    
    thread = threading.Thread(target=update_student_data, daemon=True)
    thread.start()
    

    app.run(debug=False, port=5000, threaded=True)
