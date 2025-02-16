from flask import Flask, render_template, Response, jsonify, request
import cv2
import base64
import requests

app = Flask(__name__)

# Initialize the face detector
detector = cv2.FaceDetectorYN.create("model.onnx", "", (300, 300))
capture = cv2.VideoCapture(0)

GEMINI_API_KEY = "AIzaSyDjQoPh0DUe9we7qlMmb1JSYWUiz_vBilQ"

# Route to render the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Route for live video streaming
def generate_video():
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to capture frame and analyze emotion using Gemini API
@app.route('/analyze_emotion')
def analyze_emotion():
    ret, frame = capture.read()
    if not ret:
        return jsonify({"error": "Could not capture frame"}), 500

    _, buffer = cv2.imencode('.jpg', frame)
    b64 = base64.b64encode(buffer).decode('utf-8')

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": """Analyze the facial expression in the image and respond with only one word: 
    'happy', 'sad', 'angry', or 'neutral'. Do not include any additional text or explanations."""},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64
                        }
                    }
                ]
            }
        ]
    }

    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        result = response.json()
        emotion = result['candidates'][0]['content']['parts'][0]['text'].lower().strip()
        return jsonify({"emotion": emotion})
    else:
        return jsonify({"error": f"Error {response.status_code}: {response.text}"}), 500

def get_therapy_question(emotion):
    """Generate appropriate therapy questions based on detected emotion"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    emotion_prompts = {
        'happy': "You are a supportive therapist. The user is feeling happy. Ask them a gentle, encouraging question to explore their happiness. Keep it to one short sentence. Make it conversational and friendly.",
        'sad': "You are a supportive therapist. The user is feeling sad. Ask them a compassionate question to understand their feelings better. Keep it to one short sentence. Be gentle and caring.",
        'angry': "You are a supportive therapist. The user is feeling angry. Ask them a calming question to help them process their anger. Keep it to one short sentence. Be understanding and non-judgmental.",
        'neutral': "You are a supportive therapist. The user seems neutral. Ask them a friendly question about their day or feelings. Keep it to one short sentence. Be casual and approachable."
    }
    
    payload = {
        "contents": [{
            "parts": [{"text": emotion_prompts.get(emotion.lower(), emotion_prompts['neutral'])}]
        }]
    }
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        question = response.json()['candidates'][0]['content']['parts'][0]['text']
        return question
    return "How are you feeling right now?"

def get_therapy_response(user_response, emotion):
    """Generate therapeutic response based on user's answer"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    prompt = f"""You are a supportive therapist. The user is feeling {emotion} and said: "{user_response}"
    Provide a brief, empathetic response (2-3 sentences max) that:
    1. Acknowledges their feelings
    2. Offers support or gentle guidance
    Keep it conversational and authentic. Avoid clichés and generic advice."""
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        therapy_response = response.json()['candidates'][0]['content']['parts'][0]['text']
        return therapy_response
    return "I understand. Would you like to tell me more about that?"

@app.route('/get_therapy_question/<emotion>')
def therapy_question(emotion):
    question = get_therapy_question(emotion)
    return jsonify({"question": question})

@app.route('/send_therapy_response', methods=['POST'])
def therapy_response():
    data = request.json
    user_response = data.get('response')
    emotion = data.get('emotion')
    therapy_reply = get_therapy_response(user_response, emotion)
    return jsonify({"response": therapy_reply})

if __name__ == '__main__':
    app.run(debug=True, port=5500)