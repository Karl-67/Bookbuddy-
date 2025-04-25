from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from services.text_explainer import simplify_text
from services.text_to_speech import synthesize_speech
from services.speech_to_text import transcribe_audio  # Only if you're using it

app = Flask(__name__)
CORS(app)  # Allow requests from your frontend (React)

@app.route("/")
def hello():
    return "✅ Backend is up and running!"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    # If you're already using this and it's working, leave it as is.
    result = transcribe_audio()
    return jsonify(result)

@app.route("/tts", methods=["POST"])
def tts_from_openai():
    data = request.get_json()
    original_text = data.get("text", "")

    # ✅ Step 1: Simplify the input using OpenAI
    simplified_text = simplify_text(original_text)
    if simplified_text.startswith("❌"):
        return jsonify({"error": simplified_text}), 400

    # ✅ Step 2: Use Google TTS to convert simplified response
    output_path = "ai_response.mp3"
    audio_path = synthesize_speech(simplified_text, output_path)

    if audio_path.startswith("❌"):
        return jsonify({"error": audio_path}), 500

    # ✅ Step 3: Return audio file to frontend
    return send_file(audio_path, mimetype="audio/mpeg")

if __name__ == "__main__":
    app.run(debug=True, port=8000)
