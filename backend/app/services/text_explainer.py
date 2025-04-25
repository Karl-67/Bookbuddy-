import os
import openai
import pygame
from dotenv import load_dotenv
from pathlib import Path
from google.cloud import texttospeech

# ✅ Load .env from project root
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)

# ✅ Load OpenAI API key
def get_openai_key():
    try:
        with open(os.getenv("OPEN_AI_CREDENTIALS")) as f:
            return f.read().strip()
    except Exception as e:
        print("❌ Failed to load OpenAI key:", e)
        return None

# ✅ Simplify the input using OpenAI and return the response
def simplify_text(text: str) -> str:
    print(f"🧾 Simplify Input: '{text}'")

    if not text or not text.strip():
        return "❌ No speech detected."

    prompt = (
        "You are a helpful assistant. "
        "If the input is a question, answer it clearly. "
        "If it's a complex sentence, simplify it for easier understanding.\n\n"
        f"Text: {text}\n\nResponse:"
    )

    api_key = get_openai_key()
    if not api_key:
        return "❌ OpenAI API key not found."

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You simplify or clarify any input the user gives."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        result = response.choices[0].message.content.strip()
        print(f"✅ Simplified: {result}")
        return result

    except Exception as e:
        print(f"❌ Error simplifying text:\n{e}")
        return "❌ Could not simplify. Try again."

# ✅ Google TTS synth + playback
def speak_text(text: str, output_path="simplified.mp3") -> str:
    if not text.strip():
        return "❌ Empty text."

    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("Text_to_Speech")
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        with open(output_path, "wb") as out:
            out.write(response.audio_content)
            print(f"🔊 Audio saved to: {os.path.abspath(output_path)}")

        # ✅ Play using pygame
        pygame.mixer.init()
        pygame.mixer.music.load(output_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        return output_path

    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return "❌ Failed to synthesize speech."

