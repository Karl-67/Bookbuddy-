import os
from dotenv import load_dotenv
from google.cloud import texttospeech
from google.auth.exceptions import DefaultCredentialsError
import pygame  # ✅ Replacement for playsound

# ✅ Load the .env file
load_dotenv()

def get_credentials():
    """Get Google Cloud credentials path from environment or .env file"""
    # Try the Text-to-Speech specific credentials first
    credentials_path = os.getenv("Text_to_Speech")
    
    # If not set, try the general Google credentials
    if not credentials_path:
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    if not credentials_path:
        raise ValueError("❌ Missing Google Cloud credentials. Please set Text_to_Speech in .env")
    
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"❌ Credentials file not found at: {credentials_path}")
    
    print(f"🔍 Using Text-to-Speech credentials from: {credentials_path}")
    return credentials_path

def synthesize_speech(text: str, output_path="output.mp3") -> str:
    """
    Converts text to speech using Google Cloud TTS and saves to an MP3 file.

    Args:
        text (str): The input text to convert to speech.
        output_path (str): Path to save the MP3 audio file.

    Returns:
        str: The path to the generated audio file or an error message.
    """
    if not text.strip():
        return "❌ No text provided for speech synthesis."

    try:
        # Get and verify credentials
        credentials_path = get_credentials()
        
        # Set the credentials for this specific call
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        # ✅ Initialize the Text-to-Speech client
        client = texttospeech.TextToSpeechClient()

        # ✅ Set up the text input
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # ✅ Select voice configuration
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )

        # ✅ Configure the audio output
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        # ✅ Perform the text-to-speech request
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # ✅ Save the audio file
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
            print(f"🔊 Audio saved to: {os.path.abspath(output_path)}")

        return output_path

    except DefaultCredentialsError as e:
        print(f"❌ Google Cloud credentials error: {e}")
        return "❌ Failed to authenticate with Google Cloud. Please check your credentials."
    except Exception as e:
        print(f"❌ Error during TTS: {str(e)}")
        return f"❌ Failed to synthesize speech: {str(e)}"
    finally:
        # Reset the credentials to avoid affecting other services
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

# ✅ Test block for standalone runs
if __name__ == "__main__":
    try:
        sample_text = (
            "BookBuddy simplifies complex paragraphs from your textbook. "
            "Also stay hydrated with Schweppes!"
        )

        print("📝 Input Text:")
        print(sample_text)

        audio_file = synthesize_speech(sample_text)
        print("✅ Output File:", audio_file)

        # ✅ Play the audio using pygame only if called directly
        if os.path.exists(audio_file):
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            except Exception as e:
                print(f"❌ Error playing audio: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
