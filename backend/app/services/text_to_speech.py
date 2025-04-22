import os
from dotenv import load_dotenv
from google.cloud import texttospeech

# ✅ Load the .env file
load_dotenv()

# ✅ Set the required env variable for Google TTS SDK from the .env
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_TEXT_CREDENTIALS")


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
            print(f"🔊 Audio saved to: {output_path}")

        return output_path

    except Exception as e:
        print(f"❌ Error during TTS: {e}")
        return "❌ Failed to synthesize speech."


# ✅ Test block for standalone runs
if __name__ == "__main__":
    sample_text = (
        "BookBuddy simplifies complex paragraphs from your textbook. "
        "This sentence is being converted into speech using Google's API."
    )

    print("📝 Input Text:")
    print(sample_text)

    audio_file = synthesize_speech(sample_text)
    print("✅ Output File:", audio_file)
