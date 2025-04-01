import base64
import io
from pydub import AudioSegment

def decode_base64_audio(base64_str: str) -> AudioSegment:
    audio_bytes = base64.b64decode(base64_str)
    audio_stream = io.BytesIO(audio_bytes)
    return AudioSegment.from_file(audio_stream)

def convert_to_linear16(audio: AudioSegment) -> bytes:
    """
    Converts audio to Google STT-compatible LINEAR16 format (WAV).
    """
    buffer = io.BytesIO()
    audio.set_frame_rate(16000).set_channels(1).export(buffer, format="wav")
    return buffer.getvalue()
