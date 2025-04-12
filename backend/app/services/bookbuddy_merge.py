from speech_to_text import live_transcribe
from text_explainer import simplify_text

def main():
    print("🎤 Speak a passage or phrase to simplify...")
    original_text = live_transcribe()

    if original_text.strip() == "":
        print("❌ No speech detected. Please try again.")
        return

    print("\n📥 Original Text:\n", original_text)

    simplified = simplify_text(original_text)

    print("\n✅ Simplified Text:\n", simplified)

if __name__ == "__main__":
    main()
