import os
import openai
from dotenv import load_dotenv
from pathlib import Path

# Explicitly load the .env file from the project root
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)

def get_openai_key():
    try:
        with open(os.getenv("OPEN_AI_CREDENTIALS")) as f:
            return f.read().strip()
    except Exception as e:
        print("❌ Failed to load OpenAI key:", e)
        return None

def simplify_text(text: str) -> str:
    print(f"🧾 Simplify Input: '{text}'")  # add this

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
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ Error simplifying text:\n{e}")
        return "❌ Could not simplify. Try again."
