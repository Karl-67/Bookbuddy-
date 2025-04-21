import os
import openai
from dotenv import load_dotenv

# ✅ Load the .env file
load_dotenv()

# ✅ Read the OpenAI API key from the file path specified in .env
def get_openai_key():
    try:
        with open(os.getenv("OPEN_AI_CREDENTIALS")) as f:
            return f.read().strip()
    except Exception as e:
        print("❌ Failed to load OpenAI key:", e)
        return None


def simplify_text(text: str) -> str:
    """
    Simplify the given text using an LLM.

    Args:
        text (str): The input book text to be simplified.

    Returns:
        str: A simplified version of the input text.
    """
    prompt = f"Simplify the following text so it's easier to understand:\n\n{text}\n\nSimplified version:"

    api_key = get_openai_key()
    if not api_key:
        return "❌ OpenAI API key not found."

    try:
        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that simplifies complex text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        simplified_text = response.choices[0].message.content.strip()
        return simplified_text

    except Exception as e:
        print(f"❌ Error simplifying text:\n{e}")
        return "Sorry, I couldn't simplify the text at the moment."


# ✅ Test block for standalone runs
if __name__ == "__main__":
    input_text = (
        "Whan that Aprill with his shoures soote the droghte of March hath perced to the roote..."
    )

    print("📥 Original Text:\n", input_text)

    simplified = simplify_text(input_text)

    print("\n✅ Simplified Text:\n", simplified)
