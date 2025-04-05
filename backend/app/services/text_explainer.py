import openai

# Insert your API key directly here for testing (temporary)
API_KEY = "sk-"  # Replace with your actual API key

def simplify_text(text: str) -> str:
    """
    Simplify the given text using an LLM.

    Args:
        text (str): The input book text to be simplified.

    Returns:
        str: A simplified version of the input text.
    """
    prompt = f"Simplify the following text so it's easier to understand:\n\n{text}\n\nSimplified version:"

    try:

        client = openai.OpenAI(api_key=API_KEY)

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

# Test block for standalone runs 
if __name__ == "__main__":

    input_text = (
        "Whan that Aprill with his shoures soote the droghte of March hath perced to the roote..."
    )

    print("📥 Original Text:\n", input_text)

    simplified = simplify_text(input_text)

    print("\n✅ Simplified Text:\n", simplified)
