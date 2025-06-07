import ollama


def generate_response(prompt):
    try:
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,
                "top_p": 0.9,
            },
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response."
