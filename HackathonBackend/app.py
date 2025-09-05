from flask import Flask, request, jsonify
import google.generativeai as genai
import json
import re

app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key="AIzaSyAMr4L6QNFvgNjMnDRQoCyv7HX9uQChkT4")  # Replace YOUR_API_KEY with your actual API key
model = genai.GenerativeModel("gemini-1.5-flash")


def remove_lines_starting_with_triple_backtick(text):
    """Removes lines from a string that start with triple backticks (```).

    Args:
        text: The input string.

    Returns:
        The string with lines starting with triple backticks removed.
    """
    lines = text.splitlines()
    print(line for line in lines)
    filtered_lines = [line for line in lines if not line.startswith("```")]
    return "\n".join(filtered_lines)


@app.route('/generate_flashcards', methods=['POST'])
def generate_flashcards():
    """
    Generates flashcards from class notes.

    Expects a JSON payload with a 'notes' field containing the text of the class notes.
    Returns a JSON response with a 'flashcards' field containing an array of flashcard objects.
    """
    try:
        data = request.get_json()
        notes = data.get('notes')

        if not notes:
            return jsonify({'error': 'Missing "notes" field in request body'}), 400

        prompt = f"""
**Prompt:**
You are a flashcard generator. You will receive text containing class notes. Your task is to create flashcards with **one-word** answers based on the provided notes. Output the flashcards and answers in a structured JSON format that's easy to parse. Use the following format:
{{
  "flashcards": [
    {{
      "question": "Flashcard Question 1",
      "answer": "Flashcard Answer 1"
    }},
    {{
      "question": "Flashcard Question 2",
      "answer": "Flashcard Answer 2"
    }},
    // ... more flashcards
  ]
}}

**Here are the class notes**
{notes}

Prioritize creating flashcards that test key concepts, definitions, and important relationships within the notes. Avoid creating flashcards for trivial details. Aim for clear and concise questions. If a concept from the notes doesn't lend itself to a one-word answer, try to rephrase the question so that it can be answered with a single, concise word (e.g., a key term or concept). If the notes contain lists, create separate flashcards for each list item. If the notes contain code examples, create flashcards that test understanding of the code's functionality. If the notes contain sections with headers, organize the flashcards by section. If absolutely necessary, and a single word answer is not possible, a very short, concise phrase can be used as long as it is no more than 3 words and is absolutely essential to capture the core meaning. The strong preference is always for single-word answers.
"""
        response = model.generate_content(prompt)

        json_text = remove_lines_starting_with_triple_backtick(response.text)

        try:
            flashcards_data = json.loads(json_text)
            flashcards = flashcards_data['flashcards']
        except (json.JSONDecodeError, KeyError):
            return jsonify({'error': 'Failed to parse generated flashcards. Model output is not valid JSON'}), 500

        return jsonify({'flashcards': flashcards}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
