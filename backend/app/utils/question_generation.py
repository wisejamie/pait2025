def build_question_prompt(section_text, section_title, num_questions, learning_objectives):
    objectives_block = "\n".join(f"- {obj}" for obj in learning_objectives)

    return f"""
        Generate {num_questions} multiple choice questions based on the following text.

        Each question should:
        - Be answerable using only the section text
        - Include 4 distinct answer options
        - Identify the correct option using an integer index (0–3)
        - Provide a one-sentence explanation of the answer
        - Include a difficulty (easy, medium, hard)
        - Include a skill (recall, comprehension, application, analysis)
        - Include 1–3 topic tags

        Return only a **valid JSON list** of question objects. Do **not** wrap the list inside a dictionary.
        Your output should look exactly like this:

        [
        {{
            "question_text": "...",
            "options": ["...", "...", "...", "..."],
            "correct_index": 1,
            "explanation": "...,
            "difficulty": "medium",
            "skill": "comprehension",
            "tags": ["topic 1", "topic 2"]
        }},
        ...
        ]

        Section title: "{section_title}"

        Learning objectives:
        {objectives_block if learning_objectives else 'None'}

        Section text:
        \"\"\"
        {section_text}
        \"\"\"
        Only return the raw JSON list, without any additional explanation or wrapping object.
        """
