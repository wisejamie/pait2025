def build_question_prompt(section_text, section_title, num_questions, learning_objectives):
    objectives_block = "\n".join(f"- {obj}" for obj in learning_objectives) or "Not specified"

    return f"""
        Generate **exactly** {num_questions} multiple-choice questions based on the following section.

        Requirements for each question object:
        - question_text: The question prompt (string).
        - options: Exactly 4 distinct answer choices (array of strings).
        - correct_index: Integer 0–3 indicating the correct choice.
        - explanation: One-sentence rationale for the correct answer.
        - difficulty_score: A number from 0.0 (very easy) to 1.0 (very hard).
        - concept_tags: A list of 1–3 tags naming the key topics tested.
        - salience: A number from 0.0 (peripheral) to 1.0 (core concept).
        - directness: A number from 0.0 (requires inference) to 1.0 (stated literally).

        Output format must be **exactly** a JSON array of objects that looks like this:

        [
        {{
            "question_text": "…",
            "options": ["…", "…", "…", "…"],
            "correct_index": 2,
            "explanation": "…",
            "difficulty_score": 0.47,
            "concept_tags": ["topic1", "topic2"],
            "salience": 0.85,
            "directness": 0.30
        }},
        … {num_questions} total …
        ]

        Do **not** wrap the array in an outer object or include any commentary.

        Section title:
        \"{section_title}\"

        Learning objectives:
        {objectives_block}

        Section text:
        \"\"\"
        {section_text}
        \"\"\"
        """
    # objectives_block = "\n".join(f"- {obj}" for obj in learning_objectives)

    # return f"""
    #     Generate {num_questions} multiple choice questions based on the following text.

    #     Each question should:
    #     - Be answerable using only the section text
    #     - Include 4 distinct answer options
    #     - Identify the correct option using an integer index (0–3)
    #     - Provide a one-sentence explanation of the answer
    #     - Include a difficulty (easy, medium, hard)
    #     - Include a skill (recall, comprehension, application, analysis)
    #     - Include 1–3 topic tags

    #     Return only a **valid JSON list** of question objects. Do **not** wrap the list inside a dictionary.
    #     Your output should look exactly like this:

    #     [
    #     {{
    #         "question_text": "...",
    #         "options": ["...", "...", "...", "..."],
    #         "correct_index": 1,
    #         "explanation": "...,
    #         "difficulty": "medium",
    #         "skill": "comprehension",
    #         "tags": ["topic 1", "topic 2"]
    #     }},
    #     ...
    #     ]

    #     Section title: "{section_title}"

    #     Learning objectives:
    #     {objectives_block if learning_objectives else 'None'}

    #     Section text:
    #     \"\"\"
    #     {section_text}
    #     \"\"\"
    #     Only return the raw JSON list, without any additional explanation or wrapping object.
    #     """
