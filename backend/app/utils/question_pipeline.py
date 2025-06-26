import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

import openai

load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

# Initialize the OpenAI client
client = openai.OpenAI(api_key=api_key)

# -------------------------Question Building-------------------------
def build_question_plan_prompt(section_text: str, num_questions: int, local_learning_objectives: List, learning_objectives: Dict[str, str]) -> str:
    local_goals_formatted = "\n".join([f"- {obj}" for obj in local_learning_objectives])
    learning_goals_formatted = "\n".join([f"- {obj}" for obj in learning_objectives.values()])
    return f"""
You are an AI tutor preparing to generate multiple-choice questions from a complex academic text.

Your first task is to plan a diverse set of {num_questions} questions that assess different concepts from the section below.

Use the **section-specific learning objectives** to identify the most important local ideas to assess:
{local_goals_formatted}

Use the **global learning objectives** to guide question planning toward the broader purpose of the document as a whole — they are not mandates for coverage, but thematic anchors:
{learning_goals_formatted}

Prefer concepts that are:
- aligned with the section-specific objectives,
- relevant to one or more global objectives,
- and substantively present in the section text.

Each plan item should describe:
- concept: A concise summary (5–15 words) of the specific idea to be tested.
- difficulty_score: A float between 0.0 (very easy) and 1.0 (very hard).
- salience: A float between 0.0 (minor detail) and 1.0 (central idea).
- directness: A float between 0.0 (requires inference) and 1.0 (stated literally).

Ensure:
- No duplicate or overlapping concepts.
- A variety of salience and difficulty levels.
- Coverage of ideas that are relevant to the broader learning goals above.

Respond with a JSON array of exactly {num_questions} objects using this format:
[
  {{
    "concept": "...",
    "difficulty_score": ...,
    "salience": ...,
    "directness": ...
  }},
  ...
]

Section Text:
\"\"\"
{section_text}
\"\"\"
"""

from time import sleep

def generate_question_set(section_text: str, num_questions: int, local_learning_objectives: List, learning_objectives: Dict[str, str]) -> List[Dict[str, Any]]:
    # Step 1: Generate plan
    planning_prompt = build_question_plan_prompt(section_text, num_questions, local_learning_objectives, learning_objectives)
    plan_raw = call_gpt(planning_prompt)
    print(f"""raw: {plan_raw}""")
    try:
        plan = json.loads(plan_raw)
        print(f"""plan: {plan}""")
    except json.JSONDecodeError:
        raise ValueError("Failed to parse question planning response.")

    # Step 2: For each plan item, generate one question
    questions = []
    print(plan['questions'])
    for i, plan_item in enumerate(plan['questions']):
        for attempt in range(3):
            gen_prompt = f"""
You are an AI tutor generating a multiple-choice question.

Based on the section below, create ONE question testing the concept: "{plan_item['concept']}"

The question should adhere to the following target metadata:
- difficulty_score: {plan_item['difficulty_score']}; on a scale from 0.0 (very easy) to 1.0 (very hard).
- salience: {plan_item['salience']}; on a scale from 0.0 (peripheral) to 1.0 (core concept).
- directness: {plan_item['directness']}; on a scale from 0.0 (requires inference) to 1.0 (stated literally).

Guidelines for distractors (the incorrect answer choices):
- Each distractor must be **plausible but incorrect**.
- Distractors should be **grounded in the section text**, referencing real ideas, terms, or claims from the passage—even if slightly twisted or misinterpreted.
- Distractors should be **similar in phrasing, length, and tone** to the correct answer so no option stands out.
- Do **not reuse or rephrase** the correct answer.
- For higher difficulty scores (> 0.6), distractors should be **more subtle**, potentially reflecting likely misunderstandings or partial truths.

Requirements for each question object:
    - question_text: The question prompt (string).
    - options: Exactly 4 distinct answer choices (array of strings).
    - correct_index: Integer 0–3 indicating the correct choice.
    - explanation: Short rationale for the correct answer.
    - concept_tags: A list of 1–3 tags naming the key topics tested.

Respond ONLY in valid JSON with the following fields:
{{
  "question_text": "...",
  "options": ["...", "...", "...", "..."],
  "correct_index": ...,
  "explanation": "...",
  "difficulty_score": ...,
  "concept_tags": ["...", "..."],
  "salience": ...,
  "directness": ...
}}

Section Text:
\"\"\"
{section_text}
\"\"\"
"""
            try:
                q_raw = call_gpt(gen_prompt)
                question = json.loads(q_raw)
                questions.append(question)
                break
            except Exception as e:
                print(f"Retrying question {i+1}: {e}")
                sleep(1)
    return questions
    

def call_gpt(prompt: str, model="gpt-4.1-nano", max_tokens=1000) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        response_format={
            "type": "json_object"
        }
    )
    return resp.choices[0].message.content


# Example usage:
if __name__ == "__main__":
    sample_text = """ Large teams of OpenAI researchers recently published ..."""

    local_learning_objectives = [
        "Summarize GPT-4’s performance on high-level academic reading comprehension tests",
        "Explain the significance of GPT-4’s percentile scores on the SAT, GRE, and LSAT",
        "Describe how test contamination was addressed to ensure valid evaluation of GPT-4"
    ]
                    
    learning_objectives = {
    "1": "Understand how GPT-4’s performance on discourse comprehension and academic tests compares to human performance",
    "2": "Identify the key signatures of genuine understanding in AI, such as inference, generalization, and justification",
    "3": "Explain the significance of increasing text difficulty in revealing differences between AI and human comprehension",
    "4": "Evaluate the current capabilities and limitations of GPT-4 in achieving artificial general intelligence",
    "5": "Recognize the methodological approaches used to assess AI understanding of complex texts"
    }

    quiz = generate_question_set(sample_text, 4, local_learning_objectives, learning_objectives)
    print(json.dumps(quiz, indent=2))
