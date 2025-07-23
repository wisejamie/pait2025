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

def generate_question_set(section_text: str, num_questions: int, local_learning_objectives: List[str], learning_objectives: Dict[str, str]) -> List[Dict[str, Any]]:
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
    question_plans = plan.get("questions") if isinstance(plan, dict) else None
    if not isinstance(question_plans, list):
        raise ValueError(
            f"Expected plan['questions'] to be a list, got {type(question_plans).__name__}: {question_plans!r}"
        )
    for idx, plan_item in enumerate(question_plans):
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
  "question_text": str,
  "options": [str, str, str, str],
  "correct_index": int
  "explanation": str,
  "difficulty_score": float
  "concept_tags": [str, str],
  "salience": float
  "directness": float
}}

Section Text:
\"\"\"
{section_text}
\"\"\"
"""
            try:
                concept = plan_item.get("concept")
                if not isinstance(concept, str):
                    raise ValueError(f"Bad plan item at index {idx}, missing 'concept': {plan_item!r}")
                q_raw = call_gpt(gen_prompt)
                q = json.loads(q_raw)
                # ── Enforce exactly 4 options ───────────────────────────
                opts = q.get("options", [])
                if not isinstance(opts, list):
                    raise ValueError(f"Question options not a list: {opts!r}")
                if len(opts) != 4:
                    print(
                       f"⚠️  Warning: question {idx+1} has {len(opts)} options, trimming to 4."
                    )
                    q["options"] = opts[:4]
                    # adjust correct_index if out of range
                    ci = q.get("correct_index", 0)
                    if ci >= len(q["options"]):
                        print(f"    Adjusting correct_index from {ci} to 0")
                        q["correct_index"] = 0                                  # this should probably be random(0,3)
                print(f"question {idx+1} JSON: {q!r}")
                # validate expected structure
                if (
                     not isinstance(q, dict)
                     or "question_text" not in q
                     or not isinstance(q.get("options"), list)
                 ):
                     raise ValueError(f"Invalid question format: {q!r}")
 
                questions.append(q)
                break
            except Exception as e:
                print(f"Error generating question {idx+1}, attempt {attempt+1}: {e}")
                sleep(1)
    return questions
    

def call_gpt(prompt: str, model="gpt-4.1-nano") -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        response_format={
            "type": "json_object"
        }
    )
    return resp.choices[0].message.content
