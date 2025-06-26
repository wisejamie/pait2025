import os
import json
import random
from typing import List, Dict, Any
from dotenv import load_dotenv

import openai

load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

# Initialize the OpenAI client
client = openai.OpenAI(api_key=api_key)

import ast

def _unwrap_list(resp_content: str):
    """
    Parse JSON (or Python literal). If it’s a single QA dict, wrap it in a list.
    If it’s {'key': [...]}, unwrap that. Otherwise, expect a list.
    """
    try:
        data = json.loads(resp_content)
    except json.JSONDecodeError:
        data = ast.literal_eval(resp_content)

    # Case 1: already a list
    if isinstance(data, list):
        return data

    # Case 2: a single QA dict
    if isinstance(data, dict) and set(data.keys()) == {"question", "answer"}:
        return [data]

    # Case 3: one‐key dict whose value is a list
    if isinstance(data, dict) and len(data) == 1:
        only_val = next(iter(data.values()))
        if isinstance(only_val, list):
            return only_val

    raise ValueError(f"Expected a list or a single QA dict, got:\n{data!r}")


def generate_qa_pairs(text: str, num_questions: int = 5) -> List[Dict[str, str]]:
    prompt = f"""
Generate {num_questions} clear, focused question-answer pairs based on the following text.
The questions should vary in directness and salience.
Return your answer as a JSON array of objects, each with "question" and "answer" keys.

Text:
\"\"\"
{text}
\"\"\"
"""
    resp = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    content = resp.choices[0].message.content.strip()
    # debug
    print("raw QA response:", content)
    qa_list = _unwrap_list(content)
    # validate shape
    for item in qa_list:
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            raise ValueError(f"Bad QA item:\n{item!r}")
    return qa_list

def generate_distractors(text: str, question: str, correct_answer: str, num_distractors: int = 3) -> List[str]:
    prompt = f"""
    Here’s the source text:
\"\"\"
{text}
\"\"\"
Below is a multiple-choice question about that text.
Question: {question}
Correct Answer: {correct_answer}

Generate {num_distractors} plausible but incorrect answer choices (distractors)
that are grounded in the above text—i.e. they should reference or twist actual
concepts or terminology from the passage. They should also match the correct answer’s phrasing style and length so one option doesn’t stick out. Do NOT repeat the correct answer.
Return only a JSON array of strings.
"""
    resp = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "text"}
    )
    content = resp.choices[0].message.content.strip()
    # debug
    print("raw distractors response:", content)
    dist_list = _unwrap_list(content)
    # validate
    if not all(isinstance(d, str) for d in dist_list):
        raise ValueError(f"Bad distractors list:\n{dist_list!r}")
    return dist_list


def build_quiz(text: str) -> List[Dict[str, Any]]:
    qa_pairs = generate_qa_pairs(text, num_questions=5)
    quiz = []

    for qa in qa_pairs:
        q = qa["question"]
        a = qa["answer"]
        distractors = generate_distractors(text, q, a, num_distractors=3)

        options = distractors + [a]
        random.shuffle(options)

        quiz.append({
            "question": q,
            "options": options,
            "answer": a
        })

    return quiz


def generate_mcqs(
    text: str,
    num_questions: int = 3,
    num_options: int = 4,
    model: str = "gpt-4.1-nano"
) -> List[Dict[str, Any]]:
    """
    Generate `num_questions` multiple-choice questions from `text`, each with
    `num_options` total choices (1 correct + distractors).

    Returns a list of:
      {
        "question": str,
        "options": [str, …],    # length == num_options, shuffled
        "answer": str           # exactly one of the options
      }
    """
    prompt = f"""
Here is a passage of text:

\"\"\"
{text}
\"\"\"

Please generate {num_questions} multiple-choice questions about this passage.
For each question:
- Provide the exact question as a string.
- Provide exactly {num_options} answer choices in a list called "options".
- One of the options must be the correct answer; name that choice in "answer".
- The incorrect options (distractors) must be plausible and grounded in the text:
  they should reference or subtly twist real terms or concepts from the passage,
  and match the length/style of the correct answer so none stand out.
  
Return ONLY a JSON array of objects, each with these three keys:
[
  {{
    "question": "...",
    "options": ["...","...", …],
    "answer": "..."
  }},
  …
]
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        response_format={
            "type": "json_object"
        }
    )

    # content is already a Python list of dicts
    mcqs: List[Dict[str, Any]] = resp.choices[0].message.content.strip()
    qa_list = _unwrap_list(mcqs)
    # validate shape
    for item in qa_list:
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            raise ValueError(f"Bad QA item:\n{item!r}")
    return qa_list



# -------------------------------NEW TESTING-------------------------------------
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
    sample_text = """
Large teams of OpenAI researchers recently published an extensive and detailed Technical \nReport on the capabilities, limitations, and safety characteristics of GPT -4 (17). Among the \ncapabilities that they addressed were performances on 34 academic tests covering a wide range of \n8 fields. Three of these academic tests had sections that addressed reading comprehension at higher \nlevels of difficulty than the Discourse Comprehension Test used in our section 2: SAT, GRE, and \nLSAT. \nOur section 3.1 is a review of GPT -4 performance on these three widely used and highly \nstandardized academic tests (17). They each have a large component devoted to reading \ncomprehension . OpenAI researchers verified that there was no special GPT -4 training for these \nthree tests, and they also ran contamination checks for test data appearing in the training set (17). \nIf matches to the test set were found in the training set, they were removed from the test set to \ncreate an uncontaminated test set. \nTable 5 shows the percentile achieved by GPT -4 in each test after eliminating any \ncontamination from the training set. The mean uncontaminated percentile across the three tests is \n96.3. By statistical definition, the average percentile achieved by thousands of student test -takers \nis the 50th percentile , thus revealing a substantial superiority for GPT -4 with reading \ncomprehension of difficult passages. The prompts given to GPT -4 reflected the test requirements \n(17). \nTable 5. GPT -4 Uncontaminated Percentile Scores on 3 Academic Tests that include Reading \nComprehension . \nTest Percentile \nScholastic Aptitude Test (SAT) Reading & Writing 93rd \nGraduate Record Examination (GRE) Verbal 99th \nLaw School Admission Test (LSAT) 97th \nAdapted from OpenAI 2024, their Table 9 in their Appendix G. (17) \nFor the SAT and GRE Verbal exams, scores were identical with and without contamination, \nwhile for the LSAT, GPT -4 performed slightly better on uncontaminated questions. This finding \nsupports OpenAI’s conclusion that contamination had little to no effect on GPT -4's scores and \nsuggests that GPT -4’s high scores reflect its reading comprehension abilities rather than specific \nmemorized content from training data (17). \nThe SAT is widely used for college admissions in North America. The Reading section has \nbrief passages (or a passage pair) followed by a multiple -choice question. Passages range from 25 \nto 150 words. The subject areas for Reading and Writing cover literature, history , social studies, \nhumanities, and science. Students have 64 minutes to complete the Reading and Writing section . \nReading Comprehension questions on the GRE are designed to test for the abilit y to \nunderstand the kinds of prose commonly encountered in graduate and professional schoo ls, \nincluding drawing conclusions from information , reasoning from incomplete data to infer \nmissing information , understanding how the parts of a passage relate to each other, analyzing a \ntext and reaching its conclusions , considering alternative explanations , and formulating and testing \nhypotheses. Test passages are borrowed from academic and non -academic books and articles \ncovering science, arts , humanities, business , and everyday topics. \n9 Reading comprehension passages and questions on the LSAT seem particularly well suited \nto discovering indications of true understanding as they often requi re the reader to reason beyond \nthe literal text. Their m ultiple -choice questions probe for main ideas, explicitly stated information, \ninferable information, generalization to different contexts, and analogizing. \n3.2 Other Signatures of Understanding
    """

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
    # quiz = generate_mcqs(sample_text, 5, 4)
    # print(json.dumps(quiz, indent=2))
