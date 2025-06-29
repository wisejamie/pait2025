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


# Example usage:
if __name__ == "__main__":
    sample_text = """ 
    GPT-4 generates novel sentence content, has been pre -trained on vast amounts of unlabeled text, \nand uses a transformer architecture that leverages attention mechanisms to focus on relevant parts \nof sentences that may have difficult long -range dependencies. It has been recently trained by \nOpenAI researchers on over 45GB of language data processed by a large neural network with 1.7 6 \ntrillion parameters (trainable connection weights). It is generally acknowledged to be the most \npowerful of the current LLMs. \nThe Discourse Comprehension Test (10) has several features that recommend its use for \ndetermining how well LLMs understand what they read: 1) It focuses entirely on how well verbal \ntext is understood, 2) it is unknown to LLMs because it is protected for medical use, 3) it has been \nstandardized on brain damaged patients known to have difficulty with text understanding as well \nas on neurotypical controls, and 4) its items are structured to experimentally examine the important \nvariables of directness (stated vs. implied information) and salience (main idea vs. detail). \nThis test is comprised of 12 stories describing slightly humorous events that would be \nunderstandable to most North American adults. Each story contains between 191 and 210 words \n3 combined to create 13 or 14 sentences. The stories are at the fifth or sixth grade reading level, and \nare thus relatively easy for North American adults to understand (11). In the Discourse \nComprehension Test, story comprehension is measured by eight yes/no questions characterized by \nsalience (main idea vs. detail) and directness (stated vs. implied information ). \nThere are two questions probing understanding of each of four distinct question types: \nstated main ideas, implied main ideas, stated details, and implied details, making a total of eight \nquestions per story. Questions on the main idea concern central infor mation that gets elaborated \non by other information in the story. Questions on details concern peripheral information that is \nmentioned only once in the story. Stated questions use the same wording as in the story, while \nimplied questions focus on informat ion that is not directly stated but rather must be inferred from \nother information in the story. Answering implied questions correctly thus requires a participant to \nmake bridging assumptions and draw inferences. An example story, along with its questions and \nscoring, is presented in Appendix A. \nThis test has been standardized on three groups of 20 brain -damaged patients (aphasia, \nright hemisphere brain damage, or traumatic brain injury) known to have difficulties \ncomprehending discourse, as well as 40 adults without brain damage (12). Our focus is on \ncomparing GPT -4 to the se 40 neurotypical people. Participants in each of the four human groups \nwere told five test stories after two non -scored practice stories. The three brain -damaged groups \nperformed significantly worse than did the non -brain -damaged control participants. \nIt is very unlikely that GPT -4 has previously encountered any of the stories used in the \nDiscourse Comprehension Test because this is a protected medical test in the field of Speech and \nLanguage Pathology, with stories and questions that are purposely kept out of the public eye and \near. Here we use 11 of these stories for testing GPT-4, leaving out the one story that uses true/false \nquestions rather than yes/no questions. We ran each of the 11 stories through Copilot GPT -4 on 3 \nMarch 2024, preserving the a nswers given to each of the eight questions per story (10). Every \nanswer was printed out well within the five seconds allowed for answers in the human experiment \n(12). An example of GPT -4’s responses to the 8 questions for the story in Appendix A is presented \nin Appendix B. This story is chosen because it had already been posted as an example in an article \ndescribing a human study of discourse comprehension (12). \nIn our first experiment, we use two extra prompts for GPT -4. One prompt precedes the \nstory: Read this story in preparation for answering eight yes/no questions about the story . The \nother prompt follows the story: Answer each of these yes/no questions about the story . Each story \nis itself a necessary prompt. \nIn a follow -up experiment run through Copilot GPT -4 on 2 April 2024 , we instead use a \nprompt to summarize the story and mention main ideas not stated in the story: Summarize this \nstory, mentioning main ideas that are not stated and must be inferred . \nIn our first experiment, w e test GPT -4’s ability to understand brief stories with yes/no \nquestions structured to manipulate the salience and directness of parts of a story. Each of the 88 \nanswers (8 answers per 11 stories ) is categorized as correct , wrong , or unsure . An answer is correct \nif it matches the design ated correct answer (yes or no) (10). Unlike the human participants , who \napparently always conformed to answering only yes or no in their experiment (12), GPT-4 \noccasionally hedge s by providing a neutral answer . Here is an exhaustive list of these neutral \nanswers in our experiment: The story does not specify …, not specified , not mentioned , or The \n4 story does not provide information on this . For these hedged cases, we score the answer’s \ncorrectness as .5 because it is approximately midway between correct (coded 1) and wrong (coded \n0). None of these answers merit s a score of 0, because each of the six incorrect answers are hedged ; \nthey are uncertain rather than being correct or wrong . For completeness, we also alternatively score \nhedged responses as 0, rather than .5. \n2.2 Results \nBecause there are considerably more data points in the human sample (5 stories x 8 questions x 40 \nparticipants = 1600) , we compare a single GPT-4 performance to human performance in terms of \nproportion of correct answers . Proportions correct in the human control sample are computed from \nTable 2 in the human study (12). Our Table 1 presents summary results for humans vs. GPT-4 with \neach of the two scoring methods for hedged responses. Although GPT -4 does very slightly better \nthan humans for each of the two scoring methods, both differences are far below statistical \nsignificance. For the statistical tests in this section , we use the Two Sample Independent \nProportions Test Calculator at Purdue University , a binomial test available online requiring input \nof sample size and successes for each of the two types of participants (humans and GPT -4). \n\nTable 1: Comparison of two scoring methods for GPT -4 to human proportions correct \nover all questions. \nHumans GPT-4 .5 hedge GPT-4 0 hedge \nSample size 1600 88 88 \nSuccesses 1489 85 82 \nProportion .9305 .9659 .9318 \nZ 1.2841 .0429 \np .1991 .9658 \nNote: hedged responses are scored as .5 or 0 in GPT -4. \nFigure 1 shows the proportions correct in each of the four cells of the experiment (2 \ndirectness levels x 2 salience levels ) for humans on the left and GPT -4 on the right. The overall \npattern of proportions correct on the Discourse Comprehension Test (10) for GPT -4 closely \nresembles that for humans. Average neurotypical humans do very well on this test (12) while GPT- \n4 slightly exceeds human performance overall and in three of the four experimental cells portrayed \nin Figure 1 . The pattern of proportions correct are roughly similar for humans and GPT -4 across \nthe four experimental cells. Notably, the worst -performing cell for both humans and GPT -4 is the \nimplied details cell. \n5 \nFigure 1. Proportions correct on the Discourse Comprehension Test for humans on the left and \nGPT-4 on the right, as a function of directness and salience of information . \nFor completeness, we assess whether humans and GPT -4 are performing better than \nchance , again using the Two Sample Independent Proportions Test . Here , chance performance is \ndefined by .5 of sample sizes . The Z and p values in Table 2 certif y that both neurotypical humans \nand GPT-4 models indeed perform well above chance. \nBecause of the theoretical interest in understanding of discourse via implication that goes \nbeyond stated information, we compare GPT -4 to humans on stated -information questions (Table \n3) and implied -information questions (Table 4). These comparisons use t he slightly preferred \nscoring scheme that rates hedged responses as worth .5, as in Figure 1. Again, although GPT -4 \ndoes slightly better than humans on both stated and implied question information, the differences \nin each case are far from reaching statist ical significance. \nTable 2: Comparison of human and GPT -4 performance to chance , defined as .5 success . \nHumans GPT-4 .5 hedge GPT-4 0 hedge \nSample size 1600 88 88 \nSuccesses 800 44 44 \nProportion .9305 .9659 .9318 \nZ 26.99 6.985 6.351 \np 0.0000 0.0000 0.0000 \n\nTable 3: Comparison of proportions correct on stated -information questions. \nHumans GPT-4 \nSample size 800 44 \nSuccesses 770 44 \nProportion .9625 1 \nZ 1.3080 \n\n6 p .1909 \n\nTable 4: Comparison of proportions correct on implied -information questions. \nHumans GPT-4 \nSample size 800 44 \nSuccesses 724 41 \nProportion .9050 .9315 \nZ .5946 \np .5521 \n\nIt is telling that GPT -4 never m akes a wrong response in this experiment. As no ted, it fails \nto give a yes or no response only 6 times out of 88, once on an implied main idea and five times \non implied details. It hedge s on each of these six cases, instead giving neutral uncertain response s \nand appropriate comments that justify their uncertainty . \n\nWe also experiment with GPT -4’s ability to summarize these stories, finding that they \nproduce a concise and accurate paragraph without much in the way of inferences. However, i f we \nask for a summary that mentions inferences, this opens the inferential floodgates . With that prompt, \nGPT-4 produces a total of 54 new inferences that go well beyond those used in the yes/no \nquestions. The mean number of such inferences per story is 4.91, with a standard deviation of 2.02, \nand a range of 2 to 8. An example is provided in Appendix C, using the story presented in Appendix \nA. \n\n2.3 Discussion \nOur results show that GPT -4 matches and even slightly exceeds the high level of human \nperformance on the Discourse Comprehension Test (10). Due to excellent human performance , \nthere is very little room for GPT -4 to exhibit superiority over humans . \nIt makes sense that the worst performance in both humans and GPT -4 is in the experiment \ncell for details and implied knowledge. With memory constraints, details may be ignored or \nforgotten in favor of main points. And producing implications requires additional cognitive effort . \nWe encourage readers to carefully consider the example story presented throughout \nAppendi ces A, B, and C. The combination of never giving a wrong answer while spontaneously \nproviding explanatory justifications makes it hard to believe that a story is not well understood by \nGPT-4. The same impression is given by GPT -4’s spontaneous comments about questions in each \nof the other ten stories. \n\nWe are unable to suppress hedging and comments from GPT -4. Its comments on this task \nare both appropriate and interesting, often justifying a yes-or-no answer and sometimes referring \nprecisely to the process of implication. Number of comments across the eleven stories range from \n7 0-8, with a mean of 3.27. Only one story generated no comments. Human comments were not \nrecorded beyond their yes/no responses (12). \n\nGPT-4’s strong overall performance on these novel stories suggests that it indeed \nunderstand s what it has just learned in a single shot , even when that requires inferencing beyond \nwhat is directly stated in the story . \n\nBecause inferences are required to comprehend most if not all discourses (13), it is very \nlikely that there is already considerable evidence in the LLM literature that GPT -4 uses inference \nin understanding what it reads (3,14) . What is unique about our study is the deliberate experimental \nseparation of salience and directness. This enables focus ing more precisely on how these two \nimportant variables operate and interact. Fortuitously, the existence and use of the Discourse \nComprehension Test provides data allowing a close human comparison while maintaining this \nclear separation between important variables on the same content. \n\nClassical psychology experiments on discourse comprehension typically gave participants \na paragraph to read and then asked them to write down what they remembered of the paragraph \n(15,16) . The experimenter would then count the number of correctly recalled propositions as a \nmeasure of understanding. For several reasons, t his methodology did not provide many interesting \ninsights into discourse comprehension . It confounded understanding with memory, made no \ndistinction between stated and implied information , and generally ignored the important role of \ninference based on general knowledge. In contrast, the Discourse Comprehension Test (10) \nseparates direct from implied information and GPT -4 supplies extensive general world knowledge \nthat promotes interesting and useful inferences . \n\nA close analog to asking a human participant to write out a remembered paragraph is to ask \nGPT-4 to summarize what it has just read. This results in a very concise summary with little or no \nhint of inferencing. However , as noted in the 2.2 Results section , when we request GPT -4 to \nmention inferences to accompany their concise story summary, we discover that it provid es many \ninferences that go well beyond the modest inferencing apparent in our first experiment with yes/no \nquestions. It might be interesting to see whether human participants would likewise provide \nadditional inferences if similarly prompted in this task. \n3. Understanding More Difficult Passages
    """

    local_learning_objectives = [
        "Describe GPT-4’s architecture and training data relevant to discourse comprehension",
            "Explain the features and structure of the Discourse Comprehension Test used in the experiment",
            "Summarize the experimental procedure for GPT-4’s responses to stories and questions",
            "Describe how responses were scored and categorized for correctness",
             "Compare GPT-4’s overall performance to human performance on the Discourse Comprehension Test",
            "Interpret the significance of statistical comparisons between GPT-4 and humans",
            "Identify patterns in GPT-4’s accuracy across different question types and conditions",
            "Explain how GPT-4’s responses demonstrate understanding comparable to humans",
             "Summarize the main findings regarding GPT-4’s performance on simple passages",
            "Explain why both humans and GPT-4 perform worst on implied details questions",
            "Discuss the significance of GPT-4’s ability to provide explanations and justifications",
            "Evaluate the evidence for GPT-4’s genuine understanding of simple texts based on the results"
    ]
                    
    learning_objectives = {
    "1": "Understand how GPT-4’s performance on discourse comprehension and academic tests compares to human performance",
    "2": "Identify the key signatures of genuine understanding in AI, such as inference, generalization, and justification",
    "3": "Explain the significance of increasing text difficulty in revealing differences between AI and human comprehension",
    "4": "Evaluate the current capabilities and limitations of GPT-4 in achieving artificial general intelligence",
    "5": "Recognize the methodological approaches used to assess AI understanding of complex texts"
    }

    print("Starting")
    quiz = generate_question_set(sample_text, 8, local_learning_objectives, learning_objectives)
    print(json.dumps(quiz, indent=2))
