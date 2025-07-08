import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from uuid import uuid4
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
import os
import openai
from dotenv import load_dotenv
import json
from app.utils.text_extraction import build_section_extraction_prompt, extract_section_text, SectionExtractionError, extract_text_from_pdf
# from app.utils.text_extraction import build_section_extraction_prompt, extract_section_text, SectionExtractionError
from app.utils.question_pipeline import generate_question_set
from app.models.document_models import *
from app.models.question_models import *
from app.models.quiz_models import *
from app.storage.memory import *


from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env
load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

# Initialize the OpenAI client
client = openai.OpenAI(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Section.update_forward_refs()

# Routes
@app.post("/documents/", response_model=DocumentResponse)
async def upload_document(doc: DocumentInput):
    # Simulate storing and start processing
    doc_id = str(uuid4())
    DOCUMENTS[doc_id] = {
        "title": doc.title,
        "raw_text": doc.raw_text,
        "upload_time": datetime.now().isoformat(),
        "status": "processing",
        "sections": [],
        "learning_objectives": {},
    }
    
    return {"document_id": doc_id, "status": "processing"}


@app.get("/documents", response_model=List[DocumentView])
def list_documents():
    return [
        {
            "document_id": doc_id,
            "title": doc.get("title", "Untitled"),
            "upload_time": doc.get("upload_time", datetime.now().isoformat())
        }
        for doc_id, doc in DOCUMENTS.items()
    ]

@app.get("/documents/{doc_id}", response_model=DocumentFullView)
async def get_document(doc_id: str):
    doc = DOCUMENTS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "document_id": doc_id,
        "title": doc.get("title", "Untitled"),
        "raw_text": doc.get("raw_text", ""),
        "status": doc.get("status", "ready"),
        "sections": doc.get("sections", []),
        "learning_objectives": doc.get("learning_objectives", {}),
    }

@app.get("/documents/{doc_id}/sections")
def get_sections(doc_id: str):
    doc = DOCUMENTS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    sections = doc.get("sections", [])
    learning_objectives = doc.get("learning_objectives", {})

    # # Ensure section IDs are assigned
    sections = add_section_ids(sections, doc_id)

    return {
        "sections": sections,
        "learning_objectives": learning_objectives,
    }


@app.get("/documents/{doc_id}/sections/{section_id}")
def get_section(doc_id: str, section_id: str):
    for doc in DOCUMENTS.values():
        sections = doc.get("sections", [])
        flat_sections = flatten_sections(sections)
        for section in flat_sections:
            if section.get("id") == section_id:
                return section

    raise HTTPException(status_code=404, detail="Section not found")

def flatten_sections(sections):
    """Flatten a nested list of sections and their sub_sections into a single flat list."""
    flat = []

    for section in sections:
        flat.append(section)
        if "sub_sections" in section and isinstance(section["sub_sections"], list):
            flat.extend(flatten_sections(section["sub_sections"]))

    return flat


def add_section_ids(
    sections: List[Dict],
    doc_id: str,
    parent_index: str = ""
) -> List[Dict]:
    for i, section in enumerate(sections):
        # Build section ID based on its position
        section_id = f"{doc_id}_sec{parent_index}{i}"
        section['id'] = section_id

        # Recurse into any nested subsections
        sub_secs = section.get('sub_sections')
        if sub_secs:
            # Pass along the current index path with a trailing underscore
            next_parent = f"{parent_index}{i}_"
            add_section_ids(sub_secs, doc_id, parent_index=next_parent)

    return sections


@app.post("/documents/upload-file", response_model=DocumentResponse)
async def upload_document_file(file: UploadFile = File(...), title: str = Form(...)):
    # Check file type
    if file.content_type == "application/pdf":
        raw_text = extract_text_from_pdf(file.file)
    elif file.content_type == "text/plain":
        raw_bytes = await file.read()
        raw_text = raw_bytes.decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    doc_id = str(uuid4())
    DOCUMENTS[doc_id] = {
        "title": title,
        "raw_text": raw_text,
        "upload_time": datetime.now().isoformat(),
        "status": "processing",
        "sections": None,
        "learning_objectives": None,
    }

    return {"document_id": doc_id, "status": "processing"}


@app.post("/documents/{doc_id}/sections/detect", response_model=SectionDetectionResponse)
async def detect_sections(doc_id: str):
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")

    raw_text = DOCUMENTS[doc_id]["raw_text"]

    # GPT prompt
    prompt = build_section_extraction_prompt(raw_text)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        gpt_output = response.choices[0].message.content

        # Try to safely parse JSON
        data = json.loads(gpt_output)
        sections = data["sections"]

        # Add section text to each section
        try:
            sections_with_text = extract_section_text(raw_text, sections)
            for section in sections_with_text:
                section["questions"] = []
        except SectionExtractionError as e:
            raise HTTPException(status_code=422, detail=str(e))

        # Save enriched sections
        DOCUMENTS[doc_id]["sections"] = sections_with_text
        DOCUMENTS[doc_id]["learning_objectives"] = data["learning_objectives"]
        DOCUMENTS[doc_id]["status"] = "ready"

        return {
            "sections": sections_with_text,
            "learning_objectives": data["learning_objectives"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing GPT response: {e}")


@app.get("/documents/{doc_id}", response_model=DocumentFullView)
async def get_document(doc_id: str):
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = DOCUMENTS[doc_id]

    return {
        "document_id": doc_id,
        "title": doc.get("title"),
        "raw_text": doc["raw_text"],
        "status": doc.get("status", "processing"),
        "sections": doc.get("sections"),
        "learning_objectives": doc.get("learning_objectives"),
    }

@app.get("/documents/{doc_id}/questions", response_model=List[Question])
async def get_document_questions(doc_id: str):
    doc = DOCUMENTS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    sections = doc.get("sections", [])
    all_questions = []

    def collect_questions(section_list):
        for section in section_list:
            section_id = section.get("id")
            if section_id and f"{section_id}" in QUESTIONS:
                all_questions.extend(QUESTIONS[f"{section_id}"])
            if "sub_sections" in section:
                collect_questions(section["sub_sections"])

    collect_questions(sections)
    return all_questions

@app.post("/documents/{doc_id}/questions/generate-all")
async def generate_questions_for_all_sections(doc_id: str):
    doc = DOCUMENTS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    top_sections = doc.get("sections", [])
    global_learning_objectives = doc.get("learning_objectives", {})

    # Recursively collect only leaf sections (no sub_sections)
    def get_leaf_sections(sections: List[Dict], parent_index="") -> List[Tuple[str, Dict]]:
        leaf_sections = []
        for i, sec in enumerate(sections):
            section_id = f"{doc_id}_sec{parent_index}{i}"
            sub_secs = sec.get("sub_sections", [])
            if sub_secs:
                leaf_sections.extend(get_leaf_sections(sub_secs, parent_index=f"{parent_index}{i}_"))
            else:
                leaf_sections.append((section_id, sec))
        return leaf_sections

    leaf_sections = get_leaf_sections(top_sections)
    results: Dict[str, Any] = {}

    for section_id, section in leaf_sections:
        try:
            section_text = section.get("text", "").strip()
            if not section_text:
                results[section_id] = {"error": "No section text found"}
                continue

            local_learning_objectives = section.get("learning_objectives", [])
            questions = generate_question_set(
                section_text=section_text,
                num_questions=2,
                local_learning_objectives=local_learning_objectives,
                learning_objectives=global_learning_objectives
            )

            QUESTIONS[f"{section_id}"] = questions
            results[section_id] = questions

        except Exception as e:
            results[section_id] = {"error": str(e)}

    return results

import random
@app.post("/quiz-sessions/")
async def create_quiz_session(req: QuizSessionCreateRequest):
    doc_id = req.document_id
    num_questions = req.num_questions
    sections = req.sections

    print(doc_id)
    # print(DOCUMENTS)
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")

    all_question_ids = []
    if sections:
        for sec in sections:
            key = f"{sec}"
            qs = QUESTIONS.get(key, [])
            for idx, q in enumerate(qs):
                question_id = f"{key}_q{idx}"
                all_question_ids.append((question_id, sec, q))
    
    else:
        for section_id, questions in QUESTIONS.items():
            if section_id.startswith(doc_id):
                for idx, q in enumerate(questions):
                    question_id = f"{section_id}_q{idx}"
                    all_question_ids.append((question_id, section_id, q))
    
    if not all_question_ids:
        raise HTTPException(
            status_code=400,
            detail="No questions available for the selected sections."
        )

    # validate num_questions
    if num_questions < 1:
        raise HTTPException(400, "num_questions out of range.")

    # ── Randomly sample the desired number
    selected = random.sample(all_question_ids, min(num_questions, len(all_question_ids)))

    # ── Extract IDs and question bodies for session storage
    selected_ids = [qid for qid, _, _ in selected]
    session_questions = { qid: qdict for qid, _, qdict in selected }

    # Store the session
    session_id = str(uuid4())
    QUIZ_SESSIONS[session_id] = {
        "document_id": doc_id,
        "question_refs": selected,  # (question_id, section_id, question_obj)
        "question_ids": selected_ids,
        "responses": [],
        "current_index": 0,
        "start_time": datetime.now().isoformat(),
        "end_time": None
    }

    return {
        "session_id": session_id,
        "total_questions": len(selected_ids),
        "status": "active"
    }

@app.get("/quiz-sessions/{session_id}/next")
async def get_next_question(session_id: str):
    if session_id not in QUIZ_SESSIONS:
        raise HTTPException(status_code=404, detail="Quiz session not found")

    session = QUIZ_SESSIONS[session_id]
    index = session["current_index"]
    total = len(session["question_refs"])

    if index >= total:
        return {"message": "Quiz completed", "finished": True}

    question_id, section_id, question_obj = session["question_refs"][index]

    # Return question data (hide answer metadata)
    return {
        "question_id": question_id,
        "index": index,
        "total": total,
        "question": {
            "question_text": question_obj["question_text"],
            "options": question_obj["options"]
        }
    }

@app.post("/quiz-sessions/{session_id}/answer")
async def submit_answer(session_id: str, submission: AnswerSubmission):
    if session_id not in QUIZ_SESSIONS:
        raise HTTPException(status_code=404, detail="Quiz session not found")

    session = QUIZ_SESSIONS[session_id]
    index = session["current_index"]
    total = len(session["question_refs"])

    if index >= total:
        raise HTTPException(status_code=400, detail="Quiz already completed")

    current_qid, section_id, question_obj = session["question_refs"][index]

    if submission.question_id != current_qid:
        raise HTTPException(status_code=400, detail="Submitted question does not match current question")

    is_correct = submission.selected_index == question_obj["correct_index"]

    session["responses"].append({
        "question_id": current_qid,
        "selected_index": submission.selected_index,
        "correct": is_correct
    })

    session["current_index"] += 1

    return {
        "correct": is_correct,
        "correct_index": question_obj["correct_index"],
        "explanation": question_obj.get("explanation", ""),
        "next_index": session["current_index"],
        "completed": session["current_index"] >= total
    }

# ── helper to find a section’s title by its ID
def lookup_section_title(doc_id: str, section_id: str) -> str:
    doc = DOCUMENTS.get(doc_id, {})
    sections = doc.get("sections", [])

    def recurse(secs):
        for sec in secs:
            if sec["id"] == section_id:
                return sec["title"]
            title = recurse(sec.get("sub_sections", []))
            if title:
                return title
        return None

    return recurse(sections) or section_id

@app.get("/quiz-sessions/{session_id}/summary")
async def get_quiz_summary(session_id: str):
    if session_id not in QUIZ_SESSIONS:
        raise HTTPException(status_code=404, detail="Quiz session not found")

    session = QUIZ_SESSIONS[session_id]
    total = len(session["question_refs"])
    responses = session["responses"]

    if len(responses) < total:
        return {"message": "Quiz not yet complete", "finished": False}

    num_correct = sum(1 for r in responses if r["correct"])
    num_incorrect = total - num_correct
    score_pct = round((num_correct / total) * 100, 2)

     # ── compute per‐section stats
    section_stats: Dict[str, Dict[str,int]] = {}
    for idx, response in enumerate(responses):
        _, section_id, _ = session["question_refs"][idx]
        stats = section_stats.setdefault(section_id, {"total": 0, "correct": 0})
        stats["total"] += 1
        if response["correct"]:
            stats["correct"] += 1

    section_scores = []
    for section_id, stats in section_stats.items():
        total_sec = stats["total"]
        correct_sec = stats["correct"]
        section_scores.append({
            "section_id": section_id,
            "section_title": lookup_section_title(session["document_id"], section_id),
            "correct": correct_sec,
            "incorrect": total_sec - correct_sec,
            "total": total_sec,
            "percent": round((correct_sec / total_sec) * 100, 2)
        })

    missed_questions = []
    for i, response in enumerate(responses):
        if not response["correct"]:
            qid, section_id, qobj = session["question_refs"][i]
            missed_questions.append({
                "question_id": qid,
                "question_text": qobj["question_text"],
                "options": qobj["options"],
                "selected_index": response["selected_index"],
                "correct_index": qobj["correct_index"],
                "explanation": qobj.get("explanation", "")
            })

    return {
        "total_questions": total,
        "correct": num_correct,
        "incorrect": num_incorrect,
        "score_percent": score_pct,
        "section_scores": section_scores,
        "missed_questions": missed_questions,
        "finished": True,
        "document_id": session["document_id"] 
    }

# FOR TESTING PURPOSES:
@app.on_event("startup")
async def load_test_data():
    doc_id = "test-doc"

    # Add fake document
    DOCUMENTS[doc_id] = {
        "document_id": "test-doc",
        "title": "Test Document",
        "raw_text": "This is a sample document about how AI models comprehend text. It contains one section with meaningful content.",
        "status": "ready",
        "sections": [
            {
            "title": "Introduction to AI Comprehension",
            "first_sentence": "This is a sample document about how AI models comprehend text.",
            "text": "This is a sample document about how AI models comprehend text. It contains one section with meaningful content.",
            "sub_sections": [],
            "id":"sec0"
            }
        ],
        "learning_objectives": {
            "1": "Understand how AI models interpret language.",
            "2": "Identify evaluation metrics for AI comprehension."
        }
    }

    DOCUMENTS["c8636fc3-9e38-4e1b-9e52-83052c72de9d"] = {
        "document_id": "c8636fc3-9e38-4e1b-9e52-83052c72de9d",
        "title": "Text Understanding in GPT-4 vs. Humans",
        "raw_text":"",
        "status":"ready",
        
        "sections": [
            {
            "title": "Introduction",
            "first_sentence": "Recent advances in artificial intelligence (AI) have generated vigorous debates about whether",
            "text": "Recent advances in artificial intelligence (AI) have generated vigorous debates about whether \nthese computational systems are approac hing the level of Artificial General Intelligence (AGI), \nwith humans often serving as the gold -standard of comparison. These computational systems are \nvariously described as Large Language Models (LLMs) because they are large neural networks \ntrained on enormous amounts of text, or Chat -Bots because it is possible to chat with them online , \nor generative AI because it can generate novel material . There is growing evidence that LLMs have \nmaster ed a variety of challenging human cognitive tasks involving language, mathematics, \ncomputer coding, law, medicine, vision, and more . However, there are also arguments that such \nsystems that are trained to predict the next token word in vast quantities of sentences cannot reach \nthe subtleties and power of hu man cognition and language , in some cases supported by adversarial \ninputs that manage to trip up some of the current AI systems (1,2) . \nAn important review of the empirical literature on these debates has identified a n \ninteresting recent trend wherein newer , more powerful systems (e.g. , GPT-4) have significantly \nimprove d on the shortcomings of earlier systems (e.g., GPT -3 and GPT-3.5) (3). The basic research \nstrategy in these studies is to use an LLM to play the role of human participants in psychology \nexperiment s, by now covering an increasingly wide variety of challenging cognitive tasks . In such \ncomparisons, sufficiently powerful LLMs often reach ed or even exceeded human levels of \nperformance . \nFor example, GPT -3.5 and GPT -4 were tested on a range of analogical reasoning tasks, \nincluding one with the rule structure of Raven’s Standard Progressive Matrices, a widely used test \nof human intelligence. In that IQ test, GPT-3.5 matched or exceeded average human performance \nand GPT -4 did even better (4). The authors noted that these models required no learning beyond \ntheir general training, an impressive demonstration of zero -shot learning. \n2 Another example involved Theory -of-Mind tasks, which had often been considered \nexclusive to humans and thus impossible for LLMs to solve. Thus, it was not too surprising that \nearly GPT models (3 and 3.5) solved only 20% of these tasks . However, the more powerful GPT- \n4 solved 75% of them, matching the performance of six -year-old children (5). \nOther work reviewed in (3) showed that earlier LLMs performed at average human levels \non a variety of challenging cognitive tasks, including metaphors, idioms, estimation, anaphoric \nreference, creativity, problem solving, logic, and answering common sense questions. GPT-3 also \nsimulated average human results in gambles (including some human biases), and solved a \nmultiarmed bandit task better than human participants (6). It would be interesting to see whether \nGPT-4 could exceed human performance on these tasks. It is noteworthy that causal reasoning (6), \nplanning (7), and graduate -level mathematics (8) have been more challenging for LLMs , even for \nGPT-4. \nIn this work, we focus on the critically important issue of whether LLMs understand the \ntext they have read , using standard psychological methods in which GPT -4 plays the role of a \nhuman agent . Section 2 deals with simple passages of text , while section 3 concerns considerably \nmore difficult passages. \nIn psychology , text comprehension consists of building multi -level representations of the \ninformation in a passage of text (9). The c omprehension improves when the reader has enough \nbackground knowledge to assimilate the text and as the reader constructs more representation \nlevels and more inferences at each level. Successful comprehension can be measured by any of \nseveral abilities: correctly answer ing relevant questions, draw ing relevant and correct inferences, \nasking good questions, generat ing good summaries of the text, and detect ing anomalies and \ncontradictions. Generalization is considered as a kind of inference that has fairly wide application. \nKnowledge -based inferences are constructed when background knowledge in long -term memory \nis activated and then encoded in the meaning representation of the text . We examine as many of \nthese abilities as the data allow to assess text comprehension in humans and GPT -4. \n2. Understanding Relatively Simple Passages",
            "sub_sections": []
            },
            {
            "title": "Understanding Relatively Simple Passages",
            "first_sentence": "2.1 Methods GPT-4 generates novel sentence content, has been pre-trained on vast amounts",
            "text": "2.1 Methods \nGPT-4 generates novel sentence content, has been pre -trained on vast amounts of unlabeled text, \nand uses a transformer architecture that leverages attention mechanisms to focus on relevant parts \nof sentences that may have difficult long -range dependencies. It has been recently trained by \nOpenAI researchers on over 45GB of language data processed by a large neural network with 1.7 6 \ntrillion parameters (trainable connection weights). It is generally acknowledged to be the most \npowerful of the current LLMs. \nThe Discourse Comprehension Test (10) has several features that recommend its use for \ndetermining how well LLMs understand what they read: 1) It focuses entirely on how well verbal \ntext is understood, 2) it is unknown to LLMs because it is protected for medical use, 3) it has been \nstandardized on brain damaged patients known to have difficulty with text understanding as well \nas on neurotypical controls, and 4) its items are structured to experimentally examine the important \nvariables of directness (stated vs. implied information) and salience (main idea vs. detail). \nThis test is comprised of 12 stories describing slightly humorous events that would be \nunderstandable to most North American adults. Each story contains between 191 and 210 words \n3 combined to create 13 or 14 sentences. The stories are at the fifth or sixth grade reading level, and \nare thus relatively easy for North American adults to understand (11). In the Discourse \nComprehension Test, story comprehension is measured by eight yes/no questions characterized by \nsalience (main idea vs. detail) and directness (stated vs. implied information ). \nThere are two questions probing understanding of each of four distinct question types: \nstated main ideas, implied main ideas, stated details, and implied details, making a total of eight \nquestions per story. Questions on the main idea concern central infor mation that gets elaborated \non by other information in the story. Questions on details concern peripheral information that is \nmentioned only once in the story. Stated questions use the same wording as in the story, while \nimplied questions focus on informat ion that is not directly stated but rather must be inferred from \nother information in the story. Answering implied questions correctly thus requires a participant to \nmake bridging assumptions and draw inferences. An example story, along with its questions and \nscoring, is presented in Appendix A. \nThis test has been standardized on three groups of 20 brain -damaged patients (aphasia, \nright hemisphere brain damage, or traumatic brain injury) known to have difficulties \ncomprehending discourse, as well as 40 adults without brain damage (12). Our focus is on \ncomparing GPT -4 to the se 40 neurotypical people. Participants in each of the four human groups \nwere told five test stories after two non -scored practice stories. The three brain -damaged groups \nperformed significantly worse than did the non -brain -damaged control participants. \nIt is very unlikely that GPT -4 has previously encountered any of the stories used in the \nDiscourse Comprehension Test because this is a protected medical test in the field of Speech and \nLanguage Pathology, with stories and questions that are purposely kept out of the public eye and \near. Here we use 11 of these stories for testing GPT-4, leaving out the one story that uses true/false \nquestions rather than yes/no questions. We ran each of the 11 stories through Copilot GPT -4 on 3 \nMarch 2024, preserving the a nswers given to each of the eight questions per story (10). Every \nanswer was printed out well within the five seconds allowed for answers in the human experiment \n(12). An example of GPT -4’s responses to the 8 questions for the story in Appendix A is presented \nin Appendix B. This story is chosen because it had already been posted as an example in an article \ndescribing a human study of discourse comprehension (12). \nIn our first experiment, we use two extra prompts for GPT -4. One prompt precedes the \nstory: Read this story in preparation for answering eight yes/no questions about the story . The \nother prompt follows the story: Answer each of these yes/no questions about the story . Each story \nis itself a necessary prompt. \nIn a follow -up experiment run through Copilot GPT -4 on 2 April 2024 , we instead use a \nprompt to summarize the story and mention main ideas not stated in the story: Summarize this \nstory, mentioning main ideas that are not stated and must be inferred . \nIn our first experiment, w e test GPT -4’s ability to understand brief stories with yes/no \nquestions structured to manipulate the salience and directness of parts of a story. Each of the 88 \nanswers (8 answers per 11 stories ) is categorized as correct , wrong , or unsure . An answer is correct \nif it matches the design ated correct answer (yes or no) (10). Unlike the human participants , who \napparently always conformed to answering only yes or no in their experiment (12), GPT-4 \noccasionally hedge s by providing a neutral answer . Here is an exhaustive list of these neutral \nanswers in our experiment: The story does not specify …, not specified , not mentioned , or The \n4 story does not provide information on this . For these hedged cases, we score the answer’s \ncorrectness as .5 because it is approximately midway between correct (coded 1) and wrong (coded \n0). None of these answers merit s a score of 0, because each of the six incorrect answers are hedged ; \nthey are uncertain rather than being correct or wrong . For completeness, we also alternatively score \nhedged responses as 0, rather than .5. \n2.2 Results \nBecause there are considerably more data points in the human sample (5 stories x 8 questions x 40 \nparticipants = 1600) , we compare a single GPT-4 performance to human performance in terms of \nproportion of correct answers . Proportions correct in the human control sample are computed from \nTable 2 in the human study (12). Our Table 1 presents summary results for humans vs. GPT-4 with \neach of the two scoring methods for hedged responses. Although GPT -4 does very slightly better \nthan humans for each of the two scoring methods, both differences are far below statistical \nsignificance. For the statistical tests in this section , we use the Two Sample Independent \nProportions Test Calculator at Purdue University , a binomial test available online requiring input \nof sample size and successes for each of the two types of participants (humans and GPT -4). \n\nTable 1: Comparison of two scoring methods for GPT -4 to human proportions correct \nover all questions. \nHumans GPT-4 .5 hedge GPT-4 0 hedge \nSample size 1600 88 88 \nSuccesses 1489 85 82 \nProportion .9305 .9659 .9318 \nZ 1.2841 .0429 \np .1991 .9658 \nNote: hedged responses are scored as .5 or 0 in GPT -4. \nFigure 1 shows the proportions correct in each of the four cells of the experiment (2 \ndirectness levels x 2 salience levels ) for humans on the left and GPT -4 on the right. The overall \npattern of proportions correct on the Discourse Comprehension Test (10) for GPT -4 closely \nresembles that for humans. Average neurotypical humans do very well on this test (12) while GPT- \n4 slightly exceeds human performance overall and in three of the four experimental cells portrayed \nin Figure 1 . The pattern of proportions correct are roughly similar for humans and GPT -4 across \nthe four experimental cells. Notably, the worst -performing cell for both humans and GPT -4 is the \nimplied details cell. \n5 \nFigure 1. Proportions correct on the Discourse Comprehension Test for humans on the left and \nGPT-4 on the right, as a function of directness and salience of information . \nFor completeness, we assess whether humans and GPT -4 are performing better than \nchance , again using the Two Sample Independent Proportions Test . Here , chance performance is \ndefined by .5 of sample sizes . The Z and p values in Table 2 certif y that both neurotypical humans \nand GPT-4 models indeed perform well above chance. \nBecause of the theoretical interest in understanding of discourse via implication that goes \nbeyond stated information, we compare GPT -4 to humans on stated -information questions (Table \n3) and implied -information questions (Table 4). These comparisons use t he slightly preferred \nscoring scheme that rates hedged responses as worth .5, as in Figure 1. Again, although GPT -4 \ndoes slightly better than humans on both stated and implied question information, the differences \nin each case are far from reaching statist ical significance. \nTable 2: Comparison of human and GPT -4 performance to chance , defined as .5 success . \nHumans GPT-4 .5 hedge GPT-4 0 hedge \nSample size 1600 88 88 \nSuccesses 800 44 44 \nProportion .9305 .9659 .9318 \nZ 26.99 6.985 6.351 \np 0.0000 0.0000 0.0000 \n\nTable 3: Comparison of proportions correct on stated -information questions. \nHumans GPT-4 \nSample size 800 44 \nSuccesses 770 44 \nProportion .9625 1 \nZ 1.3080 \n\n6 p .1909 \n\nTable 4: Comparison of proportions correct on implied -information questions. \nHumans GPT-4 \nSample size 800 44 \nSuccesses 724 41 \nProportion .9050 .9315 \nZ .5946 \np .5521 \n\nIt is telling that GPT -4 never m akes a wrong response in this experiment. As no ted, it fails \nto give a yes or no response only 6 times out of 88, once on an implied main idea and five times \non implied details. It hedge s on each of these six cases, instead giving neutral uncertain response s \nand appropriate comments that justify their uncertainty . \n\nWe also experiment with GPT -4’s ability to summarize these stories, finding that they \nproduce a concise and accurate paragraph without much in the way of inferences. However, i f we \nask for a summary that mentions inferences, this opens the inferential floodgates . With that prompt, \nGPT-4 produces a total of 54 new inferences that go well beyond those used in the yes/no \nquestions. The mean number of such inferences per story is 4.91, with a standard deviation of 2.02, \nand a range of 2 to 8. An example is provided in Appendix C, using the story presented in Appendix \nA. \n\n2.3 Discussion \nOur results show that GPT -4 matches and even slightly exceeds the high level of human \nperformance on the Discourse Comprehension Test (10). Due to excellent human performance , \nthere is very little room for GPT -4 to exhibit superiority over humans . \nIt makes sense that the worst performance in both humans and GPT -4 is in the experiment \ncell for details and implied knowledge. With memory constraints, details may be ignored or \nforgotten in favor of main points. And producing implications requires additional cognitive effort . \nWe encourage readers to carefully consider the example story presented throughout \nAppendi ces A, B, and C. The combination of never giving a wrong answer while spontaneously \nproviding explanatory justifications makes it hard to believe that a story is not well understood by \nGPT-4. The same impression is given by GPT -4’s spontaneous comments about questions in each \nof the other ten stories. \n\nWe are unable to suppress hedging and comments from GPT -4. Its comments on this task \nare both appropriate and interesting, often justifying a yes-or-no answer and sometimes referring \nprecisely to the process of implication. Number of comments across the eleven stories range from \n7 0-8, with a mean of 3.27. Only one story generated no comments. Human comments were not \nrecorded beyond their yes/no responses (12). \n\nGPT-4’s strong overall performance on these novel stories suggests that it indeed \nunderstand s what it has just learned in a single shot , even when that requires inferencing beyond \nwhat is directly stated in the story . \n\nBecause inferences are required to comprehend most if not all discourses (13), it is very \nlikely that there is already considerable evidence in the LLM literature that GPT -4 uses inference \nin understanding what it reads (3,14) . What is unique about our study is the deliberate experimental \nseparation of salience and directness. This enables focus ing more precisely on how these two \nimportant variables operate and interact. Fortuitously, the existence and use of the Discourse \nComprehension Test provides data allowing a close human comparison while maintaining this \nclear separation between important variables on the same content. \n\nClassical psychology experiments on discourse comprehension typically gave participants \na paragraph to read and then asked them to write down what they remembered of the paragraph \n(15,16) . The experimenter would then count the number of correctly recalled propositions as a \nmeasure of understanding. For several reasons, t his methodology did not provide many interesting \ninsights into discourse comprehension . It confounded understanding with memory, made no \ndistinction between stated and implied information , and generally ignored the important role of \ninference based on general knowledge. In contrast, the Discourse Comprehension Test (10) \nseparates direct from implied information and GPT -4 supplies extensive general world knowledge \nthat promotes interesting and useful inferences . \n\nA close analog to asking a human participant to write out a remembered paragraph is to ask \nGPT-4 to summarize what it has just read. This results in a very concise summary with little or no \nhint of inferencing. However , as noted in the 2.2 Results section , when we request GPT -4 to \nmention inferences to accompany their concise story summary, we discover that it provid es many \ninferences that go well beyond the modest inferencing apparent in our first experiment with yes/no \nquestions. It might be interesting to see whether human participants would likewise provide \nadditional inferences if similarly prompted in this task. \n3. Understanding More Difficult Passages",
            "sub_sections": [
                {
                "title": "Methods",
                "first_sentence": "GPT-4 generates novel sentence content, has been pre-trained on vast amounts of unlabeled text,",
                "sub_sections": [],
                "text": "GPT-4 generates novel sentence content, has been pre -trained on vast amounts of unlabeled text, \nand uses a transformer architecture that leverages attention mechanisms to focus on relevant parts \nof sentences that may have difficult long -range dependencies. It has been recently trained by \nOpenAI researchers on over 45GB of language data processed by a large neural network with 1.7 6 \ntrillion parameters (trainable connection weights). It is generally acknowledged to be the most \npowerful of the current LLMs. \nThe Discourse Comprehension Test (10) has several features that recommend its use for \ndetermining how well LLMs understand what they read: 1) It focuses entirely on how well verbal \ntext is understood, 2) it is unknown to LLMs because it is protected for medical use, 3) it has been \nstandardized on brain damaged patients known to have difficulty with text understanding as well \nas on neurotypical controls, and 4) its items are structured to experimentally examine the important \nvariables of directness (stated vs. implied information) and salience (main idea vs. detail). \nThis test is comprised of 12 stories describing slightly humorous events that would be \nunderstandable to most North American adults. Each story contains between 191 and 210 words \n3 combined to create 13 or 14 sentences. The stories are at the fifth or sixth grade reading level, and \nare thus relatively easy for North American adults to understand (11). In the Discourse \nComprehension Test, story comprehension is measured by eight yes/no questions characterized by \nsalience (main idea vs. detail) and directness (stated vs. implied information ). \nThere are two questions probing understanding of each of four distinct question types: \nstated main ideas, implied main ideas, stated details, and implied details, making a total of eight \nquestions per story. Questions on the main idea concern central infor mation that gets elaborated \non by other information in the story. Questions on details concern peripheral information that is \nmentioned only once in the story. Stated questions use the same wording as in the story, while \nimplied questions focus on informat ion that is not directly stated but rather must be inferred from \nother information in the story. Answering implied questions correctly thus requires a participant to \nmake bridging assumptions and draw inferences. An example story, along with its questions and \nscoring, is presented in Appendix A. \nThis test has been standardized on three groups of 20 brain -damaged patients (aphasia, \nright hemisphere brain damage, or traumatic brain injury) known to have difficulties \ncomprehending discourse, as well as 40 adults without brain damage (12). Our focus is on \ncomparing GPT -4 to the se 40 neurotypical people. Participants in each of the four human groups \nwere told five test stories after two non -scored practice stories. The three brain -damaged groups \nperformed significantly worse than did the non -brain -damaged control participants. \nIt is very unlikely that GPT -4 has previously encountered any of the stories used in the \nDiscourse Comprehension Test because this is a protected medical test in the field of Speech and \nLanguage Pathology, with stories and questions that are purposely kept out of the public eye and \near. Here we use 11 of these stories for testing GPT-4, leaving out the one story that uses true/false \nquestions rather than yes/no questions. We ran each of the 11 stories through Copilot GPT -4 on 3 \nMarch 2024, preserving the a nswers given to each of the eight questions per story (10). Every \nanswer was printed out well within the five seconds allowed for answers in the human experiment \n(12). An example of GPT -4’s responses to the 8 questions for the story in Appendix A is presented \nin Appendix B. This story is chosen because it had already been posted as an example in an article \ndescribing a human study of discourse comprehension (12). \nIn our first experiment, we use two extra prompts for GPT -4. One prompt precedes the \nstory: Read this story in preparation for answering eight yes/no questions about the story . The \nother prompt follows the story: Answer each of these yes/no questions about the story . Each story \nis itself a necessary prompt. \nIn a follow -up experiment run through Copilot GPT -4 on 2 April 2024 , we instead use a \nprompt to summarize the story and mention main ideas not stated in the story: Summarize this \nstory, mentioning main ideas that are not stated and must be inferred . \nIn our first experiment, w e test GPT -4’s ability to understand brief stories with yes/no \nquestions structured to manipulate the salience and directness of parts of a story. Each of the 88 \nanswers (8 answers per 11 stories ) is categorized as correct , wrong , or unsure . An answer is correct \nif it matches the design ated correct answer (yes or no) (10). Unlike the human participants , who \napparently always conformed to answering only yes or no in their experiment (12), GPT-4 \noccasionally hedge s by providing a neutral answer . Here is an exhaustive list of these neutral \nanswers in our experiment: The story does not specify …, not specified , not mentioned , or The \n4 story does not provide information on this . For these hedged cases, we score the answer’s \ncorrectness as .5 because it is approximately midway between correct (coded 1) and wrong (coded \n0). None of these answers merit s a score of 0, because each of the six incorrect answers are hedged ; \nthey are uncertain rather than being correct or wrong . For completeness, we also alternatively score \nhedged responses as 0, rather than .5. \n2.2 Results"
                },
                {
                "title": "Results",
                "first_sentence": "Because there are considerably more data points in the human sample (5 stories x 8 questions x 40 participants = 1600),",
                "sub_sections": [],
                "text": "Because there are considerably more data points in the human sample (5 stories x 8 questions x 40 \nparticipants = 1600) , we compare a single GPT-4 performance to human performance in terms of \nproportion of correct answers . Proportions correct in the human control sample are computed from \nTable 2 in the human study (12). Our Table 1 presents summary results for humans vs. GPT-4 with \neach of the two scoring methods for hedged responses. Although GPT -4 does very slightly better \nthan humans for each of the two scoring methods, both differences are far below statistical \nsignificance. For the statistical tests in this section , we use the Two Sample Independent \nProportions Test Calculator at Purdue University , a binomial test available online requiring input \nof sample size and successes for each of the two types of participants (humans and GPT -4). \n\nTable 1: Comparison of two scoring methods for GPT -4 to human proportions correct \nover all questions. \nHumans GPT-4 .5 hedge GPT-4 0 hedge \nSample size 1600 88 88 \nSuccesses 1489 85 82 \nProportion .9305 .9659 .9318 \nZ 1.2841 .0429 \np .1991 .9658 \nNote: hedged responses are scored as .5 or 0 in GPT -4. \nFigure 1 shows the proportions correct in each of the four cells of the experiment (2 \ndirectness levels x 2 salience levels ) for humans on the left and GPT -4 on the right. The overall \npattern of proportions correct on the Discourse Comprehension Test (10) for GPT -4 closely \nresembles that for humans. Average neurotypical humans do very well on this test (12) while GPT- \n4 slightly exceeds human performance overall and in three of the four experimental cells portrayed \nin Figure 1 . The pattern of proportions correct are roughly similar for humans and GPT -4 across \nthe four experimental cells. Notably, the worst -performing cell for both humans and GPT -4 is the \nimplied details cell. \n5 \nFigure 1. Proportions correct on the Discourse Comprehension Test for humans on the left and \nGPT-4 on the right, as a function of directness and salience of information . \nFor completeness, we assess whether humans and GPT -4 are performing better than \nchance , again using the Two Sample Independent Proportions Test . Here , chance performance is \ndefined by .5 of sample sizes . The Z and p values in Table 2 certif y that both neurotypical humans \nand GPT-4 models indeed perform well above chance. \nBecause of the theoretical interest in understanding of discourse via implication that goes \nbeyond stated information, we compare GPT -4 to humans on stated -information questions (Table \n3) and implied -information questions (Table 4). These comparisons use t he slightly preferred \nscoring scheme that rates hedged responses as worth .5, as in Figure 1. Again, although GPT -4 \ndoes slightly better than humans on both stated and implied question information, the differences \nin each case are far from reaching statist ical significance. \nTable 2: Comparison of human and GPT -4 performance to chance , defined as .5 success . \nHumans GPT-4 .5 hedge GPT-4 0 hedge \nSample size 1600 88 88 \nSuccesses 800 44 44 \nProportion .9305 .9659 .9318 \nZ 26.99 6.985 6.351 \np 0.0000 0.0000 0.0000 \n\nTable 3: Comparison of proportions correct on stated -information questions. \nHumans GPT-4 \nSample size 800 44 \nSuccesses 770 44 \nProportion .9625 1 \nZ 1.3080 \n\n6 p .1909 \n\nTable 4: Comparison of proportions correct on implied -information questions. \nHumans GPT-4 \nSample size 800 44 \nSuccesses 724 41 \nProportion .9050 .9315 \nZ .5946 \np .5521 \n\nIt is telling that GPT -4 never m akes a wrong response in this experiment. As no ted, it fails \nto give a yes or no response only 6 times out of 88, once on an implied main idea and five times \non implied details. It hedge s on each of these six cases, instead giving neutral uncertain response s \nand appropriate comments that justify their uncertainty . \n\nWe also experiment with GPT -4’s ability to summarize these stories, finding that they \nproduce a concise and accurate paragraph without much in the way of inferences. However, i f we \nask for a summary that mentions inferences, this opens the inferential floodgates . With that prompt, \nGPT-4 produces a total of 54 new inferences that go well beyond those used in the yes/no \nquestions. The mean number of such inferences per story is 4.91, with a standard deviation of 2.02, \nand a range of 2 to 8. An example is provided in Appendix C, using the story presented in Appendix \nA. \n\n2.3 Discussion"
                },
                {
                "title": "Discussion",
                "first_sentence": "Our results show that GPT -4 matches and even slightly exceeds the high level of human performance on the Discourse Comprehension Test (10).",
                "sub_sections": [],
                "text": "Our results show that GPT -4 matches and even slightly exceeds the high level of human \nperformance on the Discourse Comprehension Test (10). Due to excellent human performance , \nthere is very little room for GPT -4 to exhibit superiority over humans . \nIt makes sense that the worst performance in both humans and GPT -4 is in the experiment \ncell for details and implied knowledge. With memory constraints, details may be ignored or \nforgotten in favor of main points. And producing implications requires additional cognitive effort . \nWe encourage readers to carefully consider the example story presented throughout \nAppendi ces A, B, and C. The combination of never giving a wrong answer while spontaneously \nproviding explanatory justifications makes it hard to believe that a story is not well understood by \nGPT-4. The same impression is given by GPT -4’s spontaneous comments about questions in each \nof the other ten stories. \n\nWe are unable to suppress hedging and comments from GPT -4. Its comments on this task \nare both appropriate and interesting, often justifying a yes-or-no answer and sometimes referring \nprecisely to the process of implication. Number of comments across the eleven stories range from \n7 0-8, with a mean of 3.27. Only one story generated no comments. Human comments were not \nrecorded beyond their yes/no responses (12). \n\nGPT-4’s strong overall performance on these novel stories suggests that it indeed \nunderstand s what it has just learned in a single shot , even when that requires inferencing beyond \nwhat is directly stated in the story . \n\nBecause inferences are required to comprehend most if not all discourses (13), it is very \nlikely that there is already considerable evidence in the LLM literature that GPT -4 uses inference \nin understanding what it reads (3,14) . What is unique about our study is the deliberate experimental \nseparation of salience and directness. This enables focus ing more precisely on how these two \nimportant variables operate and interact. Fortuitously, the existence and use of the Discourse \nComprehension Test provides data allowing a close human comparison while maintaining this \nclear separation between important variables on the same content. \n\nClassical psychology experiments on discourse comprehension typically gave participants \na paragraph to read and then asked them to write down what they remembered of the paragraph \n(15,16) . The experimenter would then count the number of correctly recalled propositions as a \nmeasure of understanding. For several reasons, t his methodology did not provide many interesting \ninsights into discourse comprehension . It confounded understanding with memory, made no \ndistinction between stated and implied information , and generally ignored the important role of \ninference based on general knowledge. In contrast, the Discourse Comprehension Test (10) \nseparates direct from implied information and GPT -4 supplies extensive general world knowledge \nthat promotes interesting and useful inferences . \n\nA close analog to asking a human participant to write out a remembered paragraph is to ask \nGPT-4 to summarize what it has just read. This results in a very concise summary with little or no \nhint of inferencing. However , as noted in the 2.2 Results section , when we request GPT -4 to \nmention inferences to accompany their concise story summary, we discover that it provid es many \ninferences that go well beyond the modest inferencing apparent in our first experiment with yes/no \nquestions. It might be interesting to see whether human participants would likewise provide \nadditional inferences if similarly prompted in this task. \n3. Understanding More Difficult Passages"
                }
            ]
            },
            {
            "title": "Understanding More Difficult Passages",
            "first_sentence": "The lack of statistically significant differences between humans and GPT -4 in section 2 could be due to the relative simplicity of the stories used in the Discourse Comprehension Test (10).",
            "text": "The lack of statistically significant differences between humans and GPT -4 in section 2 could be \ndue to the relative simplicity of the stories used in the Discourse Comprehension Test (10). Both \nclasses of participants performed at a sufficiently high level that there was very little room for one \ntype to statistically exceed the performance of the other type. Our preliminary conclusion is that \nGPT-4 at least matched human performance on discourse comprehension. Here i n section 3, we \nuse considerably more difficult reading material , to allow greater possible distance between \nhumans and GPT -4 in one direction or the other . \n3.1 Overall Test Results \nLarge teams of OpenAI researchers recently published an extensive and detailed Technical \nReport on the capabilities, limitations, and safety characteristics of GPT -4 (17). Among the \ncapabilities that they addressed were performances on 34 academic tests covering a wide range of \n8 fields. Three of these academic tests had sections that addressed reading comprehension at higher \nlevels of difficulty than the Discourse Comprehension Test used in our section 2: SAT, GRE, and \nLSAT. \nOur section 3.1 is a review of GPT -4 performance on these three widely used and highly \nstandardized academic tests (17). They each have a large component devoted to reading \ncomprehension . OpenAI researchers verified that there was no special GPT -4 training for these \nthree tests, and they also ran contamination checks for test data appearing in the training set (17). \nIf matches to the test set were found in the training set, they were removed from the test set to \ncreate an uncontaminated test set. \nTable 5 shows the percentile achieved by GPT -4 in each test after eliminating any \ncontamination from the training set. The mean uncontaminated percentile across the three tests is \n96.3. By statistical definition, the average percentile achieved by thousands of student test -takers \nis the 50th percentile , thus revealing a substantial superiority for GPT -4 with reading \ncomprehension of difficult passages. The prompts given to GPT -4 reflected the test requirements \n(17). \nTable 5. GPT -4 Uncontaminated Percentile Scores on 3 Academic Tests that include Reading \nComprehension . \nTest Percentile \nScholastic Aptitude Test (SAT) Reading & Writing 93rd \nGraduate Record Examination (GRE) Verbal 99th \nLaw School Admission Test (LSAT) 97th \nAdapted from OpenAI 2024, their Table 9 in their Appendix G. (17) \nFor the SAT and GRE Verbal exams, scores were identical with and without contamination, \nwhile for the LSAT, GPT -4 performed slightly better on uncontaminated questions. This finding \nsupports OpenAI’s conclusion that contamination had little to no effect on GPT -4's scores and \nsuggests that GPT -4’s high scores reflect its reading comprehension abilities rather than specific \nmemorized content from training data (17). \nThe SAT is widely used for college admissions in North America. The Reading section has \nbrief passages (or a passage pair) followed by a multiple -choice question. Passages range from 25 \nto 150 words. The subject areas for Reading and Writing cover literature, history , social studies, \nhumanities, and science. Students have 64 minutes to complete the Reading and Writing section . \nReading Comprehension questions on the GRE are designed to test for the abilit y to \nunderstand the kinds of prose commonly encountered in graduate and professional schoo ls, \nincluding drawing conclusions from information , reasoning from incomplete data to infer \nmissing information , understanding how the parts of a passage relate to each other, analyzing a \ntext and reaching its conclusions , considering alternative explanations , and formulating and testing \nhypotheses. Test passages are borrowed from academic and non -academic books and articles \ncovering science, arts , humanities, business , and everyday topics. \n9 Reading comprehension passages and questions on the LSAT seem particularly well suited \nto discovering indications of true understanding as they often requi re the reader to reason beyond \nthe literal text. Their m ultiple -choice questions probe for main ideas, explicitly stated information, \ninferable information, generalization to different contexts, and analogizing. \n3.2 Other Signatures of Understanding \nAlthough there are no precise experimental distinctions in these academic tests between stated and \ninferred information and between main points and details, as in The Discourse Comprehension \nTest (10), it is still possible to identify important signatures of text understanding such as \ngeneralization and inference. Our next step was to probe this more deeply by running a GPT -4 \nexperiment with online available LSAT passages which were accompanied by explanation s for the \ncorrectness and incorrectness of multiple -choice responses . Human students could read the \ncorrectness information for each multiple -choice answer as a useful pre-test study guide, while we \ninstead prompt GPT-4 to provide justification for each of its answers in the test . Providing \njustifications is quite different and more challenging than the OpenAI testing which more closely \nmatched the testing conditions for LSAT student test takers where justifications were not requested \n(17). \n3.3 Method for probing LSAT performance more deeply \nOur simulation experiment presents GPT -4 with three single passages and a pair of two related \npassages. For each of th ese four events, the generic prompt to GPT-4 is “ Read this passage and \nthen answer the two multiple -choice questions that follow. Also justify your answer to each \nquestion. ” The n umber of questions mentioned in the prompt varies from two to eight. The four \ntest passages concern Roy Lichtenstein’s pop art (3 questions), indigenous rights in Canada (2 \nquestions), a n alleged speculative bubble in tulip prices in the seventeenth -century Dutch tulip \nmarket (2 questions), and the extent of human involvement in climate change ( pair of two passages, \n7 questions). This LSAT prep test was arbitrarily chosen from several such LSAT prep tests \navailable online. The simulation was performed with Copilot GPT -4 on 31 May 2024. \n3.4 Resu lts \nThe result is that GPT -4 gets all fourteen questions correct , approximately consistent with \nOpenAI’s 97th percentile GPT-4 performance on entire LSAT s (17). To examine GPT-4’s cognitive \nperfo rmance in more detail , we display here the speculative -bubble passage as a single , but \nrepresentative , example . In blue font are the prompt , passage , and questions . We encourage our \nreaders to read this passage and then quickly answer multiple -choice questions 6 and 7 before \nreading the answers and explanations supplied by GPT -4 (below) or the test -maker (in Appendix \nD). This would provide a good idea of what the students and GPT -4 were up against in the LSAT. \nRead this passage and then answer the two multiple -choice questions that follow. Also justify your \nanswer to each question. \nIn economics, the term “speculative bubble” refers to a large upward move in an asset’s price \ndriven not by the asset’s fundamentals —that is, by the earnings derivable from the asset —but \nrather by mere speculation that someone else will be willing to pay a higher price for it. The price \nincrease is then followed by a dramatic decline in price, due to a loss in confidence that the price \nwill continue to rise, and the “bubble” is said to have burst. According to Charles Mackay’s classic \nnineteenth -century acc ount, the seventeenth -century Dutch tulip market provides an example of a \n10 speculative bubble. But the economist Peter Garber challenges Mackay’s view, arguing that there \nis no evidence that the Dutch tulip market really involved a speculative bubble. \nBy the seventeenth century, the Netherlands had become a center of cultivation and development \nof new tulip varieties, and a market had developed in which rare varieties of bulbs sold at high \nprices. For example, a Semper Augustus bulb sold in 1625 for an amount of gold worth about U.S. \n$11,000 in 1999. Common bulb varieties, on the other hand, sold for very low prices. According \nto Mackay, by 1636 rapid price rises attracted speculators, and prices of many varieties surged \nupward from November 1636 through January 1637. Mackay further states that in February 1637 \nprices suddenly collapsed; bulbs could not be sold at 10 percent of their peak values. By 1739, the \nprices of all the most prized kinds of bulbs had fallen to no more than one two -hundredth of 1 \npercent of Semper Augustus’s peak price. \nGarber acknowledges that bulb prices increased dramatically from 1636 to 1637 and eventually \nreached very low levels. But he argues that this episode should not be described as a speculative \nbubble, for the increase and eventual decline in bulb prices can be explained in terms of the \nfundamentals. Garber argues that a standard pricing pattern occurs for new varieties of flowers. \nWhen a particularly prized variety is developed, its original bulb sells for a high price. Thus, the \ndramatic rise in the price of some original tulip bulbs could have resulted as tulips in general, and \ncertain varieties in particular, became fashionable. However, as the prized bulbs become more \nreadily available through reproduction from the original bulb, their price falls rapidly; after less \nthan 30 years, bulbs sell at reproduction cost. But this does not mean that the high prices of original \nbulbs are irrational, for earnings derivable from the millions of bulbs descendent from the original \nbulbs can be very high, even if each in dividual descendent bulb commands a very low price. Given \nthat an original bulb can generate a reasonable return on investment even if the price of descendent \nbulbs decreases dramatically, a rapid rise and eventual fall of tulip bulb prices need not indica te a \nspeculative bubble. \nQuestion 6 \nThe phrase “standard pricing pattern” as used in the middle of the last paragraph most nearly means \na pricing pattern \nA. against which other pricing patterns are to be measured \nB. that conforms to a commonly agreed -upon criterion \nC. that is merely acceptable \nD. that regularly recurs in certain types of cases \nE. that serves as an exemplar \n\nQuestion 7 \n\nGiven Garber’s account of the seventeenth -century Dutch tulip market, which one of the following \nis most analogous to someone who bought a tulip bulb of a certain variety in that market at a very \nhigh price, only to sell a bulb of that variety at a much lo wer price? \n11 A. someone who, after learning that many others had withdrawn their applications for a \nparticular job, applied for the job in the belief that there would be less competition for it \nB. an art dealer who, after paying a very high price for a new painting, sells it at a very low \nprice because it is now considered to be an inferior work \nC. someone who, after buying a box of rare motorcycle parts at a very high price, is forced to \nsell them at a much lower price because of the sudden availability of cheap substitute parts \nD. a publisher who pays an extremely high price for a new novel only to sell copies at a price \naffordable to nearly everyone \nE. an airline that, after selling most of the tickets for seats on a plane at a very high price, must \nsell the remaining tickets at a very low price \nHere are GPT -4’s responses , also in blue font : \nQuestion 6 The correct answer is D. that regularly recurs in certain types of cases . The passage \nexplains that Garber argues a “standard pricing pattern” occurs for new varieties of flowers, where \nthe price of a particularly prized variety is initially high but falls rapidly as the bulbs become more \nreadily available through reproduction . This pattern is not described as an exemplar (E), a \ncommonly agreed -upon criterion (B), or a measure against which other patterns are compared (A). \nIt’s also not described as merely acceptable (C). Instead, it’s a pattern that regularly recurs in th e \nspecific case of new flower varieties. \nQuestion 7 The correct answer is D. a publisher who pays an extremely high price for a new \nnovel only to sell copies at a price affordable to nearly everyone . This is analogous to buying \na tulip bulb of a certain variety at a very high price and then selling bulbs of that variety at a much \nlower price, as the original bulb can generate many descendent bulbs. Just like a publisher who \npays a high price for a ne w novel (the original “bulb”) and then sells many copies (the “descendent \nbulbs”) at a lower price. The other options do not capture this dynamic of an initial high -cost \ninvestment that can generate many lower -cost items. \nThese justifications from GPT -4 suggest that it is , among other things , reasoning by \ngeneralization to other examples of price changes that conform to the standard pricing pattern of \ninvesting a lot in what people want and eventual selling it in high volume at a low price. For \nexample, a publishing company pays a lot for a book which can then be sold to many readers at a \nmuch lower price. Such generalization strengthens explanations that apply more widely and are \nmore correct and more useful. Generalizatio n is a standard inferential signature of understanding \ntextual discourse. \nGeneralization was also evident in two other of our four passages: pop art and climate \nchange. There is a question about Lichtenstein’s motivation for doing pop art in the way he did. \nBecause motivation is not mentioned in the passage, the reader must generalize across the author’s \ncomments about Lichtenstein’s approach to art, his rebellion against abstract expressionism, his \nincorporation of realism and naiveté, and his depiction of contemporary life . \nIn the two climate -change passages, global warming is attributed to human activities in \npassage A and to natural cycles in Passage B. In each of the two passages, there is a generalized \nexplanation of several contemporary extreme weather phenomena , pitting one generalized \nexplanation against another and enabling correct answers to several questions . \n12 In the passage on indigenous rights in Canada, there is an important inference that \nindigenous oral tradition is necessary for establishing legal documentation of land claims . This is \nbecause the only cultural customs being considered occurred prior to the establishment of British \nsovereignty over the specific territor ies. Relevant written documentation would be come available \nonly after colonization . \nWe noticed in this experiment that GPT-4 explains its responses in far fewer words than \nthe test maker s used to explain the correctness of answers to students trying to ace the ir test. The \ntest-maker explanations are available in Appendix D, where r eaders can judge for themselves \nwhich explanations they would prefer. We prefer the more concise explanations given by GPT -4 \nbecause they are just as informative and easier to understand compared to the wordier test-maker \nexplanations . Using too many words in an explanation stresses memory and ma kes understanding \nmore difficult. As we noted in the introduction, making good summaries of text is considered an \nindicator of understanding. \nMany of the words used in test -maker explanations were used to explain why each of the \nfour wrong answers were not correct. Even if we remove all the words addressing wrong answers \nfrom the statistical analysis, there is still a very strong tendency for GP T-4’s explanations to be \nmore concise than those of the test -makers, t(13) = 7.48, p < .0001, as shown in Figure 2. The 95% \nconfidence interval of the mean difference is from 102.81 to 186.34. \n\n\nFigure 2: Number of words used to explain answer correctness. \nIt is likely that some of the extra words used by the test makers are useful in some way. \nNonetheless, it is clear from these data that the more concise justifications provided by GPT -4 are \nsufficient to support perfect performance on this test by GPT -4, and that the more verbose \nexplanations supplied by the test makers are not sufficient to produce perfect performance in \nhuman test takers, as the average human performance on LSAT s falls far below the 100th percentile \nachieved here by GPT -4. \n4. General Discussion",
            "sub_sections": [
                {
                "title": "Overall Test Results",
                "first_sentence": "Large teams of OpenAI researchers recently published an extensive and detailed Technical Report on the capabilities, limitations, and safety characteristics of GPT -4 (17).",
                "sub_sections": [],
                "text": "Large teams of OpenAI researchers recently published an extensive and detailed Technical \nReport on the capabilities, limitations, and safety characteristics of GPT -4 (17). Among the \ncapabilities that they addressed were performances on 34 academic tests covering a wide range of \n8 fields. Three of these academic tests had sections that addressed reading comprehension at higher \nlevels of difficulty than the Discourse Comprehension Test used in our section 2: SAT, GRE, and \nLSAT. \nOur section 3.1 is a review of GPT -4 performance on these three widely used and highly \nstandardized academic tests (17). They each have a large component devoted to reading \ncomprehension . OpenAI researchers verified that there was no special GPT -4 training for these \nthree tests, and they also ran contamination checks for test data appearing in the training set (17). \nIf matches to the test set were found in the training set, they were removed from the test set to \ncreate an uncontaminated test set. \nTable 5 shows the percentile achieved by GPT -4 in each test after eliminating any \ncontamination from the training set. The mean uncontaminated percentile across the three tests is \n96.3. By statistical definition, the average percentile achieved by thousands of student test -takers \nis the 50th percentile , thus revealing a substantial superiority for GPT -4 with reading \ncomprehension of difficult passages. The prompts given to GPT -4 reflected the test requirements \n(17). \nTable 5. GPT -4 Uncontaminated Percentile Scores on 3 Academic Tests that include Reading \nComprehension . \nTest Percentile \nScholastic Aptitude Test (SAT) Reading & Writing 93rd \nGraduate Record Examination (GRE) Verbal 99th \nLaw School Admission Test (LSAT) 97th \nAdapted from OpenAI 2024, their Table 9 in their Appendix G. (17) \nFor the SAT and GRE Verbal exams, scores were identical with and without contamination, \nwhile for the LSAT, GPT -4 performed slightly better on uncontaminated questions. This finding \nsupports OpenAI’s conclusion that contamination had little to no effect on GPT -4's scores and \nsuggests that GPT -4’s high scores reflect its reading comprehension abilities rather than specific \nmemorized content from training data (17). \nThe SAT is widely used for college admissions in North America. The Reading section has \nbrief passages (or a passage pair) followed by a multiple -choice question. Passages range from 25 \nto 150 words. The subject areas for Reading and Writing cover literature, history , social studies, \nhumanities, and science. Students have 64 minutes to complete the Reading and Writing section . \nReading Comprehension questions on the GRE are designed to test for the abilit y to \nunderstand the kinds of prose commonly encountered in graduate and professional schoo ls, \nincluding drawing conclusions from information , reasoning from incomplete data to infer \nmissing information , understanding how the parts of a passage relate to each other, analyzing a \ntext and reaching its conclusions , considering alternative explanations , and formulating and testing \nhypotheses. Test passages are borrowed from academic and non -academic books and articles \ncovering science, arts , humanities, business , and everyday topics. \n9 Reading comprehension passages and questions on the LSAT seem particularly well suited \nto discovering indications of true understanding as they often requi re the reader to reason beyond \nthe literal text. Their m ultiple -choice questions probe for main ideas, explicitly stated information, \ninferable information, generalization to different contexts, and analogizing. \n3.2 Other Signatures of Understanding"
                },
                {
                "title": "Other Signatures of Understanding",
                "first_sentence": "Although there are no precise experimental distinctions in these academic tests between stated and inferred information and between main points and details, as in The Discourse Comprehension Test (10), it is still possible to identify important signatures of text understanding such as generalization and inference.",
                "sub_sections": [],
                "text": "Although there are no precise experimental distinctions in these academic tests between stated and \ninferred information and between main points and details, as in The Discourse Comprehension \nTest (10), it is still possible to identify important signatures of text understanding such as \ngeneralization and inference. Our next step was to probe this more deeply by running a GPT -4 \nexperiment with online available LSAT passages which were accompanied by explanation s for the \ncorrectness and incorrectness of multiple -choice responses . Human students could read the \ncorrectness information for each multiple -choice answer as a useful pre-test study guide, while we \ninstead prompt GPT-4 to provide justification for each of its answers in the test . Providing \njustifications is quite different and more challenging than the OpenAI testing which more closely \nmatched the testing conditions for LSAT student test takers where justifications were not requested \n(17). \n3.3 Method for probing LSAT performance more deeply"
                },
                {
                "title": "Method for probing LSAT performance more deeply",
                "first_sentence": "Our simulation experiment presents GPT -4 with three single passages and a pair of two related passages.",
                "sub_sections": [],
                "text": "Our simulation experiment presents GPT -4 with three single passages and a pair of two related \npassages. For each of th ese four events, the generic prompt to GPT-4 is “ Read this passage and \nthen answer the two multiple -choice questions that follow. Also justify your answer to each \nquestion. ” The n umber of questions mentioned in the prompt varies from two to eight. The four \ntest passages concern Roy Lichtenstein’s pop art (3 questions), indigenous rights in Canada (2 \nquestions), a n alleged speculative bubble in tulip prices in the seventeenth -century Dutch tulip \nmarket (2 questions), and the extent of human involvement in climate change ( pair of two passages, \n7 questions). This LSAT prep test was arbitrarily chosen from several such LSAT prep tests \navailable online. The simulation was performed with Copilot GPT -4 on 31 May 2024. \n3.4 Resu lts"
                },
                {
                "title": "Results",
                "first_sentence": "The result is that GPT -4 gets all fourteen questions correct, approximately consistent with OpenAI’s 97th percentile GPT-4 performance on entire LSATs (17).",
                "sub_sections": [],
                "text": "The result is that GPT -4 gets all fourteen questions correct , approximately consistent with \nOpenAI’s 97th percentile GPT-4 performance on entire LSAT s (17). To examine GPT-4’s cognitive \nperfo rmance in more detail , we display here the speculative -bubble passage as a single , but \nrepresentative , example . In blue font are the prompt , passage , and questions . We encourage our \nreaders to read this passage and then quickly answer multiple -choice questions 6 and 7 before \nreading the answers and explanations supplied by GPT -4 (below) or the test -maker (in Appendix \nD). This would provide a good idea of what the students and GPT -4 were up against in the LSAT. \nRead this passage and then answer the two multiple -choice questions that follow. Also justify your \nanswer to each question. \nIn economics, the term “speculative bubble” refers to a large upward move in an asset’s price \ndriven not by the asset’s fundamentals —that is, by the earnings derivable from the asset —but \nrather by mere speculation that someone else will be willing to pay a higher price for it. The price \nincrease is then followed by a dramatic decline in price, due to a loss in confidence that the price \nwill continue to rise, and the “bubble” is said to have burst. According to Charles Mackay’s classic \nnineteenth -century acc ount, the seventeenth -century Dutch tulip market provides an example of a \n10 speculative bubble. But the economist Peter Garber challenges Mackay’s view, arguing that there \nis no evidence that the Dutch tulip market really involved a speculative bubble. \nBy the seventeenth century, the Netherlands had become a center of cultivation and development \nof new tulip varieties, and a market had developed in which rare varieties of bulbs sold at high \nprices. For example, a Semper Augustus bulb sold in 1625 for an amount of gold worth about U.S. \n$11,000 in 1999. Common bulb varieties, on the other hand, sold for very low prices. According \nto Mackay, by 1636 rapid price rises attracted speculators, and prices of many varieties surged \nupward from November 1636 through January 1637. Mackay further states that in February 1637 \nprices suddenly collapsed; bulbs could not be sold at 10 percent of their peak values. By 1739, the \nprices of all the most prized kinds of bulbs had fallen to no more than one two -hundredth of 1 \npercent of Semper Augustus’s peak price. \nGarber acknowledges that bulb prices increased dramatically from 1636 to 1637 and eventually \nreached very low levels. But he argues that this episode should not be described as a speculative \nbubble, for the increase and eventual decline in bulb prices can be explained in terms of the \nfundamentals. Garber argues that a standard pricing pattern occurs for new varieties of flowers. \nWhen a particularly prized variety is developed, its original bulb sells for a high price. Thus, the \ndramatic rise in the price of some original tulip bulbs could have resulted as tulips in general, and \ncertain varieties in particular, became fashionable. However, as the prized bulbs become more \nreadily available through reproduction from the original bulb, their price falls rapidly; after less \nthan 30 years, bulbs sell at reproduction cost. But this does not mean that the high prices of original \nbulbs are irrational, for earnings derivable from the millions of bulbs descendent from the original \nbulbs can be very high, even if each in dividual descendent bulb commands a very low price. Given \nthat an original bulb can generate a reasonable return on investment even if the price of descendent \nbulbs decreases dramatically, a rapid rise and eventual fall of tulip bulb prices need not indica te a \nspeculative bubble. \nQuestion 6 \nThe phrase “standard pricing pattern” as used in the middle of the last paragraph most nearly means \na pricing pattern \nA. against which other pricing patterns are to be measured \nB. that conforms to a commonly agreed -upon criterion \nC. that is merely acceptable \nD. that regularly recurs in certain types of cases \nE. that serves as an exemplar \n\nQuestion 7 \n\nGiven Garber’s account of the seventeenth -century Dutch tulip market, which one of the following \nis most analogous to someone who bought a tulip bulb of a certain variety in that market at a very \nhigh price, only to sell a bulb of that variety at a much lo wer price? \n11 A. someone who, after learning that many others had withdrawn their applications for a \nparticular job, applied for the job in the belief that there would be less competition for it \nB. an art dealer who, after paying a very high price for a new painting, sells it at a very low \nprice because it is now considered to be an inferior work \nC. someone who, after buying a box of rare motorcycle parts at a very high price, is forced to \nsell them at a much lower price because of the sudden availability of cheap substitute parts \nD. a publisher who pays an extremely high price for a new novel only to sell copies at a price \naffordable to nearly everyone \nE. an airline that, after selling most of the tickets for seats on a plane at a very high price, must \nsell the remaining tickets at a very low price \nHere are GPT -4’s responses , also in blue font : \nQuestion 6 The correct answer is D. that regularly recurs in certain types of cases . The passage \nexplains that Garber argues a “standard pricing pattern” occurs for new varieties of flowers, where \nthe price of a particularly prized variety is initially high but falls rapidly as the bulbs become more \nreadily available through reproduction . This pattern is not described as an exemplar (E), a \ncommonly agreed -upon criterion (B), or a measure against which other patterns are compared (A). \nIt’s also not described as merely acceptable (C). Instead, it’s a pattern that regularly recurs in th e \nspecific case of new flower varieties. \nQuestion 7 The correct answer is D. a publisher who pays an extremely high price for a new \nnovel only to sell copies at a price affordable to nearly everyone . This is analogous to buying \na tulip bulb of a certain variety at a very high price and then selling bulbs of that variety at a much \nlower price, as the original bulb can generate many descendent bulbs. Just like a publisher who \npays a high price for a ne w novel (the original “bulb”) and then sells many copies (the “descendent \nbulbs”) at a lower price. The other options do not capture this dynamic of an initial high -cost \ninvestment that can generate many lower -cost items. \nThese justifications from GPT -4 suggest that it is , among other things , reasoning by \ngeneralization to other examples of price changes that conform to the standard pricing pattern of \ninvesting a lot in what people want and eventual selling it in high volume at a low price. For \nexample, a publishing company pays a lot for a book which can then be sold to many readers at a \nmuch lower price. Such generalization strengthens explanations that apply more widely and are \nmore correct and more useful. Generalizatio n is a standard inferential signature of understanding \ntextual discourse. \nGeneralization was also evident in two other of our four passages: pop art and climate \nchange. There is a question about Lichtenstein’s motivation for doing pop art in the way he did. \nBecause motivation is not mentioned in the passage, the reader must generalize across the author’s \ncomments about Lichtenstein’s approach to art, his rebellion against abstract expressionism, his \nincorporation of realism and naiveté, and his depiction of contemporary life . \nIn the two climate -change passages, global warming is attributed to human activities in \npassage A and to natural cycles in Passage B. In each of the two passages, there is a generalized \nexplanation of several contemporary extreme weather phenomena , pitting one generalized \nexplanation against another and enabling correct answers to several questions . \n12 In the passage on indigenous rights in Canada, there is an important inference that \nindigenous oral tradition is necessary for establishing legal documentation of land claims . This is \nbecause the only cultural customs being considered occurred prior to the establishment of British \nsovereignty over the specific territor ies. Relevant written documentation would be come available \nonly after colonization . \nWe noticed in this experiment that GPT-4 explains its responses in far fewer words than \nthe test maker s used to explain the correctness of answers to students trying to ace the ir test. The \ntest-maker explanations are available in Appendix D, where r eaders can judge for themselves \nwhich explanations they would prefer. We prefer the more concise explanations given by GPT -4 \nbecause they are just as informative and easier to understand compared to the wordier test-maker \nexplanations . Using too many words in an explanation stresses memory and ma kes understanding \nmore difficult. As we noted in the introduction, making good summaries of text is considered an \nindicator of understanding. \nMany of the words used in test -maker explanations were used to explain why each of the \nfour wrong answers were not correct. Even if we remove all the words addressing wrong answers \nfrom the statistical analysis, there is still a very strong tendency for GP T-4’s explanations to be \nmore concise than those of the test -makers, t(13) = 7.48, p < .0001, as shown in Figure 2. The 95% \nconfidence interval of the mean difference is from 102.81 to 186.34. \n\n\nFigure 2: Number of words used to explain answer correctness. \nIt is likely that some of the extra words used by the test makers are useful in some way. \nNonetheless, it is clear from these data that the more concise justifications provided by GPT -4 are \nsufficient to support perfect performance on this test by GPT -4, and that the more verbose \nexplanations supplied by the test makers are not sufficient to produce perfect performance in \nhuman test takers, as the average human performance on LSAT s falls far below the 100th percentile \nachieved here by GPT -4. \n4. General Discussion"
                }
            ]
            },
            {
            "title": "General Discussion",
            "first_sentence": "We report in section 2 that GPT -4 matches the performance of average adult humans on the Discourse Comprehension Test (10).",
            "text": "13 We report in section 2 that GPT -4 matches the performance of average adult humans on the \nDiscourse Comprehension Test (10). This is notable on its own, but there is more to say about this \nphenomenon. Because the stories in that test are rather simple (5th and 6th grade reading levels) , \nboth humans and GPT -4 perform at a very high level. This raises the possibility that there is not \nsufficient room for one type of participant to perform at a higher level than the other type. We find \nin section 3.1 that increasing the difficulty of the text enables greater separation of the two subject \ntypes . GPT-4 here performs at a much higher level than do the humans for whom the more difficult \ntests were designed , i.e., highly motivated students striving to extend their education by doing well \non admi ssion tests . Performance di fferences on these more difficult passages and test questions are \nin the general neighborhood of 2:1 in favo ur of GPT -4 on the percentile scale . This provides \nsubstantial evidence that increasing the difficulty of text passages creates a strong interaction with \nparticipant type. Average h umans do about as well with understanding simple text as does GPT -4, \nbut GPT -4 can greatly exceed the performance of intelligent humans on more difficult passage s. \nOur converging evidence for genuine understanding of text in GPT -4 is strong due to a high \nlevel of correct responding to questions about the text , correct and concise justifications for those \nanswers , and subtle signatures of understanding that go beyond the literal text such as \ngeneralization inferences and other inference s. It suggests that GPT-4 can exceed human levels of \ntext understanding, provided that the text is sufficiently difficult. \nDoes this mean that GPT-4 has achieved Artificial General Intelligence (AGI)? It does not, \nbecause reading comprehension is only one skill in the AGI sphere, albeit a critically important \nskill because reading is the principal way that an LLM currently receives its knowledge. \nNonetheless , reading comprehension can now be added to the long and growing list of cognitive \nskills on which GPT -4 has matched or exceeded average human performance: decision making \n(6), metaphors, idioms, estimation, anaphoric reference, creativity, problem solving, and \nanswering common sense questions (3). \nHowever, as we noted in section 1 , LLMs still have difficulty with causal reasoning (6), \nplanning (7), and graduate -level mathematics (8). Causal reasoning might well require models of \nagentive interaction with the physical world, involving the control of purpose ful sensorimotor \ninteractions (18). As well, a case could be made that such factors should be employed in AI \nconstruction well before the agents are capable of reading. Even very young infants act as though \nthey are aware that their actions cause certain effects (19,20) . Although infants are far from being \nable to read, their causal knowledge gets firmly grounded in purposely acting on the world to \nproduce desired effects . \nAnother promising approach to causal reasoning involves using LLMs to provide prior \nprobabilit ies to construct causal graphs (21). Such information from LLMs was found to i mprove \nperformance on commonsense causal benchmark phenomena, especially when deciding what is \nthe cause and what is the effect. \nOne thing to keep in mind when considering the high percentiles achieved by GPT -4 on \ndifficult academic tests is that there are a small number of human test -takers who achieved even \nhigher percentiles than achieved by GTP -4 on those tests. It could be interesting to study such \npeople while monitoring the ongoing quest of LLMs striving for perfect success on important \ntasks . We happened to encounter some aspects of this issue in our final experiment using a small \n14-question snippet of an LSAT test on which GPT -4 got every question correct. Some of those \nquestions were flagged as very difficult by test -makers based on their difficulty for humans. \n14 Although our article is presented in a kind of competition between humans and AI, we \nbelieve that LLMs could rapidly become of great service to interested humans. For example, \nhumans could come to rely on yet to be fully developed AI systems that can read widely and \nreliably explain to us what they read and underst and. \nAcknowledgments \nThis research is supported by a grant from the Natural Sciences and Engineering Research Council \nof Canada to TRS: NSERC 2021 -03434. \nReferences \n1. Chomsky, N. The False Promise of ChatGPT. New York Times. 2023 Mar 8; \n2. Marcus, G., Davis, E. Rebooting AI: building artificial intelligence we can trust. New York: \nPantheon Books; 2019. \n3. Sartori, G., Orru, G. Language models and psychological sciences. Frontiers in Psychology. \n2023;14:1 –16. (doi.org/10.3389/fpsyg.2023.1279317 ) \n4. Webb, T., Holyoak, K. & Lu, H. Emergent analogical reasoning in large language models. \nNature Human Behaviour. 2023;7:1526 –41. (doi.org/10.1038/s41562 -023-01659 -w) \n5. Kosinski, M. Theory of mind may have spontaneously emerged in LLMs. Cornell \nUniversity: arXiv; 2023. (2302.02083). \n6. Binz M., & Schulz E. Using cognitive psychology to understand GPT -3. Proceedings of the \nNational Academy of Sciences. 2023;120(6). (doi.org/10.1073/pnas.2218523120 ) \n7. Valmeekam, K., Olmo, A., Sreedharan, S., & Kambhampati, S. LLMs still can’t plan (a \nbenchmark for LLMs on planning and reasoning about change). Cornell University: arXiv; \n2022. (2206.10498). \n8. Frieder S, Pinchetti L, and Griffiths RR, Salvatori T, Lukasiewicz T, Petersen P, et al. \nMathematical Capabilities of ChatGPT. In: Oh A, Naumann T, Globerson A, Saenko K, \nHardt M, Levine S, editors. Advances in Neural Information Processing Systems [Int ernet]. \nCurran Associates, Inc.; 2023. p. 27699 –744. Available from: \nhttps://proceedings.neurips.cc/paper_files/paper/2023/file/58168e8a92994655d6da3939e7cc \n0918 -Paper -Datasets_and_Benchmarks.pdf \n9. Graesser, A., Singer, M., Trabasso, T. Constructing Inferences During Narrative Text \nComprehension. Psychological Review. 1994;101(3):371 –95. \n10. Brookshire RH, Nicholas LE. Discourse Comprehension Test: Test Manual [Internet]. BRK \nPublishers; 1997. Available from: https://books.google.ca/books?id=vAQFzQEACAAJ \n11. Dale, E., & Chall, J. A formula for predicting readability. Educational Research Bulletin. \n1948;27:11 –20. \n15 12. Nicholas, L., Brookshire, R. Comprehension of spoken narrative discourse by adults with \naphasia, right -hemisphere brain damage, or traumatic brain injury. American Journal of \nSpeech -Language Pathology. 1995;4(3):69 –81. \n13. Singer, M. Inference processing in discourse comprehension. In: The Oxford Handbook of \nPsycholinguistics. Oxford, UK: Oxford University Press; 2007. p. 343 –59. \n14. Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., et al. Sparks \nof artificial general intelligence: early experiments with gpt -4. Cornell University: arXiv; \n2023. (2303 -08774v6). \n15. Frederiksen, C. Acquisition of Semantic Information from Discourse: Effects of Repeated \nExposures. Journal of Verbal Learning and Verbal Behavior. 1975;14:158 –69. \n16. Kintsch, W., Kozminsky, E., Streby, W., McKoon, G., Keenan, J. Comprehension and Recall \nof Text as a Function of Content Variables. Journal of Verbal Learning and Verbal Behavior. \n1975;14:196 –214. \n17. OpenAI. GPT -4 Technical Report. Cornell University: arXiv; 2024. \n18. Pezzullo, G., Parr, T., Cisek, P., Clark, A., & Friston, K. Generating meaning: active \ninference and the scope and limits of passive AI. Trends in Cognitive Sciences. \n2024;28(2):97 –112. \n19. Hauf, P., & Aschersleben, G. Action –effect anticipation in infant action control. \nPsychological Research. 2008;72:203 –10. \n20. Rovee, C., & Rovee, D. Conjugate reinforcement of infants’ exploratory behavior. Journal of \nExperimental Child Psychology. 1969;8:33 –9. \n21. Darvariu, V., Hailes, S., & Musolesi, M. Large Language Models are Effective Priors for \nCausal Graph Discovery. Cornell University: arXiv; 2024. (2405.13551v1). \n\n\n16 Appendices \nAppendix A: Example story and its questions, with their correct answers and values of the \nsalience and directness variables. \nAppendix B: Responses of GPT -4 to the story in Appendix A. \nAppendix C: Responses of GPT -4 to the story in Appendix A when prompted for inferences, \nwhich are highlighted by us in square brackets. \nAppendix D: Test -maker explanations of answer correctness in Section 3. \n\n17 Appendix A: Example story and its questions, with their correct answers and values of \nthe salience and directness variables in Section 2. \nRead this story in preparation for answering eight yes/no questions about the story. \nOne day last Fall, several women on Willow Street decided to have a garage sale. They collected \nodds and ends from all over the neighborhood. Then they spent an entire day putting prices on the \nthings that they had collected. On the first day of the sale, they put up signs at both ends of the \nblock and another one at a nearby shopping center. Next, they made a pitcher of iced tea and sat \ndown in a shady spot beside the Anderson’s garage to wait for their first customer. Soon a man \ndrove up in an old truck. He looked around and finally stopped by a lumpy old mattress that was \nleaning against the wall. He gestured to it and asked how much they wanted for it. Mrs. Anderson \ntold him that it wasn’t for sale. Then she added that they were going to put it out for t he trash \ncollectors the next day. The man asked if he could have it. Mrs. Anderson said that he could. Then \nshe asked, “Why do you want such a terrible mattress?” “Well,” he said, “My no -good father -in- \nlaw is coming to visit next week, and I don’t want him to get too comfortable.” \n1. Did several women have a party? No, stated main idea. \n2. Were there a large number of things at the garage sale? Yes, implied main idea. \n3. Did the women put up a sign at a shopping center? Yes, stated detail. \n4. Was it cold the day of the garage sale? No, implied detail. \n5. Was the man driving a car? No, stated detail. \n6. Was the mattress in terrible condition? Yes, stated main idea. \n7. Was the man married? Yes, implied detail. \n8. Was the man fond of his father -in-law? No, implied main idea. \n\n18 Appendix B: Responses of GPT -4 to the story in Appendix 1. \nNote that we add parentheses around the spontaneous elaborations given by GPT -4 beyond its \nyes/no answers, which are printed in bold font by GPT -4. \nAnswer each of these yes/no questions about the story. \n1. Did several women have a party? No \n2. Were there a large number of things at the garage sale? Yes \n3. Did the women put up a sign at a shopping center? Yes \n4. Was it cold the day of the garage sale? No (The story does not provide information about the \ntemperature) \n5. Was the man driving a car? No (He drove up in an old truck) \n6. Was the mattress in terrible condition? Yes \n7. Was the man married? Yes (He mentioned his father -in-law) \n8. Was the man fond of his father -in-law? No (He referred to his father -in-law as “no -good” and \ndidn’t want him to get too comfortable) \n\n\n19 Appendix C: Responses of GPT -4 to the story in Appendix 1 when prompted for \ninferences, which are highlighted by us in square brackets. The underlined sentence is courtesy \nof GPT -4. \nSummarize this story, mentioning main ideas that are not stated and must be inferred. \nCertainly! Here’s a concise summary of the paragraph: \nSeveral women on Willow Street organized a garage sale during the fall. They gathered various \nitems from the neighborhood, priced them, and set up signs for the sale. On the first day, they \nsat down near the Anderson’s garage, waiting for customers. A man arrived in an old truck \nand showed interest in a lumpy old mattress leaning against the wall. Mrs. Anderson \ninformed him that it wasn’t for sale and would be discarded. \nCertainly! Let’s delve into the main ideas that can be inferred from this delightful story: \n1. Community Spirit and Resourcefulness : \n1.1. The women on Willow Street come together to organize a garage sale, [demonstrating \ntheir community spirit ]. \n1.2. They collect various items from their neighborhood, [showcasing resourcefulness and a \ndesire to repurpose things ]. \n2. Preparation and Effort : \n2.1. The women spend an entire day pricing the collected items, [emphasizing their dedication \nand hard work ]. \n2.2. They strategically place signs at both ends of the block and near a shopping center to \nattract customers, [highlighting their thoughtful preparation ]. \n3. Unexpected Interaction : \n3.1. A man arrives in an old truck, showing interest in a lumpy old mattress leaning against the \nwall. \n3.2. Mrs. Anderson informs him that the mattress isn’t for sale and will be discarded. \n3.3. The man’s [request to take the mattress leads to an unexpected interaction ]. \n4. Humorous Twist : \n4.1. When Mrs. Anderson questions why he wants such a terrible mattress, the man reveals his \nmotive: his “no -good” father -in-law is visiting next week, and he doesn’t want him to get \ntoo comfortable. \n4.2. [The humorous twist lies in using the uncomfortable mattress as a subtle way to \ninconvenience the unwelcome guest ]. \n5. Overall, the story highlights camaraderie, practicality, and a touch of wit in dealing with \nunwanted family visits. \n\n20 Appendix D: Test-maker explanations of answer correctness in Section 3 . \n\nWe believe that the t est-maker explanations are less effective than GPT -4’s justifications because \nGPT-4 is much more concise. See Figure 6 in the main text. \n\nQuestion 6 \nThe phrase “standard pricing pattern” as used in the middle of the last paragraph most nearly means \na pricing pattern \nA. against which other pricing patterns are to be measured \nB. that conforms to a commonly agreed -upon criterion \nC. that is merely acceptable \nD. that regularly recurs in certain types of cases \nE. that serves as an exemplar \n\nExplanation for Question 6 \n\nThis question requires the test taker to understand from context the meaning of the phrase \n“standard pricing pattern,” which is used by the author in a particular way. \nThe correct answer choice is (D). The phrase occurs in the last paragraph of the passage. The \npurpose of this paragraph is to detail Garber’s reasons for thinking that, contrary to Mackay’s view, \nthe seventeenth -century Dutch tulip market did not involve a speculative bubble. It is in this \ncontext that the author uses the phrase in question. The complete sentence reads, “Garber argues \nthat a standard pricing pattern occurs for new varieties of flowers.” The author then explains this \nstandard pricing pattern : original bulbs for prized new varieties initially command a high price, \nbut descendants produced from the original bulbs cost dramatically less over time. It is clear that \nthe author takes Garber to be describing a regularly recurring pattern about the p ricing of new \nvarieties of flowers, and then asserting that the particular details about the pricing of tulip bulbs in \nthe seventeenth century fit this recurring pattern. Thus, answer choice (D) is correct, since it \nparaphrases the use of the term “standar d pricing pattern” as a pricing pattern “that regularly recurs \nin certain types of cases.” \n\nAnswer choice (A) is incorrect. Nowhere does the author suggest that pricing patterns can or \nshould be “measured” against one another, much less against a pricing pattern that is for some \nreason taken to be the benchmark. \n\nAnswer choice (B) is incorrect. The passage as a whole does concern the interpretation of the \npricing of tulip bulbs in the seventeenth century, and it might be said that the debate between \nMackay and Garber concerns whether this case fits commonly agreed -upon criteria regarding \nspeculative bubbles. However, in the middle of the last paragraph Garber’s point is simply about \nprices fitting a pattern observed in a number of other cases. In this way, it is a point about \nconformance to a historical pattern, not to agreed -upon standards. \n\nAnswer choice (C) is incorrect. There is no reason to think that the author views pricing patterns \nas “acceptable” or unacceptable, or that the author believes there is a standard for acceptability. \nAnswer choice (E) is incorrect. An “exemplar” would be a particular case that serves as some kind \nof model or ideal. No particular case is being offered up as a model in the third paragraph. Instead \n21 the “standard pricing pattern” is only described generally, not by reference to some paradigm \nexample of the pattern Garber has in mind. \n\nBased on the number of test takers who answered this question correctly when it appeared on the \nLSAT, this was a difficult question. \n\nQuestion 7 \n\nGiven Garber’s account of the seventeenth -century Dutch tulip market, which one of the following \nis most analogous to someone who bought a tulip bulb of a certain variety in that market at a very \nhigh price, only to sell a bulb of that variety at a much lo wer price? \nA. someone who, after learning that many others had withdrawn their applications for a \nparticular job, applied for the job in the belief that there would be less competition for it \nB. an art dealer who, after paying a very high price for a new painting, sells it at a very low \nprice because it is now considered to be an inferior work \nC. someone who, after buying a box of rare motorcycle parts at a very high price, is forced to \nsell them at a much lower price because of the sudden availability of cheap substitute parts \nD. a publisher who pays an extremely high price for a new novel only to sell copies at a price \naffordable to nearly everyone \nE. an airline that, after selling most of the tickets for seats on a plane at a very high price, must \nsell the remaining tickets at a very low price \n\nExplanation for Question 7 \n\nThis question requires the test taker to identify the scenario that is most analogous to the way in \nwhich Garber would view the purchase of a tulip bulb at a very high price, and the later sale of \ntulip bulbs of that same variety at a much lower price. Bef ore looking at the answer choices, it is \nworth getting clear on the specifics of Garber’s account. In Garber’s view, the value of the original \nbulb reflects the earnings that can be made from the descendant bulbs. Since an original bulb will \nproduce multip le descendants, the value of the original will be much greater than the value of any \nindividual descendant. The value of the original reflects the cumulative value of the descendants. \nThus, someone could buy an original bulb at a very high price and still turn a profit by selling \ndescendant bulbs at a much lower price. \n\nThe correct answer choice is (D). The relation between the manuscript of a new novel and the \ncopies that can be made of that novel is analogous to the relation between an original bulb and its \ndescendants. From the original novel, the publisher can produce many copies. The copies can then \nbe sold for a much lower price than the original. The value of the new novel reflects the cumulative \nvalue of the sales of the copies. \n\nAnswer choice (A) is incorrect. The scenario described does not include anything akin to the \nrelationship between an original bulb and later descendants. Instead, it presents an example of \nsomeone who applies for a job based on a perception about the degre e of competition for that job. \n\n22 Answer choice (B) is incorrect. In this scenario, the value of the painting has dropped due to critical \nor public opinion. This represents a case in which the art dealer has taken a loss, not one where the \nart dealer recoups the original value of the paint ing through an accumulation of smaller sales. \n\nAnswer choice (C) is incorrect. On the surface, the drop in price of the motorcycle parts due to a \nflooded market of replacement parts seems similar to the drop in price of the bulbs of a variety of \nflowers. However, the situation is disanalogous in crucia l respects. The cheap substitute parts \ncannot be described as anything like “descendants” of the original rare parts, and the owner of the \nbox of rare parts does not get the value back through the cumulative sales of the cheap \nreplacements. Indeed, the own er of the box of rare motorcycle parts was simply forced to sell the \nparts at a loss. \n\nAnswer choice (E) is incorrect. The airline had a certain number of seats for which they could sell \ntickets. The drop in price over time is not a product of increased availability, as in the case of the \nflower bulbs. In this case, the number of available s eats has actually decreased. While it is surely \nrational for the airline to reduce the price of the seats, the situation is not analogous to the drop in \nprice of descendant flower bulbs. \n\nBased on the number of test takers who answered this question correctly when it appeared on the \nLSAT, this was a difficult question.",
            "sub_sections": [
                {
                "title": "Summary of findings",
                "first_sentence": "Our converging evidence for genuine understanding of text in GPT -4 is strong due to a high level of correct responding to questions about the text, correct and concise justifications for those answers, and subtle signatures of understanding that go beyond the literal text such as generalization inferences and other inferences.",
                "sub_sections": [],
                "text": "Our converging evidence for genuine understanding of text in GPT -4 is strong due to a high \nlevel of correct responding to questions about the text , correct and concise justifications for those \nanswers , and subtle signatures of understanding that go beyond the literal text such as \ngeneralization inferences and other inference s. It suggests that GPT-4 can exceed human levels of \ntext understanding, provided that the text is sufficiently difficult. \nDoes this mean that GPT-4 has achieved Artificial General Intelligence (AGI)? It does not, \nbecause reading comprehension is only one skill in the AGI sphere, albeit a critically important \nskill because reading is the principal way that an LLM currently receives its knowledge. \nNonetheless , reading comprehension can now be added to the long and growing list of cognitive \nskills on which GPT -4 has matched or exceeded average human performance: decision making \n(6), metaphors, idioms, estimation, anaphoric reference, creativity, problem solving, and \nanswering common sense questions (3). \nHowever, as we noted in section 1 , LLMs still have difficulty with causal reasoning (6), \nplanning (7), and graduate -level mathematics (8). Causal reasoning might well require models of \nagentive interaction with the physical world, involving the control of purpose ful sensorimotor \ninteractions (18). As well, a case could be made that such factors should be employed in AI \nconstruction well before the agents are capable of reading. Even very young infants act as though \nthey are aware that their actions cause certain effects (19,20) . Although infants are far from being \nable to read, their causal knowledge gets firmly grounded in purposely acting on the world to \nproduce desired effects . \nAnother promising approach to causal reasoning involves using LLMs to provide prior \nprobabilit ies to construct causal graphs (21). Such information from LLMs was found to i mprove \nperformance on commonsense causal benchmark phenomena, especially when deciding what is \nthe cause and what is the effect. \nOne thing to keep in mind when considering the high percentiles achieved by GPT -4 on \ndifficult academic tests is that there are a small number of human test -takers who achieved even \nhigher percentiles than achieved by GTP -4 on those tests. It could be interesting to study such \npeople while monitoring the ongoing quest of LLMs striving for perfect success on important \ntasks . We happened to encounter some aspects of this issue in our final experiment using a small \n14-question snippet of an LSAT test on which GPT -4 got every question correct. Some of those \nquestions were flagged as very difficult by test -makers based on their difficulty for humans. \n14"
                },
                {
                "title": "Implications for AI and human understanding",
                "first_sentence": "Although our article is presented in a kind of competition between humans and AI, we believe that LLMs could rapidly become of great service to interested humans.",
                "sub_sections": [],
                "text": "Although our article is presented in a kind of competition between humans and AI, we \nbelieve that LLMs could rapidly become of great service to interested humans. For example, \nhumans could come to rely on yet to be fully developed AI systems that can read widely and \nreliably explain to us what they read and underst and. \nAcknowledgments \nThis research is supported by a grant from the Natural Sciences and Engineering Research Council \nof Canada to TRS: NSERC 2021 -03434. \nReferences \n1. Chomsky, N. The False Promise of ChatGPT. New York Times. 2023 Mar 8; \n2. Marcus, G., Davis, E. Rebooting AI: building artificial intelligence we can trust. New York: \nPantheon Books; 2019. \n3. Sartori, G., Orru, G. Language models and psychological sciences. Frontiers in Psychology. \n2023;14:1 –16. (doi.org/10.3389/fpsyg.2023.1279317 ) \n4. Webb, T., Holyoak, K. & Lu, H. Emergent analogical reasoning in large language models. \nNature Human Behaviour. 2023;7:1526 –41. (doi.org/10.1038/s41562 -023-01659 -w) \n5. Kosinski, M. Theory of mind may have spontaneously emerged in LLMs. Cornell \nUniversity: arXiv; 2023. (2302.02083). \n6. Binz M., & Schulz E. Using cognitive psychology to understand GPT -3. Proceedings of the \nNational Academy of Sciences. 2023;120(6). (doi.org/10.1073/pnas.2218523120 ) \n7. Valmeekam, K., Olmo, A., Sreedharan, S., & Kambhampati, S. LLMs still can’t plan (a \nbenchmark for LLMs on planning and reasoning about change). Cornell University: arXiv; \n2022. (2206.10498). \n8. Frieder S, Pinchetti L, and Griffiths RR, Salvatori T, Lukasiewicz T, Petersen P, et al. \nMathematical Capabilities of ChatGPT. In: Oh A, Naumann T, Globerson A, Saenko K, \nHardt M, Levine S, editors. Advances in Neural Information Processing Systems [Int ernet]. \nCurran Associates, Inc.; 2023. p. 27699 –744. Available from: \nhttps://proceedings.neurips.cc/paper_files/paper/2023/file/58168e8a92994655d6da3939e7cc \n0918 -Paper -Datasets_and_Benchmarks.pdf \n9. Graesser, A., Singer, M., Trabasso, T. Constructing Inferences During Narrative Text \nComprehension. Psychological Review. 1994;101(3):371 –95. \n10. Brookshire RH, Nicholas LE. Discourse Comprehension Test: Test Manual [Internet]. BRK \nPublishers; 1997. Available from: https://books.google.ca/books?id=vAQFzQEACAAJ \n11. Dale, E., & Chall, J. A formula for predicting readability. Educational Research Bulletin. \n1948;27:11 –20. \n15 12. Nicholas, L., Brookshire, R. Comprehension of spoken narrative discourse by adults with \naphasia, right -hemisphere brain damage, or traumatic brain injury. American Journal of \nSpeech -Language Pathology. 1995;4(3):69 –81. \n13. Singer, M. Inference processing in discourse comprehension. In: The Oxford Handbook of \nPsycholinguistics. Oxford, UK: Oxford University Press; 2007. p. 343 –59. \n14. Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., et al. Sparks \nof artificial general intelligence: early experiments with gpt -4. Cornell University: arXiv; \n2023. (2303 -08774v6). \n15. Frederiksen, C. Acquisition of Semantic Information from Discourse: Effects of Repeated \nExposures. Journal of Verbal Learning and Verbal Behavior. 1975;14:158 –69. \n16. Kintsch, W., Kozminsky, E., Streby, W., McKoon, G., Keenan, J. Comprehension and Recall \nof Text as a Function of Content Variables. Journal of Verbal Learning and Verbal Behavior. \n1975;14:196 –214. \n17. OpenAI. GPT -4 Technical Report. Cornell University: arXiv; 2024. \n18. Pezzullo, G., Parr, T., Cisek, P., Clark, A., & Friston, K. Generating meaning: active \ninference and the scope and limits of passive AI. Trends in Cognitive Sciences. \n2024;28(2):97 –112. \n19. Hauf, P., & Aschersleben, G. Action –effect anticipation in infant action control. \nPsychological Research. 2008;72:203 –10. \n20. Rovee, C., & Rovee, D. Conjugate reinforcement of infants’ exploratory behavior. Journal of \nExperimental Child Psychology. 1969;8:33 –9. \n21. Darvariu, V., Hailes, S., & Musolesi, M. Large Language Models are Effective Priors for \nCausal Graph Discovery. Cornell University: arXiv; 2024. (2405.13551v1). \n\n\n16 Appendices \nAppendix A: Example story and its questions, with their correct answers and values of the \nsalience and directness variables. \nAppendix B: Responses of GPT -4 to the story in Appendix A. \nAppendix C: Responses of GPT -4 to the story in Appendix A when prompted for inferences, \nwhich are highlighted by us in square brackets. \nAppendix D: Test -maker explanations of answer correctness in Section 3. \n\n17 Appendix A: Example story and its questions, with their correct answers and values of \nthe salience and directness variables in Section 2. \nRead this story in preparation for answering eight yes/no questions about the story. \nOne day last Fall, several women on Willow Street decided to have a garage sale. They collected \nodds and ends from all over the neighborhood. Then they spent an entire day putting prices on the \nthings that they had collected. On the first day of the sale, they put up signs at both ends of the \nblock and another one at a nearby shopping center. Next, they made a pitcher of iced tea and sat \ndown in a shady spot beside the Anderson’s garage to wait for their first customer. Soon a man \ndrove up in an old truck. He looked around and finally stopped by a lumpy old mattress that was \nleaning against the wall. He gestured to it and asked how much they wanted for it. Mrs. Anderson \ntold him that it wasn’t for sale. Then she added that they were going to put it out for t he trash \ncollectors the next day. The man asked if he could have it. Mrs. Anderson said that he could. Then \nshe asked, “Why do you want such a terrible mattress?” “Well,” he said, “My no -good father -in- \nlaw is coming to visit next week, and I don’t want him to get too comfortable.” \n1. Did several women have a party? No, stated main idea. \n2. Were there a large number of things at the garage sale? Yes, implied main idea. \n3. Did the women put up a sign at a shopping center? Yes, stated detail. \n4. Was it cold the day of the garage sale? No, implied detail. \n5. Was the man driving a car? No, stated detail. \n6. Was the mattress in terrible condition? Yes, stated main idea. \n7. Was the man married? Yes, implied detail. \n8. Was the man fond of his father -in-law? No, implied main idea. \n\n18 Appendix B: Responses of GPT -4 to the story in Appendix 1. \nNote that we add parentheses around the spontaneous elaborations given by GPT -4 beyond its \nyes/no answers, which are printed in bold font by GPT -4. \nAnswer each of these yes/no questions about the story. \n1. Did several women have a party? No \n2. Were there a large number of things at the garage sale? Yes \n3. Did the women put up a sign at a shopping center? Yes \n4. Was it cold the day of the garage sale? No (The story does not provide information about the \ntemperature) \n5. Was the man driving a car? No (He drove up in an old truck) \n6. Was the mattress in terrible condition? Yes \n7. Was the man married? Yes (He mentioned his father -in-law) \n8. Was the man fond of his father -in-law? No (He referred to his father -in-law as “no -good” and \ndidn’t want him to get too comfortable) \n\n\n19 Appendix C: Responses of GPT -4 to the story in Appendix 1 when prompted for \ninferences, which are highlighted by us in square brackets. The underlined sentence is courtesy \nof GPT -4. \nSummarize this story, mentioning main ideas that are not stated and must be inferred. \nCertainly! Here’s a concise summary of the paragraph: \nSeveral women on Willow Street organized a garage sale during the fall. They gathered various \nitems from the neighborhood, priced them, and set up signs for the sale. On the first day, they \nsat down near the Anderson’s garage, waiting for customers. A man arrived in an old truck \nand showed interest in a lumpy old mattress leaning against the wall. Mrs. Anderson \ninformed him that it wasn’t for sale and would be discarded. \nCertainly! Let’s delve into the main ideas that can be inferred from this delightful story: \n1. Community Spirit and Resourcefulness : \n1.1. The women on Willow Street come together to organize a garage sale, [demonstrating \ntheir community spirit ]. \n1.2. They collect various items from their neighborhood, [showcasing resourcefulness and a \ndesire to repurpose things ]. \n2. Preparation and Effort : \n2.1. The women spend an entire day pricing the collected items, [emphasizing their dedication \nand hard work ]. \n2.2. They strategically place signs at both ends of the block and near a shopping center to \nattract customers, [highlighting their thoughtful preparation ]. \n3. Unexpected Interaction : \n3.1. A man arrives in an old truck, showing interest in a lumpy old mattress leaning against the \nwall. \n3.2. Mrs. Anderson informs him that the mattress isn’t for sale and will be discarded. \n3.3. The man’s [request to take the mattress leads to an unexpected interaction ]. \n4. Humorous Twist : \n4.1. When Mrs. Anderson questions why he wants such a terrible mattress, the man reveals his \nmotive: his “no -good” father -in-law is visiting next week, and he doesn’t want him to get \ntoo comfortable. \n4.2. [The humorous twist lies in using the uncomfortable mattress as a subtle way to \ninconvenience the unwelcome guest ]. \n5. Overall, the story highlights camaraderie, practicality, and a touch of wit in dealing with \nunwanted family visits. \n\n20 Appendix D: Test-maker explanations of answer correctness in Section 3 . \n\nWe believe that the t est-maker explanations are less effective than GPT -4’s justifications because \nGPT-4 is much more concise. See Figure 6 in the main text. \n\nQuestion 6 \nThe phrase “standard pricing pattern” as used in the middle of the last paragraph most nearly means \na pricing pattern \nA. against which other pricing patterns are to be measured \nB. that conforms to a commonly agreed -upon criterion \nC. that is merely acceptable \nD. that regularly recurs in certain types of cases \nE. that serves as an exemplar \n\nExplanation for Question 6 \n\nThis question requires the test taker to understand from context the meaning of the phrase \n“standard pricing pattern,” which is used by the author in a particular way. \nThe correct answer choice is (D). The phrase occurs in the last paragraph of the passage. The \npurpose of this paragraph is to detail Garber’s reasons for thinking that, contrary to Mackay’s view, \nthe seventeenth -century Dutch tulip market did not involve a speculative bubble. It is in this \ncontext that the author uses the phrase in question. The complete sentence reads, “Garber argues \nthat a standard pricing pattern occurs for new varieties of flowers.” The author then explains this \nstandard pricing pattern : original bulbs for prized new varieties initially command a high price, \nbut descendants produced from the original bulbs cost dramatically less over time. It is clear that \nthe author takes Garber to be describing a regularly recurring pattern about the p ricing of new \nvarieties of flowers, and then asserting that the particular details about the pricing of tulip bulbs in \nthe seventeenth century fit this recurring pattern. Thus, answer choice (D) is correct, since it \nparaphrases the use of the term “standar d pricing pattern” as a pricing pattern “that regularly recurs \nin certain types of cases.” \n\nAnswer choice (A) is incorrect. Nowhere does the author suggest that pricing patterns can or \nshould be “measured” against one another, much less against a pricing pattern that is for some \nreason taken to be the benchmark. \n\nAnswer choice (B) is incorrect. The passage as a whole does concern the interpretation of the \npricing of tulip bulbs in the seventeenth century, and it might be said that the debate between \nMackay and Garber concerns whether this case fits commonly agreed -upon criteria regarding \nspeculative bubbles. However, in the middle of the last paragraph Garber’s point is simply about \nprices fitting a pattern observed in a number of other cases. In this way, it is a point about \nconformance to a historical pattern, not to agreed -upon standards. \n\nAnswer choice (C) is incorrect. There is no reason to think that the author views pricing patterns \nas “acceptable” or unacceptable, or that the author believes there is a standard for acceptability. \nAnswer choice (E) is incorrect. An “exemplar” would be a particular case that serves as some kind \nof model or ideal. No particular case is being offered up as a model in the third paragraph. Instead \n21 the “standard pricing pattern” is only described generally, not by reference to some paradigm \nexample of the pattern Garber has in mind. \n\nBased on the number of test takers who answered this question correctly when it appeared on the \nLSAT, this was a difficult question. \n\nQuestion 7 \n\nGiven Garber’s account of the seventeenth -century Dutch tulip market, which one of the following \nis most analogous to someone who bought a tulip bulb of a certain variety in that market at a very \nhigh price, only to sell a bulb of that variety at a much lo wer price? \nA. someone who, after learning that many others had withdrawn their applications for a \nparticular job, applied for the job in the belief that there would be less competition for it \nB. an art dealer who, after paying a very high price for a new painting, sells it at a very low \nprice because it is now considered to be an inferior work \nC. someone who, after buying a box of rare motorcycle parts at a very high price, is forced to \nsell them at a much lower price because of the sudden availability of cheap substitute parts \nD. a publisher who pays an extremely high price for a new novel only to sell copies at a price \naffordable to nearly everyone \nE. an airline that, after selling most of the tickets for seats on a plane at a very high price, must \nsell the remaining tickets at a very low price \n\nExplanation for Question 7 \n\nThis question requires the test taker to identify the scenario that is most analogous to the way in \nwhich Garber would view the purchase of a tulip bulb at a very high price, and the later sale of \ntulip bulbs of that same variety at a much lower price. Bef ore looking at the answer choices, it is \nworth getting clear on the specifics of Garber’s account. In Garber’s view, the value of the original \nbulb reflects the earnings that can be made from the descendant bulbs. Since an original bulb will \nproduce multip le descendants, the value of the original will be much greater than the value of any \nindividual descendant. The value of the original reflects the cumulative value of the descendants. \nThus, someone could buy an original bulb at a very high price and still turn a profit by selling \ndescendant bulbs at a much lower price. \n\nThe correct answer choice is (D). The relation between the manuscript of a new novel and the \ncopies that can be made of that novel is analogous to the relation between an original bulb and its \ndescendants. From the original novel, the publisher can produce many copies. The copies can then \nbe sold for a much lower price than the original. The value of the new novel reflects the cumulative \nvalue of the sales of the copies. \n\nAnswer choice (A) is incorrect. The scenario described does not include anything akin to the \nrelationship between an original bulb and later descendants. Instead, it presents an example of \nsomeone who applies for a job based on a perception about the degre e of competition for that job. \n\n22 Answer choice (B) is incorrect. In this scenario, the value of the painting has dropped due to critical \nor public opinion. This represents a case in which the art dealer has taken a loss, not one where the \nart dealer recoups the original value of the paint ing through an accumulation of smaller sales. \n\nAnswer choice (C) is incorrect. On the surface, the drop in price of the motorcycle parts due to a \nflooded market of replacement parts seems similar to the drop in price of the bulbs of a variety of \nflowers. However, the situation is disanalogous in crucia l respects. The cheap substitute parts \ncannot be described as anything like “descendants” of the original rare parts, and the owner of the \nbox of rare parts does not get the value back through the cumulative sales of the cheap \nreplacements. Indeed, the own er of the box of rare motorcycle parts was simply forced to sell the \nparts at a loss. \n\nAnswer choice (E) is incorrect. The airline had a certain number of seats for which they could sell \ntickets. The drop in price over time is not a product of increased availability, as in the case of the \nflower bulbs. In this case, the number of available s eats has actually decreased. While it is surely \nrational for the airline to reduce the price of the seats, the situation is not analogous to the drop in \nprice of descendant flower bulbs. \n\nBased on the number of test takers who answered this question correctly when it appeared on the \nLSAT, this was a difficult question."
                }
            ]
            }
        ],
        "learning_objectives": {
            "1": "Understand how GPT-4's performance on discourse comprehension tests compares to human performance.",
            "2": "Recognize the signatures of genuine understanding in AI, such as inference and generalization.",
            "3": "Appreciate the methods used to evaluate AI comprehension of simple and complex texts.",
            "4": "Identify the limitations of GPT-4 in causal reasoning, planning, and advanced mathematics.",
            "5": "Reflect on the implications of AI's text understanding capabilities for future AI development and applications."
        }

    }


    # Add fake questions for this doc
    doc1 = DOCUMENTS.get(doc_id)
    sections = doc1.get("sections", [])
    for section in sections:
        section_id = section.get("id")
    QUESTIONS[f"{doc_id}_{section_id}"] = [
        {
            "question_text": "What is the main topic of the document?",
            "options": [
            "How AI models comprehend text",
            "History of artificial intelligence",
            "Syntax in natural languages",
            "Computer vision techniques"
            ],
            "correct_index": 0,
            "explanation": "The document is about how AI models comprehend text.",
            "difficulty_score": 0.2,
            "concept_tags": ["AI comprehension"],
            "salience": 0.8,
            "directness": 0.9
        },
        {
            "question_text": "Which of the following best describes the content of the section?",
            "options": [
            "Details computer vision models",
            "Introduces the topic of AI understanding of text",
            "Explains hardware requirements for AI",
            "Describes neural network architectures"
            ],
            "correct_index": 1,
            "explanation": "The section introduces how AI models comprehend language.",
            "difficulty_score": 0.3,
            "concept_tags": ["language understanding"],
            "salience": 0.7,
            "directness": 0.85
        }
    ]

    new_block = {
            "c8636fc3-9e38-4e1b-9e52-83052c72de9d_sec0": [
                {
                "question_text": "In debates about AI systems approaching human-level cognition, what role does human performance typically serve as in evaluating these systems?",
                "options": [
                    "It acts as the primary benchmark to determine if AI has achieved general intelligence.",
                    "It is considered an irrelevant comparison because AI systems process information differently from humans.",
                    "It serves as a secondary measure, with emphasis placed solely on AI performance metrics.",
                    "It is used only to assess the efficiency of AI systems rather than their cognitive capabilities."
                ],
                "correct_index": 0,
                "explanation": "The section highlights that humans are often used as the gold standard or benchmark when debating whether AI systems are approaching human-level cognition, especially in tasks like language understanding and reasoning.",
                "difficulty_score": 0.75,
                "concept_tags": [
                    "AI cognition debate",
                    "human performance as benchmark",
                    "AGI comparison"
                ],
                "salience": 0.9,
                "directness": 0.8
                },
                {
                "question_text": "Which of the following best describes how large language models like GPT-4 are trained and their impact on cognitive task performance?",
                "options": [
                    "They are trained on vast amounts of text data to predict the next token, which enables them to perform a variety of challenging cognitive tasks at or above human levels without additional learning.",
                    "They are trained on specific labeled datasets for each task, which limits their ability to generalize but ensures high accuracy in targeted cognitive tasks.",
                    "They are trained through reinforcement learning with human feedback, which primarily enhances their ability to generate more human-like responses but does not significantly improve task performance.",
                    "They are trained on structured knowledge bases and logical rules, which allows them to solve problems requiring deep reasoning but restricts their language generation capabilities."
                ],
                "correct_index": 0,
                "explanation": "Large language models like GPT-4 are trained on enormous amounts of text data to predict the next token, a process known as unsupervised learning, which has been shown to improve performance on a wide range of challenging cognitive tasks often reaching or exceeding human levels.",
                "difficulty_score": 0.6,
                "concept_tags": [
                    "training data",
                    "large language models",
                    "cognitive task performance"
                ],
                "salience": 0.85,
                "directness": 0.7
                }
            ],
            "c8636fc3-9e38-4e1b-9e52-83052c72de9d_sec1_0": [
                {
                "question_text": "Which of the following best describes the features and structure of the Discourse Comprehension Test as discussed in the section?",
                "options": [
                    "It includes stories with multiple-choice questions focused on vocabulary understanding, scored based on accuracy and speed.",
                    "It consists of stories with yes/no questions examining salience and directness, with questions categorized into four types and scored for correctness, including handling hedged responses.",
                    "It uses true/false questions to assess factual knowledge, with scoring based on the number of correct responses and the difficulty level of stories.",
                    "It involves open-ended questions about story content, with scoring based on the depth of inference and the coherence of answers."
                ],
                "correct_index": 1,
                "explanation": "The section describes the Discourse Comprehension Test as comprising stories followed by eight yes/no questions that examine salience and directness, categorized into four question types. Answers are scored for correctness, with special handling for hedged responses, making option 2 the accurate description.",
                "difficulty_score": 0.7,
                "concept_tags": [
                    "Discourse Comprehension Test",
                    "Question Types",
                    "Scoring Criteria"
                ],
                "salience": 0.9,
                "directness": 1
                },
                {
                "question_text": "Which aspect of GPT-4's training architecture primarily enables it to understand discourse by focusing on relevant parts of sentences, especially those with long-range dependencies?",
                "options": [
                    "Its use of a transformer architecture with attention mechanisms that weigh relevant information across the input",
                    "Its training on a limited dataset of structured stories to enhance comprehension of main ideas",
                    "Its reliance on supervised labeled datasets specifically designed for discourse analysis",
                    "Its hierarchical neural network that processes sentences sequentially to capture context"
                ],
                "correct_index": 0,
                "explanation": "GPT-4's transformer architecture leverages attention mechanisms to focus on relevant parts of sentences, enabling it to handle long-range dependencies crucial for discourse comprehension.",
                "difficulty_score": 0.8,
                "concept_tags": [
                    "transformer attention mechanisms",
                    "training scale",
                    "discourse comprehension"
                ],
                "salience": 0.8,
                "directness": 0.6
                }
            ],
            "c8636fc3-9e38-4e1b-9e52-83052c72de9d_sec1_1": [
                {
                "question_text": "Based on the comparison of GPT-4 and human performance on the Discourse Comprehension Test, which statement is most accurate?",
                "options": [
                    "GPT-4 significantly outperforms humans in all types of questions on the test.",
                    "GPT-4 slightly exceeds human performance overall, but the differences are not statistically significant.",
                    "Humans perform better than GPT-4 on implied details, especially when questions require inference.",
                    "GPT-4 and humans perform equally, with no differences observed in any of the test conditions."
                ],
                "correct_index": 1,
                "explanation": "The section states that GPT-4 slightly exceeds human performance overall, but both differences are well below statistical significance, indicating no meaningful performance difference.",
                "difficulty_score": 0.4,
                "concept_tags": [
                    "GPT-4 vs. human performance",
                    "Discourse Comprehension Test",
                    "performance comparison"
                ],
                "salience": 0.9,
                "directness": 1
                },
                {
                "question_text": "Based on the statistical results presented, what does the phrase \"differences are far below statistical significance\" imply about the performance comparison between GPT-4 and humans?",
                "options": [
                    "GPT-4's performance is conclusively better than human performance with high confidence.",
                    "There is no meaningful difference in performance between GPT-4 and humans based on the data, as the observed differences could be due to chance.",
                    "Humans perform significantly better than GPT-4, but the difference is small enough to be considered negligible.",
                    "The performance difference favors humans, but the statistical tests were not appropriate for this comparison."
                ],
                "correct_index": 1,
                "explanation": "The phrase indicates that the observed differences in performance between GPT-4 and humans are not statistically significant, meaning any apparent difference could be due to random variation rather than a true performance gap.",
                "difficulty_score": 0.6,
                "concept_tags": [
                    "statistical significance",
                    "performance comparison",
                    "data interpretation"
                ],
                "salience": 0.8,
                "directness": 1
                }
            ],
            "c8636fc3-9e38-4e1b-9e52-83052c72de9d_sec1_2": [
                {
                "question_text": "What does GPT-4's performance on simple passages imply about its understanding capabilities?",
                "options": [
                    "GPT-4's high performance indicates it fully comprehends all implied and detailed information in passages.",
                    "GPT-4 can perform well on simple passages, suggesting it has a strong ability to understand and infer beyond the explicitly stated content.",
                    "GPT-4's performance shows it relies solely on memorized data rather than true understanding of passages.",
                    "GPT-4 struggles with simple passages, indicating limitations in basic comprehension and inference."
                ],
                "correct_index": 1,
                "explanation": "The section highlights that GPT-4's strong performance on simple stories suggests it understands what it has learned in a single shot, including making inferences beyond what is explicitly stated, indicating a meaningful level of comprehension.",
                "difficulty_score": 0.4,
                "concept_tags": [
                    "GPT-4 performance",
                    "discourse comprehension",
                    "understanding inference"
                ],
                "salience": 0.9,
                "directness": 0.8
                },
                {
                "question_text": "What is a primary challenge humans and GPT-4 face when answering questions that require understanding implied details in a story?",
                "options": [
                    "They often confuse explicit information with implied knowledge, making inference difficult.",
                    "They tend to focus too much on main points, neglecting the details and implications.",
                    "They may ignore or forget details due to memory constraints, complicating inference.",
                    "They frequently provide inaccurate summaries because they lack world knowledge."
                ],
                "correct_index": 2,
                "explanation": "Both humans and GPT-4 struggle with implied details because memory constraints can lead to ignoring or forgetting details, which are essential for making inferences.",
                "difficulty_score": 0.6,
                "concept_tags": [
                    "implied details",
                    "inference challenges",
                    "discourse comprehension"
                ],
                "salience": 0.8,
                "directness": 0.3
                }
            ],
            "c8636fc3-9e38-4e1b-9e52-83052c72de9d_sec2_0": [
                {
                "question_text": "What does GPT-4's percentile score of 96.3 across three academic reading tests primarily indicate about its reading comprehension ability?",
                "options": [
                    "It demonstrates that GPT-4 has a superior ability to understand and interpret complex passages compared to the average human test-taker.",
                    "It suggests that GPT-4's training data included specific examples from these tests, leading to high performance through memorization.",
                    "It indicates that GPT-4's high scores are likely due to random guessing, given the difficulty of the tests.",
                    "It reflects that GPT-4's reading comprehension skills are limited to the types of questions it was explicitly trained on."
                ],
                "correct_index": 0,
                "explanation": "The high percentile scores, especially after contamination checks, suggest that GPT-4's performance reflects genuine reading comprehension ability rather than memorization of test content.",
                "difficulty_score": 0.6,
                "concept_tags": [
                    "GPT-4 reading comprehension",
                    "percentile scores",
                    "test interpretation"
                ],
                "salience": 0.9,
                "directness": 1
                },
                {
                "question_text": "Why is it important to use uncontaminated test scores when assessing GPT-4’s true reading comprehension capabilities?",
                "options": [
                    "Because contaminated scores may include memorized training data, inflating perceived performance.",
                    "Because uncontaminated scores ensure the test questions are more challenging for GPT-4.",
                    "Because contamination checks are necessary only for the LSAT, not for other tests.",
                    "Because uncontaminated scores are easier to compare across different AI models."
                ],
                "correct_index": 0,
                "explanation": "Using uncontaminated scores removes the influence of training data overlap, providing a more accurate measure of GPT-4’s genuine reading comprehension abilities.",
                "difficulty_score": 0.7,
                "concept_tags": [
                    "assessment validity",
                    "test data contamination",
                    "AI performance evaluation"
                ],
                "salience": 0.8,
                "directness": 1
                }
            ],
            "c8636fc3-9e38-4e1b-9e52-83052c72de9d_sec2_1": [
                {
                "question_text": "How does GPT-4 demonstrate understanding of text through the experiment described in the passage?",
                "options": [
                    "By providing justifications for its multiple-choice answers, showcasing its ability to infer and generalize from the passage.",
                    "By memorizing correct answers from the LSAT passages without explaining its reasoning.",
                    "By matching human student responses in the test without generating explanations or inferences.",
                    "By identifying precise experimental distinctions between stated and inferred information in the passages."
                ],
                "correct_index": 0,
                "explanation": "GPT-4 demonstrates understanding by providing justifications for its answers, which involves inference and generalization beyond mere memorization.",
                "difficulty_score": 0.6,
                "concept_tags": ["generalization", "inference"],
                "salience": 0.9,
                "directness": 0.7
                },
                {
                "question_text": "What experimental approach was used to assess GPT-4's ability to justify answers on LSAT passages?",
                "options": [
                    "Prompting GPT-4 to provide justifications for each of its answers on LSAT passages with explanations for correctness and incorrectness.",
                    "Comparing GPT-4's multiple-choice responses to human student answers without requesting explanations.",
                    "Analyzing GPT-4's ability to distinguish between main points and details in LSAT passages without additional prompts.",
                    "Using GPT-4 to generate new LSAT passages and evaluating its understanding based on passage creation."
                ],
                "correct_index": 0,
                "explanation": "The experiment involved prompting GPT-4 to provide justifications for each answer on LSAT passages, including explanations for correctness and incorrectness, to assess its ability to justify responses.",
                "difficulty_score": 0.7,
                "concept_tags": [
                    "experimental assessment",
                    "GPT-4 justification",
                    "LSAT comprehension"
                ],
                "salience": 1,
                "directness": 0.8
                }
            ],
            "c8636fc3-9e38-4e1b-9e52-83052c72de9d_sec2_2": [
                {
                "question_text": "What is the primary instruction given to GPT-4 for answering LSAT questions in the simulation experiment?",
                "options": [
                    "To read the passage, answer the questions, and justify each answer.",
                    "To summarize the passage before answering the questions.",
                    "To answer the questions without providing any justification.",
                    "To analyze the passage and then choose the most probable answer."
                ],
                "correct_index": 0,
                "explanation": "The prompt instructs GPT-4 to read the passage, answer the questions, and justify each answer, emphasizing the importance of explanation.",
                "difficulty_score": 0.3,
                "concept_tags": [
                    "GPT-4 prompt instructions",
                    "LSAT question answering",
                    "AI explanation requirement"
                ],
                "salience": 0.9,
                "directness": 1
                },
                {
                "question_text": "What types of passages and question formats are used in the LSAT simulation described in the section?",
                "options": [
                    "The simulation uses a variety of passages, including single and paired passages, with multiple-choice questions that range from two to eight per passage.",
                    "The simulation only includes single passages with a fixed set of five questions each, focusing on factual recall.",
                    "The simulation features paired passages exclusively on scientific topics, with questions requiring inference and synthesis.",
                    "The simulation employs only one passage type—short, single-paragraph texts—with true/false questions."
                ],
                "correct_index": 0,
                "explanation": "The section describes the use of both single and paired passages, with the number of questions varying from two to eight, covering diverse topics, which aligns with answer A.",
                "difficulty_score": 0.4,
                "concept_tags": ["Passage Types", "Question Formats", "LSAT Simulation"],
                "salience": 0.8,
                "directness": 1
                }
            ],
            "c8636fc3-9e38-4e1b-9e52-83052c72de9d_sec2_3": [
                {
                "question_text": "In the context of the passage, what does the phrase 'standard pricing pattern' most nearly mean?",
                "options": [
                    "A. against which other pricing patterns are to be measured",
                    "B. that conforms to a commonly agreed-upon criterion",
                    "C. that is merely acceptable",
                    "D. that regularly recurs in certain types of cases"
                ],
                "correct_index": 3,
                "explanation": "The passage explains that Garber describes a 'standard pricing pattern' as a recurring pattern in the pricing of new flower varieties, where prices initially are high and then fall rapidly as the variety becomes more available. This indicates a pattern that regularly recurs in specific cases, making option D the correct choice.",
                "difficulty_score": 0.6,
                "concept_tags": [
                    "interpretation of terminology",
                    "contextual understanding",
                    "economic patterns"
                ],
                "salience": 0.9,
                "directness": 1
                },
                {
                "question_text": "Based on Garber’s description of the tulip market, which analogy best reflects the pattern of buying a tulip bulb at a high price and then selling similar bulbs at a much lower price?",
                "options": [
                    "A. someone who, after learning that many others had withdrawn their applications for a particular job, applied for the job in the belief that there would be less competition for it",
                    "B. an art dealer who, after paying a very high price for a new painting, sells it at a very low price because it is now considered to be an inferior work",
                    "C. someone who, after buying a box of rare motorcycle parts at a very high price, is forced to sell them at a much lower price because of the sudden availability of cheap substitute parts",
                    "D. a publisher who pays an extremely high price for a new novel only to sell copies at a price affordable to nearly everyone"
                ],
                "correct_index": 3,
                "explanation": "Answer D is correct because it mirrors Garber’s description: an initial high investment (buying the original tulip bulb) leads to many lower-cost items (reproduced bulbs or copies) being sold at much lower prices, similar to a publisher paying a high price upfront and then selling many copies at a low price. The other options involve different dynamics that do not directly reflect the pattern of high initial investment followed by widespread low-price sales.",
                "difficulty_score": 0.7,
                "concept_tags": [
                    "economic analogy",
                    "speculative bubble",
                    "pricing patterns"
                ],
                "salience": 0.8,
                "directness": 1
                }
            ],
            "c8636fc3-9e38-4e1b-9e52-83052c72de9d_sec3": [
                {
                "question_text": "How does increasing passage difficulty affect GPT-4's performance compared to human performance?",
                "options": [
                    "GPT-4's performance surpasses humans only on simple texts, with no significant difference on more difficult passages.",
                    "Both GPT-4 and humans perform equally well on difficult passages, but GPT-4 struggles more with simple texts.",
                    "GPT-4 significantly outperforms humans on more difficult passages, whereas performance on simple texts remains comparable.",
                    "Humans outperform GPT-4 on difficult passages, but GPT-4 performs better on simple texts."
                ],
                "correct_index": 2,
                "explanation": "The section indicates that both GPT-4 and humans perform well on simple texts, but as passage difficulty increases, GPT-4's performance exceeds that of humans by a substantial margin, with a performance ratio of about 2:1 in favor of GPT-4.",
                "difficulty_score": 0.7,
                "concept_tags": [
                    "passage difficulty",
                    "GPT-4 performance",
                    "human vs. AI comparison"
                ],
                "salience": 0.9,
                "directness": 0.8
                },
                {
                "question_text": "Which of the following best illustrates the kind of inference or generalization that demonstrates GPT-4's understanding of text as described in the section?",
                "options": [
                    "GPT-4 provides literal answers without making inferences beyond the explicit text.",
                    "GPT-4 correctly infers unstated main ideas and generalizes from specific details to broader concepts.",
                    "GPT-4 relies solely on memorized information without applying reasoning to new passages.",
                    "GPT-4 focuses only on surface-level details and fails to recognize implicit meanings."
                ],
                "correct_index": 1,
                "explanation": "The correct answer highlights GPT-4's ability to go beyond literal text by making inferences and generalizations, which are signatures of understanding discussed in the section. The other options describe behaviors that do not demonstrate inference or generalization, which are key indicators of GPT-4's deeper comprehension.",
                "difficulty_score": 0.8,
                "concept_tags": [
                    "inference",
                    "generalization",
                    "signatures of understanding"
                ],
                "salience": 1,
                "directness": 0.6
                }
            ]
            }

    QUESTIONS.update(new_block)