import re
from rapidfuzz import fuzz
import io
import PyPDF2
from app.utils.text_cleaning import clean_text
import fitz

class SectionExtractionError(Exception):
    """Raised when section text could not be extracted reliably."""
    def __init__(self, section_title, message=None):
        self.section_title = section_title
        self.message = message or f"Failed to extract text for section: {section_title}"
        super().__init__(self.message)

def build_section_extraction_prompt(raw_text: str) -> str:
    return f"""
You are an AI Tutor tasked with analyzing an academic article and identifying its informative sections and sub-sections. Your goal is to help students understand this document by organizing it and identifying key learning goals.

Instructions:

1. Identify a hierarchical list of sections and sub-sections that contain meaningful content.
   - **Exclude** the abstract, references, acknowledgments, and appendix.
   - Use only content-rich parts of the article.
   
2. For each section and sub-section, include:
   - "title": A short, informative label summarizing its content.
   - "first_sentence": The **first 15 words** of the section, exactly as they appear in the article.
   - "sub_sections": A list of nested sub-sections (each with the same structure).
   - "learning_objectives": A list of **2–5** specific objectives a user should achieve after reading this section.
   
   Section-level objectives should:
   - Focus on **local comprehension** — that is, concepts, arguments, methods, or findings presented in the section.
   - Be **factually correct** and **clearly grounded in the section text** (either explicitly stated or strongly supported).
   - Be phrased as **measurable learning goals** — not vague outcomes like “Understand X” or “Know about Y”.

   Here are some examples of strong learning objectives formats:
   - “Define <concept> to mean <correct definition or explanation>”
   - “Describe <process or finding> as <correct mechanism or outcome>”
   - “Explain how <cause> leads to <effect>, as shown in the text”
   - “Compare <A> and <B> with respect to <dimension>, noting that <key distinction>”

3. Also include a separate list of **global learning objectives** for the document as a whole.
   - These should reflect the **big-picture educational goals** across the entire article.
   - Provide 3–5 objectives, each describing an important insight or skill the user should gain.

Return **only valid JSON** in the following structure:

{{
  "sections": [
    {{
      "title": "<Section Title>",
      "first_sentence": "<First 15 words>",
      "learning_objectives": ["<Objective 1>", "<Objective 2>", "<Objective 3>"],
      "sub_sections": [
        {{
          "title": "<Subsection Title>",
          "first_sentence": "<First 15 words>",
          "learning_objectives": ["<Objective 1>", "<Objective 2>"],
          "sub_sections": []
        }}
      ]
    }},
    ...
  ],
  "learning_objectives": {{
    "1": "<Global Objective 1>",
    "2": "<Global Objective 2>",
    ...
  }}
}}

Here is the article:
\"\"\"
{raw_text}
\"\"\"
"""


def find_fuzzy_sentence(article_words, first_sentence,
                        chunk_size=5,
                        chunk_match_threshold=80,
                        full_sentence_threshold=70):
    """
    Return the start index of the best match for `first_sentence` in `article_words`,
    or None if no acceptable match is found.
    """
    first_words = first_sentence.split()[:chunk_size]
    best = None  # tuple (full_score, index)
    for i in range(len(article_words) - chunk_size + 1):
        chunk = article_words[i : i + chunk_size]
        score = fuzz.ratio(" ".join(chunk), " ".join(first_words))
        if score >= chunk_match_threshold:
            # try full-sentence match window
            window = article_words[i : i + len(first_sentence.split())]
            full_score = fuzz.ratio(" ".join(window), first_sentence)
            if full_score >= full_sentence_threshold:
                return i
            if best is None or full_score > best[0]:
                best = (full_score, i)
    # fallback if partial matches exist
    if best and best[0] >= full_sentence_threshold * 0.8:
        return best[1]
    return None


def flatten_sections(sections, depth=0):
    """
    Turn nested sections into a flat list in reading order.
    Annotate each with a `_depth` field.
    """
    flat = []
    for sec in sections:
        sec['_depth'] = depth
        flat.append(sec)
        for sub in sec.get('sub_sections', []):
            flat.extend(flatten_sections([sub], depth + 1))
    return flat


def extract_section_text(article_text: str, sections: list) -> list:
    """
    Enrich the nested `sections` structure by populating each dict's `text` field.

    1. Tokenize the article into `article_words`.
    2. Flatten sections (recording depth).
    3. Locate each section's start via fuzzy matching of its `first_sentence`.
    4. For each section in the flat list, its end is the next item's start
       at depth <= its own depth (or end of article).
    5. Slice `article_words[start:end]`, rebuild text with spaces and newlines,
       and assign back to each section.

    Returns the original nested `sections` with `text` fields filled.
    """
    # 1) tokenize
    article_words = re.findall(r'\S+|\n', article_text)

    # 2) flatten
    flat_secs = flatten_sections(sections)

    # 3) find start indices
    for sec in flat_secs:
        idx = find_fuzzy_sentence(article_words, sec['first_sentence'])
        if idx is None:
            raise SectionExtractionError(sec['title'])
        sec['_start_idx'] = idx

    # 4) extract spans
    for i, sec in enumerate(flat_secs):
        start = sec['_start_idx']
        depth = sec['_depth']
        # default to end of article
        end = len(article_words)
        # look ahead for next boundary at <= depth
        for nxt in flat_secs[i+1:]:
            if nxt['_depth'] <= depth:
                end = nxt['_start_idx']
                break

        # 5) slice and assign, preserving spaces
        span_tokens = article_words[start:end]
        text = ''
        for tok in span_tokens:
            if tok == '\n':
                text += '\n'
            else:
                text += tok + ' '
        sec['text'] = text.strip()

        # cleanup intermediate keys
        del sec['_start_idx'], sec['_depth']

    return sections


# def extract_text_from_pdf(file) -> str:
#     """Extracts all text from a PDF file-like object."""
#     with io.BytesIO(file.read()) as file_stream:
#         reader = PyPDF2.PdfReader(file_stream)
#         text = ''
#         for page in reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + '\n'
#     text = clean_text(text)
#     return text
def extract_text_from_pdf(file):
    """
    Extracts raw text from a PDF using PyMuPDF.
    Attempts to preserve paragraph structure by pulling text block-wise from each page.
    """
    file_bytes = file.read()  # Read file contents into bytes
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_blocks = []

    for page_num, page in enumerate(doc, start=1):
        # Use layout-aware text extraction
        page_text = page.get_text("text")
        text_blocks.append(page_text.strip())

    # Combine pages with double newlines to encourage paragraph breaks
    full_text = "\n\n".join(text_blocks)
    full_text = clean_text(full_text)
    return full_text