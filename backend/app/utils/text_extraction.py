import re
from rapidfuzz import fuzz
import io
import PyPDF2

class SectionExtractionError(Exception):
    """Raised when section text could not be extracted reliably."""
    def __init__(self, section_title, message=None):
        self.section_title = section_title
        self.message = message or f"Failed to extract text for section: {section_title}"
        super().__init__(self.message)


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


def extract_text_from_pdf(file) -> str:
    """Extracts all text from a PDF file-like object."""
    with io.BytesIO(file.read()) as file_stream:
        reader = PyPDF2.PdfReader(file_stream)
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text