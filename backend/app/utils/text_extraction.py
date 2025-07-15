import re
from rapidfuzz import fuzz

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
                        full_sentence_threshold=90):
    """
    Return the start index of the best match for `first_sentence` in `article_words`,
    or None if no acceptable match is found.
    """
    print(f"\n🔍 Matching sentence: {first_sentence}")
    first_words = first_sentence.split()[:chunk_size]
    best = None  # tuple (full_score, index)

    for i in range(len(article_words) - chunk_size + 1):
        chunk = article_words[i : i + chunk_size]
        chunk_score = fuzz.ratio(" ".join(chunk), " ".join(first_words))
        if chunk_score >= chunk_match_threshold:
            window = article_words[i : i + len(first_sentence.split())]
            full_score = fuzz.ratio(" ".join(window), first_sentence)
            print(f"  ✅ Chunk match @ {i}: chunk='{chunk}' | full_score={full_score}")

            if full_score >= full_sentence_threshold:
                print(f"  🎯 Found full match @ index {i}")
                return i

            if best is None or full_score > best[0]:
                best = (full_score, i)

        elif chunk_score >= chunk_match_threshold - 30:
            print("lower thresh")
            window = article_words[i : i + len(first_sentence.split())]
            full_score = fuzz.ratio(" ".join(window), first_sentence)
            print(f"  ✅ Chunk match @ {i}: chunk='{chunk}' | full_score={full_score}")

            if full_score >= full_sentence_threshold:
                print(f"  🎯 Found full match @ index {i}")
                return i

            if best is None or full_score > best[0]:
                best = (full_score, i)


    if best:
        print(f"  ⚠️ Best partial match score: {best[0]} at index {best[1]}")

    if best and best[0] >= full_sentence_threshold * 0.6:
        print(f"  ⬅️ Accepting fallback partial match @ index {best[1]}")
        return best[1]

    print("  ❌ No suitable match found.")
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
    Enrich the nested `sections` structure by populating each dict's `text` field,
    include any preceding text in the same paragraph, and
    also include any Markdown headings immediately above that paragraph.
    """
    # 1. Tokenize
    article_words = re.findall(r'\S+|\n', article_text)
    print(f"\n📝 Article tokenized into {len(article_words)} words")

    # 2. Flatten
    flat_secs = flatten_sections(sections)
    print(f"📚 Found {len(flat_secs)} sections (flattened)")

    # helper: back up to paragraph start
    def find_para_start(idx):
        p = idx
        # while p >= 2 and not (article_words[p-1] == '\n' and article_words[p-2] == '\n'):
        while p >= 2 and not (article_words[p-1] == '\n'):
            p -= 1
        return p

    # helper: include any header lines (starting with '#') immediately above
    def include_preceding_headers(p: int, article_words: list[str]) -> int:
        """
        Starting from token‐index p (the paragraph start), walk backward,
        skipping blank lines, and include every Markdown header line
        (lines starting with '#') until you hit non‐header content.
        Returns the new start index.
        """
        curr = p
        while True:
            # 1) find the end of the previous line
            prev_nl = next((i for i in range(curr - 1, -1, -1) if article_words[i] == '\n'), None)
            if prev_nl is None:
                break

            # 2) find the start of that line
            prev2 = next((i for i in range(prev_nl - 1, -1, -1) if article_words[i] == '\n'), -1)

            # 3) reconstruct that line
            line_tokens = article_words[prev2 + 1 : prev_nl]
            line = ''.join(tok if tok == '\n' else tok + ' ' for tok in line_tokens).strip()

            # 4) if it’s blank, skip it and continue looking higher
            if not line:
                curr = prev2 + 1
                continue

            # 5) if it’s a Markdown header, include it & keep going
            if re.match(r'^\s*#+\s+', line):
                curr = prev2 + 1
                continue

            # 6) otherwise, stop
            break

        return curr

    # 3. Fuzzy match first sentences, then back up
    for sec in flat_secs:
        print(f"\n--- Matching section: '{sec['title']}' ---")
        idx = find_fuzzy_sentence(article_words, sec['first_sentence'])
        if idx is None:
            print(f"❌ Could not find match for: {sec['first_sentence']}")
            raise SectionExtractionError(sec['title'])

        # 3a) back up to start of paragraph
        para_start = find_para_start(idx)
        # 3b) then include any headers directly above
        start_idx = include_preceding_headers(para_start, article_words)

        sec['_start_idx'] = start_idx
        print(f"✅ Section '{sec['title']}' start at token {start_idx}"
              f" (para was {para_start}, match was {idx})")

    # 4. Extract spans
    for i, sec in enumerate(flat_secs):
        start = sec['_start_idx']
        depth = sec['_depth']
        # default end at EOF
        end = len(article_words)
        # find next same-or-higher section
        for nxt in flat_secs[i+1:]:
            if nxt['_depth'] <= depth:
                end = nxt['_start_idx']
                break

        # reassemble
        span_tokens = article_words[start:end]
        text = ''.join('\n' if tok=='\n' else tok+' ' for tok in span_tokens).strip()
        sec['text'] = text
        print(f"✂️ Extracted '{sec['title']}' → {len(text)} chars")

        # cleanup
        del sec['_start_idx'], sec['_depth']

    return sections