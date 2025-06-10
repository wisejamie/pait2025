import re
from rapidfuzz import fuzz

class SectionExtractionError(Exception):
    """Raised when section text could not be extracted reliably."""
    def __init__(self, section_title, message=None):
        self.section_title = section_title
        self.message = message or f"Failed to extract text for section: {section_title}"
        super().__init__(self.message)


def extract_section_text(article_text, sections):
    """
    Enriches each section with its full text, extracted from the article_text.
    Matches based on fuzzy matching of first sentences.
    """

    def find_fuzzy_sentence(article_words, first_sentence, chunk_size=5, chunk_match_threshold=80, full_sentence_threshold=70):
        first_sentence_words = first_sentence.split()

        for i in range(len(article_words) - chunk_size + 1):
            article_chunk = article_words[i:i + chunk_size]
            sentence_chunk = first_sentence_words[:chunk_size]

            chunk_match = fuzz.ratio(' '.join(article_chunk), ' '.join(sentence_chunk))
            if chunk_match >= chunk_match_threshold:
                article_window = ' '.join(article_words[i:i + len(first_sentence_words) + 10])
                full_match = fuzz.ratio(article_window, first_sentence)
                if full_match >= full_sentence_threshold:
                    return i

        return None

    article_words = re.findall(r'\S+|\n', article_text)

    def enrich(section_list):
         for idx, section in enumerate(section_list):
            first_sentence = section["first_sentence"]
            start_idx = find_fuzzy_sentence(article_words, first_sentence)

            if start_idx is None:
                raise SectionExtractionError(section.get("title", "Unknown"))

             # Find end index using the next section's start sentence
            next_idx = None
            if idx + 1 < len(section_list):
                next_first = section_list[idx + 1]["first_sentence"]
                next_idx = find_fuzzy_sentence(article_words, next_first)

            end_idx = next_idx if next_idx is not None else len(article_words)
            section["text"] = ' '.join(article_words[start_idx:end_idx])

            # Recurse into sub-sections
            if "sub_sections" in section and isinstance(section["sub_sections"], list):
                enrich(section["sub_sections"])

    enrich(sections)
    return sections