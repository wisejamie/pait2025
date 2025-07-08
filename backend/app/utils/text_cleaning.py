import re

def clean_text(raw_text):
    cleaned_lines = []
    buffer = []

    for line in raw_text.splitlines():
        line = line.strip()

        # 2) skip pure-page-number lines
        if re.fullmatch(r'\d+', line):
            continue

        if not line:
            # Empty line = paragraph break
            if buffer:
                cleaned_lines.append(" ".join(buffer))
                buffer = []
        else:
            # Join lines unless the previous line ends with punctuation (strong stop)
            if buffer and re.search(r"[.:;!?]$", buffer[-1]):
                cleaned_lines.append(" ".join(buffer))
                buffer = [line]
            else:
                buffer.append(line)

    if buffer:
        cleaned_lines.append(" ".join(buffer))

    return "\n\n".join(cleaned_lines)
