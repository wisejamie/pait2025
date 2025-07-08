def build_summary_prompt(text: str, level: str) -> str:
    if level == "tldr":
        return f"Summarize the following text in one concise sentence:\n\n{text}"
    elif level == "short":
        return f"Summarize the following text in 2–3 sentences:\n\n{text}"
    elif level == "bullets":
        return f"Summarize the following text in 3–5 bullet points, focusing on the main ideas:\n\n{text}"
    elif level == "simple":
        return f"Explain the following text in simple, clear language that a 12-year-old could understand:\n\n{text}"
    else:
        raise ValueError("Invalid summary level")
