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

# def build_simplify_prompt(text: str) -> str:
#     # Simple MVP prompt—can refine later
#     return (
#         "Can you please rewrite the following text with simpler language, preserving original meaning and paragraph structure."
#         "Do *not* include any introductory phrases or preamble words —output *only* the rewritten text:\n\n"
#         f"{text}"
#     )

def build_transform_prompt(text: str, mode: str) -> str:
    if mode == "simplify":
        # reuse your simple prompt or inline it
        return (
            "Rewrite the following text using simpler language, preserving original meaning and paragraph structure. "
            "Do not include any introductory phrases or preamble words—output only the rewritten text:\n\n"
            f"{text}"
        )
    elif mode == "elaborate":
        return (
            "Rewrite the following text, and whenever there is a technical term or complex idea, expand it with a clear definition or more detailed explanation."
            "Preserve paragraph structure:"
            "Do not include any introductory phrases or preamble words—output only the rewritten text:\n\n"
            f"{text}"
        )
    elif mode == "distill":
        return (
            "Rewrite the following text focusing only on its core ideas; omit examples and side-details so it is as concise as possible."
            "Use complete sentences and preserve paragraph structure:"
            "Do not include any introductory phrases or preamble words—output only the rewritten text:\n\n"
            f"{text}"
        )
    else:
        raise ValueError(f"Unknown transform mode: {mode}")