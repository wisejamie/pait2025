import json
import re
import openai
import os
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocItemLabel, DocumentStream, InputFormat

load_dotenv()
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def filter_non_content_md(blocks: list[str]) -> list[str]:
    """
    Ask GPT to remove any block that is definitely non-content:
    running headers/footers, author lines, emails, references, acknowledgments, footnotes.
    This version robustly handles unexpected or malformed GPT outputs.
    """
    entries = [{"idx": i, "snippet": blk[:200]} for i, blk in enumerate(blocks)]
    allowed_reasons = [
        "author line",
        "affiliation line",
        "publication info",
    ]
    reasons_md = "\n".join(f"- {r}" for r in allowed_reasons)
    prompt = f"""
You are an AI assistant. You will be given a JSON array “blocks” of Markdown text snippets, each one corresponds to a chunk of a document. Some chunks do not have to do with the ongoing narrative of the document. Your job is to remove some of these non-content blocks. **Only** remove blocks that you are **100% certain** are one of:
{reasons_md}

Your job is to **only** remove those blocks you are **100% certain** are non-content.
Do **not** remove any block containing actual article prose, tables, figures, or statistics.
If you are unsure of a block's class, you should not remove it.

Input format:
{{"blocks": {json.dumps(entries, indent=2)} }}

Return valid JSON:
{{  
  "remove": [
    {{ "idx": <integer>, "reason": "<one of the allowed reasons>" }},
    …
  ]
}}
""".strip()

    # Call GPT
    resp = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    # Attempt to parse GPT's JSON response
    try:
        out = json.loads(resp.choices[0].message.content)
    except (ValueError, TypeError) as e:
        print(f"Warning: Unable to parse GPT response: {e}. Keeping all blocks.")
        return blocks
    
    removal_info = out.get("remove", [])
    print("\n🗑️ GPT decided to remove the following blocks:")
    for item in removal_info:
        idx = item.get("idx")
        reason = item.get("reason", "<no reason>")
        snippet = blocks[idx][:200].replace("\n", " ")
        print(f"  • idx={idx:3} reason={reason!r}\n    snippet={snippet!r}")

    # 3) Build set of indices to drop
    remove_indices = {item["idx"] for item in removal_info if isinstance(item, dict) and isinstance(item.get("idx"), int)}

    # 4) Filter and return
    kept_blocks = [blk for i, blk in enumerate(blocks) if i not in remove_indices]
    print(f"\n🔖 Kept {len(kept_blocks)}/{len(blocks)} blocks after GPT filtering.\n")
    return kept_blocks

def merge_continuations(blocks: list[str]) -> list[str]:
    """
    If a block ends without terminal punctuation and the next starts lowercase,
    merge them as a mid-sentence continuation.
    """
    merged = []
    for blk in blocks:
        if merged:
            prev = merged[-1].rstrip()
            if prev and prev[-1] not in ".!?\"" and blk and (blk[0].islower() or prev[-1] in ";,-" or blk[0].isdigit()):
                merged[-1] = prev + " " + blk.lstrip()
                continue
        merged.append(blk)
    return merged


from io import BytesIO
from typing import Union

def preprocess_pdf(path_or_bytes: Union[str, bytes], chunk_size: int = 100) -> str:
    """
    1) Load the PDF via Docling
    2) Extract full Markdown with Docling
    3) Split into blocks, toss obvious non-content
    4) In batches of `chunk_size` blocks, filter via GPT
    5) Merge sentence continuations
    6) Return the final Markdown string
    """
    # 1) prepare Docling “source”
    if isinstance(path_or_bytes, (bytes, bytearray)):
        source = DocumentStream(
            name="upload.pdf",
            stream=BytesIO(path_or_bytes),
            format=InputFormat.PDF
        )
    else:
        # you can pass a file path or URL directly
        source = path_or_bytes

    # 2) convert and extract Markdown
    converter = DocumentConverter()
    result = converter.convert(source)
    labels_to_keep = {    
        DocItemLabel.SECTION_HEADER,  # headings (##, ###, etc.)
        DocItemLabel.PARAGRAPH,       # logical paragraphs
        DocItemLabel.TEXT,            # stray text runs
        DocItemLabel.LIST_ITEM,       # bullet/numbered list entries
        DocItemLabel.CODE,            # code blocks, if any
        DocItemLabel.TABLE,           # tables as markdown
        DocItemLabel.FORMULA,         # math formulas (e.g. LaTeX)
        DocItemLabel.CAPTION,         # figure/table captions
        DocItemLabel.CHART,           # chart items (if you want them)
    }

    md = result.document.export_to_markdown(labels=labels_to_keep,
    include_annotations=False)
    # └── this is your full Markdown export :contentReference[oaicite:0]{index=0}

    md = re.sub(r'(?m)^-\s', r'\n- ', md)
    # └── this ensures that list items are properly separated

    # 3) split & drop obvious non-content
    blocks = [b.strip() for b in md.split("\n\n") if b.strip()]

    # find the index of the first “References” or “Appendix” block
    cutoff = next(
        (
        i for i, b in enumerate(blocks)
        if re.match(r'^(references|appendix|appendices|acknowledgements|acknowledgments)\b', b.strip(), re.IGNORECASE)
        ),
        None
    )
    if cutoff is not None:
        blocks = blocks[:cutoff]

    # 4) filter in GPT-sized batches
    kept = []
    for start in range(0, len(blocks), chunk_size):
        batch = blocks[start : start + chunk_size]
        filtered = filter_non_content_md(batch)
        kept.extend(filtered)

    # 5) stitch broken continuations
    merged = merge_continuations(kept)

    # 6) return final Markdown
    return "\n\n".join(merged)

def prune_tree(sections: list[dict], max_depth: int = 1, _depth: int = 0) -> list[dict]:
    """
    Recursively strip out any subsections below `max_depth`.
    - sections: list of section-dicts with a "subsections" key (which may be absent or empty).
    - max_depth: deepest level to keep (0 = only top-level, 1 = keep top+their direct children).
    """
    pruned = []
    for sec in sections:
        # Copy all keys except subsections for now
        new_sec = {k: v for k, v in sec.items() if k != "sub_sections"}
        if _depth < max_depth:
            # Only recurse if we haven't hit max depth
            children = sec.get("sub_sections", [])
            new_sec["sub_sections"] = prune_tree(children, max_depth, _depth + 1)
        else:
            # At depth == max_depth: drop all deeper children
            new_sec["sub_sections"] = []
        pruned.append(new_sec)
    # print_structure(sections)
    # print_structure(pruned)
    return pruned

def print_structure(sections: list[dict], level: int = 0) -> None:
    """
    Pretty-print a list of section dicts (with keys "title" and optional "subsections").
    Top-level titles print without a bar; each deeper level gets an additional indent and bar.
    """
    for sec in sections:
        # build prefix: no bar at level 0, then '| ' repeated per level
        prefix = ""
        if level > 0:
            prefix = ("|   " * (level - 1)) + "|-- "
        # print the title
        print(f"{prefix}{sec.get('title', '<No Title>')}")
        # recurse into subsections
        children = sec.get("sub_sections", [])
        if children:
            print_structure(children, level + 1)


