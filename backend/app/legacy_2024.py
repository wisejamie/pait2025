# legacy_2024.py
# --------------------------------------
# Minimal, robust 2024-style pipeline:
# - Global article text setter/getter
# - LLM outline: sections + learning objectives
# - Fuzzy anchoring of section spans to the SAME raw text
# - No pretty display transforms; reliability > aesthetics

from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Tuple, Optional

# ---- OpenAI compatibility (new & legacy SDKs) ----
try:
    # Newer SDK (openai>=1.0)
    from openai import OpenAI  # type: ignore
    _OPENAI_CLIENT = OpenAI()
    _USE_NEW_SDK = True
except Exception:
    _OPENAI_CLIENT = None
    _USE_NEW_SDK = False

try:
    # Legacy SDK (openai<1.0)
    import openai  # type: ignore
    _HAVE_LEGACY = True
except Exception:
    _HAVE_LEGACY = False

# ---- Fuzzy matching libs ----
try:
    from rapidfuzz import fuzz, process  # fast + no C dep required
    _HAVE_RAPIDFUZZ = True
except Exception:
    _HAVE_RAPIDFUZZ = False

import difflib

# ------------- Global State (like 2024) -------------
_ARTICLE_TEXT: str = ""
# You can cache the most recent outline if desired; we keep it simple:
_SECTIONS_CACHE: Optional[Dict[str, Any]] = None

# ------------- Public API (called by FastAPI wrapper) -------------

def set_article_text(text: str) -> None:
    """Set the global article text and clear caches (2024 behavior)."""
    global _ARTICLE_TEXT, _SECTIONS_CACHE
    _ARTICLE_TEXT = text or ""
    _SECTIONS_CACHE = None


def get_article_text() -> str:
    """Get the global article text (2024 behavior)."""
    return _ARTICLE_TEXT


def extract_sections(model: str = "gpt-4.1-nano", max_chars: int = 35000) -> Dict[str, Any]:
    """
    Ask the LLM for a hierarchical outline + learning objectives.
    Return:
        {
          "sections": [
            {
              "title": "...",
              "first_sentence": "...",
              "sub_sections": [
                {"title": "...", "first_sentence": "...", "sub_sections": [...]}
              ]
            },
            ...
          ],
          "learning_objectives": {
              "objectives": ["...","...",...]
          }
        }
    """
    if not _ARTICLE_TEXT.strip():
        raise ValueError("No article text has been set. Call set_article_text() first.")

    article_text = _ARTICLE_TEXT[:max_chars]
    prompt = _build_outline_prompt(article_text)

    raw = _call_llm_json(prompt, model=model)
    # Handle possible code fences:
    cleaned = _strip_code_fences(raw)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        # Sometimes the model returns a blob with JSON inside—try to yank JSON with a regex:
        json_str = _extract_first_json_object(cleaned)
        if not json_str:
            raise ValueError("Failed to parse LLM outline JSON.")
        parsed = json.loads(json_str)

    # Basic shape checks & normalization
    sections = parsed.get("sections") or []
    learning_objectives = parsed.get("learning_objectives") or {}
    _normalize_section_nodes_inplace(sections)

    return {"sections": sections, "learning_objectives": learning_objectives}


def extract_section_text(article_text: str, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Given the article_text and the hierarchical sections (with title + first_sentence),
    fuzzy-anchor each section's text span in the SAME article_text used for prompting.
    Adds a "text" field to each node.
    Returns a deep-copied, anchored sections list (doesn't mutate input).
    """
    if not article_text:
        raise ValueError("article_text is empty.")

    # Flatten sections with depth for ordered boundary computation
    flat = []
    _flatten_sections(sections, depth=0, out=flat)

    # Find start indices using first_sentence
    for node in flat:
        fs = (node.get("first_sentence") or "").strip()
        if not fs:
            node["_start_idx"] = None
            continue
        idx = _find_anchor_index(article_text, fs)
        node["_start_idx"] = idx

    # Sort by (start_idx asc, depth asc) so parents generally precede children if anchored same place
    flat_sorted = sorted(
        flat,
        key=lambda n: (n["_start_idx"] if isinstance(n.get("_start_idx"), int) else 10**12, n["_depth"])
    )

    # Compute end indices using "next node with depth <= current depth"
    for i, node in enumerate(flat_sorted):
        start = node["_start_idx"]
        if start is None:
            node["_end_idx"] = None
            continue

        # Look for the next candidate
        end = len(article_text)
        for j in range(i + 1, len(flat_sorted)):
            nxt = flat_sorted[j]
            nxt_start = nxt["_start_idx"]
            if nxt_start is None:
                continue
            if nxt["_depth"] <= node["_depth"]:
                end = nxt_start
                break
        node["_end_idx"] = end

    # Slice text and map back to original hierarchy
    id_map = {id(node): node for node in flat}
    anchored = _deepcopy_sections(sections)

    def fill_text(n: Dict[str, Any]):
        # Find the corresponding flat node (match by identity via an index built during flatten)
        # We will re-flatten anchored with identity tags to align.
        pass

    # Instead of matching identities (messy across copies), we re-flatten both structures in-order:
    anchored_flat = []
    _flatten_sections(anchored, depth=0, out=anchored_flat)

    # Map by (title, first_sentence, depth) tuple — robust enough for our usage
    def keyer(n: Dict[str, Any]) -> Tuple[str, str, int]:
        return (
            (n.get("title") or "").strip(),
            (n.get("first_sentence") or "").strip(),
            int(n.get("_depth")),
        )

    src_map = {keyer(n): n for n in flat_sorted}
    for n in anchored_flat:
        k = keyer(n)
        src = src_map.get(k)
        if not src:
            n["text"] = ""
            continue
        s, e = src.get("_start_idx"), src.get("_end_idx")
        if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(article_text):
            # Clean up whitespace a bit, but keep raw content otherwise
            chunk = article_text[s:e]
            n["text"] = _normalize_whitespace(chunk)
        else:
            n["text"] = ""

    # Remove internal helpers
    for n in anchored_flat:
        for k in ["_depth", "_start_idx", "_end_idx"]:
            if k in n:
                del n[k]

    return anchored


# ------------- Helpers: prompts, flattening, anchoring -------------

def _build_outline_prompt(article_text: str) -> str:
    """
    The 2024-style outline prompt: ask for sections with title + first_sentence,
    plus learning objectives. Keep it strict JSON.
    """
    return f"""
You are an assistant that extracts a clean section outline and learning objectives from an academic article.

Return STRICT JSON with this schema:
{{
  "sections": [
    {{
      "title": "string",
      "first_sentence": "string",
      "sub_sections": [ ... same shape recursively ... ]
    }}
  ],
  "learning_objectives": {{
    "objectives": ["string", "string", "string"]
  }}
}}

Guidelines:
- Identify the main sections and subsections that would guide a student through the paper.
- For each (sub)section, include the VERY FIRST sentence FROM THE ARTICLE as "first_sentence" EXACTLY as written (do not paraphrase).
- Keep subsections only when they’re meaningful (avoid overly deep trees).
- Keep the JSON compact and valid. Do NOT include any commentary.

ARTICLE:
\"\"\"{article_text}\"\"\"
""".strip()


def _call_llm_json(prompt: str, model: str = "gpt-4.1-nano") -> str:
    """
    Return the raw string content that SHOULD be JSON (or contain a JSON object).
    Supports both new and legacy OpenAI SDKs.
    """
    # Prefer new SDK if available
    if _USE_NEW_SDK and _OPENAI_CLIENT is not None:
        resp = _OPENAI_CLIENT.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content or ""
        return content

    # Fall back to legacy
    if _HAVE_LEGACY:
        # If OPENAI_API_KEY env is set, legacy client will pick it up
        res = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return res["choices"][0]["message"]["content"] or ""

    raise RuntimeError("OpenAI SDK not available. Install `openai` and set OPENAI_API_KEY.")


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    # remove ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_first_json_object(s: str) -> Optional[str]:
    """
    Try to extract the first {...} JSON object from a string.
    """
    start = s.find("{")
    if start == -1:
        return None
    # naive brace matching
    depth = 0
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None


def _normalize_section_nodes_inplace(nodes: List[Dict[str, Any]], depth: int = 0) -> None:
    """Ensure keys exist and track depth for later anchoring."""
    for n in nodes:
        n.setdefault("title", "")
        n.setdefault("first_sentence", "")
        n.setdefault("sub_sections", [])
        n["_depth"] = depth
        _normalize_section_nodes_inplace(n["sub_sections"], depth + 1)


def _flatten_sections(nodes: List[Dict[str, Any]], depth: int, out: List[Dict[str, Any]]) -> None:
    for n in nodes:
        n["_depth"] = depth
        out.append(n)
        subs = n.get("sub_sections") or []
        _flatten_sections(subs, depth + 1, out)


def _deepcopy_sections(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Simple JSON roundtrip for deep copy (safe since nodes are JSON-like)
    return json.loads(json.dumps(nodes))

def _normalize_with_mapping(s: str) -> tuple[str, list[int]]:
    """
    Normalize text while keeping a map from normalized indices back to raw indices.
    Rules:
      - collapse any whitespace runs (spaces/tabs/newlines) into a single space
      - join hyphenated line breaks: '-\\s*\\n\\s*' is removed entirely (word join)
    Returns:
      (normalized_string, map_norm_to_raw_index)
    """
    out_chars = []
    idx_map = []  # idx_map[i] = raw index in s for out_chars[i]
    n = len(s)
    i = 0

    while i < n:
        ch = s[i]

        # Join hyphenated line breaks: "-<ws><newline><ws>" -> '' (skip)
        if ch == '-' and i + 1 < n:
            j = i + 1
            consumed_ws_newline = False
            # consume any spaces/tabs/newlines after the hyphen
            while j < n and s[j] in (' ', '\t', '\r', '\n'):
                consumed_ws_newline = True
                j += 1
            if consumed_ws_newline:
                # drop the hyphen and the whitespace/newlines that follow
                i = j
                continue

        # Collapse whitespace runs into a single space
        if ch.isspace():
            j = i
            while j < n and s[j].isspace():
                j += 1
            out_chars.append(' ')
            idx_map.append(i)  # map this single space to the first raw index of the run
            i = j
            continue

        # Regular character
        out_chars.append(ch)
        idx_map.append(i)
        i += 1

    return ''.join(out_chars), idx_map



def _normalize_whitespace(text: str) -> str:
    # Collapse excessive internal whitespace but preserve line breaks reasonably
    # Strategy: normalize CRLF, collapse multiple spaces, trim.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # don't destroy single newlines; collapse spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)
    # collapse >2 newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _prepare_for_match(s: str) -> str:
    """Light normalization to reduce linebreak/hyphen mismatch without paraphrasing."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # join hyphenated line breaks: e.g., "infor-\nmation" -> "information"
    s = re.sub(r"-\s*\n\s*", "", s)
    # flatten newlines to spaces for matching
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _find_anchor_index(article_text: str, first_sentence: str) -> Optional[int]:
    """
    Find the exact raw start index of `first_sentence` in `article_text`.
    Strategy:
      1) Normalize BOTH strings with a reversible index map.
      2) Exact substring search in normalized space.
      3) If not found, do a coarse fuzzy search over normalized blocks and map back.
    Returns:
      raw start index in `article_text` if found with confidence; else None.
    """
    if not first_sentence:
        return None

    raw_doc = article_text

    # Build normalized forms with index maps
    doc_norm, map_doc = _normalize_with_mapping(raw_doc)
    fs_norm, _ = _normalize_with_mapping(first_sentence)

    # 1) Exact normalized match
    pos_norm = doc_norm.find(fs_norm)
    if pos_norm != -1:
        return map_doc[pos_norm]  # precise map back to raw index

    # 2) Fuzzy fallback in normalized space, then map back using map_doc
    # Split normalized doc into coarse blocks (sentences / lines)
    blocks = re.split(r"(?<=[\.\!\?\:])\s+|\n{1,}", doc_norm)
    offsets = []
    acc = 0
    for b in blocks:
        offsets.append(acc)
        acc += len(b) + 1  # add a soft separator that the split consumed

    # score helper
    def score(a: str, b: str) -> float:
        if _HAVE_RAPIDFUZZ:
            return float(fuzz.token_set_ratio(a, b))
        return 100.0 * difflib.SequenceMatcher(None, a, b).ratio()

    best_score = -1.0
    best_norm_idx = None
    for i, b in enumerate(blocks):
        sc = score(fs_norm, b)
        if sc > best_score:
            best_score = sc
            best_norm_idx = offsets[i]

    # Tighten threshold a bit since you want sharper anchoring
    if best_score >= 78.0 and best_norm_idx is not None:
        # Optional micro-refinement: try to snap to a closer local exact match near best_norm_idx
        window_radius = max(40, len(fs_norm) // 2)
        lo = max(0, best_norm_idx - window_radius)
        hi = min(len(doc_norm), best_norm_idx + len(fs_norm) + window_radius)
        local = doc_norm[lo:hi]
        local_pos = local.find(fs_norm)
        if local_pos != -1:
            pos_norm = lo + local_pos
        else:
            pos_norm = best_norm_idx

        # Map normalized position back to raw
        return map_doc[min(pos_norm, len(map_doc) - 1)]

    return None
