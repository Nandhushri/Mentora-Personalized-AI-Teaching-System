"""
visuals/visual_selector.py
---------------------------
The Visual Decision Engine (Phase 4).

Split of responsibility (the core design rule for this phase):
    DECISION  -> this module asks Gemini "what visual would help, and with
                 what data?" and gets back plain JSON fields (numbers,
                 short strings, short lists).
    RENDERING -> visuals/visual_renderer.py takes that already-validated
                 data and draws it using fixed, hardcoded Streamlit/
                 matplotlib calls that WE control.

The model is NEVER asked for and NEVER allowed to supply executable code,
raw HTML/JS, or plotting/rendering instructions of any kind - only plain
data fields in a fixed schema. Everything coming back from Gemini is
re-validated here before it's treated as safe to render; anything
malformed, incomplete, or nonsensical is dropped (visual_type "none"),
never raised as an error - a broken visual should never block a lesson.
"""

from teacher import _call_gemini_json, _format_context_chunks, GeminiResponseError

VISUAL_TYPES = ["equation", "graph", "diagram", "code", "table", "image", "none"]

# ---------------------------------------------------------------------------
# Schema Gemini must respond in. Flat rather than "one field per type with
# nested sub-objects", because structured-output schemas need a fixed shape
# - the model just leaves irrelevant fields empty. Everything is validated
# and reshaped into a clean nested structure afterward (see _VALIDATORS).
# ---------------------------------------------------------------------------

VISUAL_SELECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "visual_type": {"type": "string", "enum": VISUAL_TYPES},
        "title": {"type": "string"},
        "visual_explanation": {
            "type": "string",
            "description": "One short sentence on what the visual shows and why it helps.",
        },
        "equation": {
            "type": "string",
            "description": "Only for visual_type=equation. Plain math notation, e.g. 'F = ma'.",
        },
        "graph_x_label": {"type": "string"},
        "graph_y_label": {"type": "string"},
        "graph_points": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": (
                "Only for visual_type=graph. List of [x, y] numeric pairs. "
                "Only use values that are actually known/supported - never invent numbers."
            ),
        },
        "table_columns": {"type": "array", "items": {"type": "string"}},
        "table_rows": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}},
            "description": "Only for visual_type=table. Each row must have the same length as table_columns.",
        },
        "code_language": {"type": "string"},
        "code_snippet": {"type": "string", "description": "Only for visual_type=code."},
        "diagram_nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "label": {"type": "string"}},
                "required": ["id", "label"],
            },
            "description": "Only for visual_type=diagram. Short labeled steps/boxes.",
        },
        "diagram_connections": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}},
            "description": "Only for visual_type=diagram. Each item is [from_node_id, to_node_id].",
        },
        "image_query": {
            "type": "string",
            "description": "Only for visual_type=image. A short description of what the image should show.",
        },
    },
    "required": ["visual_type"],
}

# Deliberately guidance, not hardcoded per-subject logic - the prompt-level
# equivalent of "use deterministic rules where appropriate and LLM
# reasoning where useful" without trying to enumerate every subject.
SUBJECT_AWARE_GUIDANCE = """
General patterns for choosing a visual (use judgment - these are patterns, not rigid rules):
- Mathematics: equations -> equation; numerical relationships -> graph or table; geometry -> diagram; functions -> graph.
- Physics: formulas -> equation; physical systems, forces, or processes -> diagram; motion over time -> graph.
- Biology: a labeled structure (a cell, an organ, an organism) is usually clearest as an IMAGE with
  labels; a process or cycle (e.g. photosynthesis steps, a life cycle) is usually clearer as a diagram;
  comparisons -> table.
- Geography: physical locations, regions, or spatial relationships -> image (a map); comparisons -> table.
- History: chronological sequences -> diagram (as a simple ordered flow); comparisons -> table; a
  specific historical place or artifact -> image only if genuinely illustrative.
- Programming: source code -> code; algorithms or execution flow -> diagram; data structures -> diagram or table.
- Data Science: numerical data -> table; trends -> graph; a pipeline/workflow -> diagram; code -> code.

On images specifically: choose "image" only when an actual picture would teach something a diagram,
equation, or text genuinely can't - labeled anatomy, a map, a real-world object's appearance. Do NOT
choose "image" for things a diagram already handles well (algorithms, workflows, system architecture,
class/object relationships, process flows, physics force diagrams) - keep using "diagram" for those,
per the existing diagram renderer. Never choose "image" just because the subject sounds visual.

If nothing above fits, or the concept is abstract and a visual wouldn't genuinely add value beyond
the text explanation, return visual_type = "none". Do not force a visual just to have one.
"""


def _build_prompt(concept: dict, explanation_text: str, profile: dict,
                   context_chunks: list, subject_hint: str) -> str:
    if context_chunks:
        context_block = f"""
RETRIEVED DOCUMENT CONTEXT (if the visual you choose needs specific facts or
numbers - e.g. an equation or graph - only use values that actually appear
here. Do not invent numbers. If the data needed for a good visual isn't
available here, choose a different visual type, or "none"):
{_format_context_chunks(context_chunks)}
"""
    else:
        context_block = ""

    level = profile.get("level", "Beginner")
    level_guidance = (
        "Keep it simple: minimal information, a basic equation/graph/diagram - nothing overloaded."
        if level == "Beginner"
        else "You may include more technical detail appropriate to this level."
    )

    return f"""
You are deciding whether a visual aid would help a student understand a
concept that was just taught, and if so, exactly which type and what data
it should contain.

Concept: {concept['name']}
Subject area (if known): {subject_hint or "unknown - infer it from the concept"}
Explanation just given to the student:
{explanation_text}

Student level: {level}
{level_guidance}
{context_block}
{SUBJECT_AWARE_GUIDANCE}

Choose the SINGLE visual type that would most help, or "none" if no visual
would genuinely add value beyond the text explanation. Only fill in the
fields relevant to the visual_type you choose; leave every other field
blank or empty.

Respond only with JSON matching the required schema.
"""


def _validate_equation(raw: dict):
    equation = (raw.get("equation") or "").strip()
    if not equation:
        return None
    return {"equation": equation}


def _validate_graph(raw: dict):
    clean_points = []
    for p in raw.get("graph_points") or []:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            try:
                clean_points.append([float(p[0]), float(p[1])])
            except (TypeError, ValueError):
                continue
    if len(clean_points) < 2:
        return None  # not enough real data for a meaningful graph - don't fabricate the rest
    return {
        "x_label": (raw.get("graph_x_label") or "x").strip() or "x",
        "y_label": (raw.get("graph_y_label") or "y").strip() or "y",
        "points": clean_points,
    }


def _validate_table(raw: dict):
    columns = [str(c).strip() for c in (raw.get("table_columns") or []) if str(c).strip()]
    if not columns:
        return None
    clean_rows = []
    for row in raw.get("table_rows") or []:
        if isinstance(row, (list, tuple)) and len(row) == len(columns):
            clean_rows.append([str(v) for v in row])
    if not clean_rows:
        return None
    return {"columns": columns, "rows": clean_rows}


def _validate_code(raw: dict):
    snippet = (raw.get("code_snippet") or "").strip()
    if not snippet:
        return None
    return {"language": (raw.get("code_language") or "text").strip() or "text", "code": snippet}


def _validate_diagram(raw: dict):
    nodes = []
    ids = set()
    for n in raw.get("diagram_nodes") or []:
        if isinstance(n, dict) and str(n.get("id", "")).strip() and str(n.get("label", "")).strip():
            node_id = str(n["id"]).strip()
            nodes.append({"id": node_id, "label": str(n["label"]).strip()})
            ids.add(node_id)
    if len(nodes) < 2:
        return None  # not really a diagram with fewer than 2 nodes

    connections = []
    for c in raw.get("diagram_connections") or []:
        if isinstance(c, (list, tuple)) and len(c) == 2 and c[0] in ids and c[1] in ids:
            connections.append([str(c[0]), str(c[1])])

    return {"nodes": nodes, "connections": connections}


def _validate_image(raw: dict):
    query = (raw.get("image_query") or raw.get("title") or "").strip()
    if not query:
        return None
    return {"query": query}


_VALIDATORS = {
    "equation": _validate_equation,
    "graph": _validate_graph,
    "table": _validate_table,
    "code": _validate_code,
    "diagram": _validate_diagram,
    "image": _validate_image,
}


def select_visual(concept: dict, explanation_text: str, profile: dict,
                   context_chunks: list = None, subject_hint: str = None) -> dict:
    """
    Ask Gemini whether a visual would help teach this concept, and if so,
    what kind and what data it needs. Returns a clean, already-validated
    dict ready for visuals.visual_renderer.render_visual(), or None if no
    visual is appropriate, the model didn't return anything usable, or the
    call failed for any reason.

    This function is designed to never raise - a broken or unavailable
    visual should never take down the teaching loop. Callers can call it
    directly without wrapping it in their own try/except (though app.py
    does anyway, defensively).
    """
    try:
        prompt = _build_prompt(concept, explanation_text, profile, context_chunks, subject_hint)
        raw = _call_gemini_json(prompt, VISUAL_SELECTOR_SCHEMA, temperature=0.3)
    except GeminiResponseError:
        return None
    except Exception:
        return None

    visual_type = (raw.get("visual_type") or "none").strip().lower()
    if visual_type == "none" or visual_type not in _VALIDATORS:
        return None

    visual_data = _VALIDATORS[visual_type](raw)
    if visual_data is None:
        return None  # malformed or incomplete data for this type - skip gracefully

    return {
        "visual_type": visual_type,
        "title": (raw.get("title") or concept.get("name") or "").strip(),
        "visual_data": visual_data,
        "visual_explanation": (raw.get("visual_explanation") or "").strip(),
    }
