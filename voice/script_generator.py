"""
voice/script_generator.py
---------------------------
Converts educational content into a natural, speakable teacher script.

This is a deliberate layer between "what the Teaching Agent produced" and
"what actually gets sent to TTS" - per the Phase 5 spec, raw explanation
text (which may contain markdown, citations, page numbers, or phrasing
that reads fine but sounds robotic out loud) is never sent to TTS directly.
This module's only job is producing plain, natural spoken text.

Reuses teacher.py's existing Gemini call/retry machinery (call_gemini_json)
rather than duplicating client setup - this module never touches the
Gemini client directly.
"""

from teacher import call_gemini_json, GeminiResponseError

SPEECH_SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "script": {"type": "string"},
    },
    "required": ["script"],
}

# Explicit "never do this" list, mirrored directly from the spec, so the
# instruction is unambiguous rather than relying on the model to infer it.
SCRIPT_RULES = """
Rules for the spoken script:
- Plain natural speech only. No markdown, no bullet points, no JSON, no headers.
- No citations, page numbers, or source references (e.g. never say "Source: Page 12"
  or "according to the retrieved context" or "according to the document").
- Never refer to "the system", "the AI", "the database", or any UI element
  (buttons, labels, progress bars). Just teach.
- Do not read out loud anything that isn't meant to be heard, like scores or metadata.
- Prefer short, clear sentences over long compound ones - it's easier to follow by ear
  than by eye. Use natural punctuation (commas, periods) so pauses land in the right
  places when read aloud.
- Avoid unnecessary repetition - say each idea once, clearly.
- Keep it reasonably short: a few natural sentences, not a lecture.
- Technical accuracy comes first. Sound like a real tutor, not a chatbot being cute -
  don't add filler enthusiasm or forced conversational flourishes at the expense of
  precision. A little warmth is good; oversimplifying to the point of being wrong is not.
- Adapt vocabulary and pacing to the student's level (a beginner needs plainer words
  and more grounding than an advanced student).
- Do not invent facts, numbers, or claims that are not already in the explanation
  provided to you - your job is to make it sound natural, not to add new content.
"""

# How to talk about a visual out loud depends heavily on what kind it is -
# reading a graph's raw data points aloud is useless, but describing its
# trend is exactly what a teacher would do at a whiteboard. One block per
# visual_type, selected below - never generic advice for every type.
_VISUAL_TYPE_GUIDANCE = {
    "equation": (
        "The visual is an equation. Verbally decompose it - say what each symbol or "
        "term stands for and how they relate (e.g. \"F is force, M is mass, and A is "
        "acceleration\"), not just read the symbols as letters."
    ),
    "graph": (
        "The visual is a graph. Explain what each axis represents, describe the overall "
        "trend (rising, falling, constant, etc.), and point out what the student should "
        "actually notice - not a list of individual data points."
    ),
    "diagram": (
        "The visual is a diagram. Explain the important components, how they relate to "
        "each other, and the direction/flow between them if there is one - walk through "
        "it the way a teacher would while pointing at a whiteboard."
    ),
    "table": (
        "The visual is a table. Describe the pattern or comparison it shows in words, "
        "not a row-by-row recitation of every cell."
    ),
    "code": (
        "The visual is a code snippet. Explain the important logic and what the code "
        "accomplishes - do not read the code line by line or narrate syntax."
    ),
}


def _language_instruction(language: str) -> str:
    language = (language or "English").strip()
    if language.lower() == "hindi":
        return "Speak naturally in Hindi."
    if language.lower() == "hinglish":
        return (
            "Speak naturally in Hinglish (mix Hindi and English the way a real Indian "
            "tutor would in conversation, e.g. \"Acceleration ka matlab hai velocity kitni "
            "rapidly change ho rahi hai.\"). Do not translate technical terms unnecessarily - "
            "keep terms like force, mass, acceleration, function, etc. in English."
        )
    return "Speak naturally in English."


def generate_speech_script(explanation: str, concept: str, student_level: str,
                            language: str = "English", visual_explanation: str = None,
                            visual_type: str = None) -> str:
    """
    Turn a written explanation into a natural spoken teacher script.

    `explanation` should be plain teaching content only - the caller is
    responsible for never passing citation text (e.g. "Source: Page 12") in.
    This is a transform of that existing, already-generated explanation -
    NOT a second independent knowledge-generation step, so a PDF-grounded
    explanation stays grounded (see SCRIPT_RULES's no-new-facts rule) rather
    than the speech layer re-deriving its own answer from scratch.

    `visual_explanation` + `visual_type` are optional context about a visual
    shown alongside the speech, so the script describes the SAME thing the
    student is looking at, and describes it appropriately for its kind
    (an equation is decomposed differently than a graph or a diagram - see
    _VISUAL_TYPE_GUIDANCE) - never a different visual, and never a generic
    "here's a visual" description.

    Returns plain spoken text. Raises GeminiResponseError on failure -
    callers (app.py) are expected to catch this and fall back to text-only,
    never crash the lesson.
    """
    if visual_explanation:
        type_guidance = _VISUAL_TYPE_GUIDANCE.get(
            visual_type, "Briefly describe what the visual shows in your own words."
        )
        visual_block = (
            f'\nA visual is shown alongside this explanation: "{visual_explanation}". '
            f"{type_guidance} Do not describe anything other than this visual."
        )
    else:
        visual_block = ""

    prompt = f"""
You are turning a written teaching explanation into a natural spoken script,
as if a warm, patient human teacher were saying it out loud to one student.

Concept: {concept}
Student level: {student_level}

Written explanation to convert (preserve its factual meaning exactly - do not
add facts that aren't here, and do not drop any that are):
{explanation}
{visual_block}

{_language_instruction(language)}

{SCRIPT_RULES}

Example of the tone to aim for:
Written: "Encapsulation is an OOP concept where data and methods are bundled
together inside a class."
Spoken: "Think of encapsulation like a capsule. You keep the important data
and the methods that work with it together, inside one class. The outside
world doesn't need to know all the internal details."

Respond only with JSON matching the required schema.
"""
    result = call_gemini_json(prompt, SPEECH_SCRIPT_SCHEMA, temperature=0.7)
    script = (result.get("script") or "").strip()
    if not script:
        raise GeminiResponseError("Speech script came back empty.")
    return script
