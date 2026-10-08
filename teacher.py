"""
teacher.py
----------
Core AI teaching engine for the AI Teacher hackathon project (Phase 1).

This module is intentionally kept separate from the Streamlit UI (app.py).
It owns all the "teacher brain" logic:

    Student Profile -> Lesson Planner -> Teaching Engine -> Question Generator
    -> Answer Evaluator -> Adaptive Teacher -> Report Generator

Every function that talks to Gemini asks for a structured JSON response
(using response_schema) so the rest of the app never has to parse free text.
"""

import json
import os
import time
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Gemini client setup
# ---------------------------------------------------------------------------

class GeminiNotConfiguredError(Exception):
    """Raised when the GEMINI_API_KEY is missing or invalid."""
    pass


class GeminiResponseError(Exception):
    """Raised when Gemini fails, or returns a response we can't use."""
    pass


# Model name is configurable via env var so it's easy to bump to a newer
# Gemini model later without touching code.
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

_client = None


def get_client():
    """Create (once) and return the Gemini client, or raise a clear error."""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiNotConfiguredError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    _client = genai.Client(api_key=api_key)
    return _client


def _is_transient_error(error: Exception) -> bool:
    """True for errors worth a quick retry: server overload, rate limits, timeouts."""
    text = str(error)
    return any(marker in text for marker in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "timeout"))


def _call_gemini_json(prompt: str, schema: dict, temperature: float = 0.6, max_retries: int = 2) -> dict:
    """
    Call Gemini and force a JSON response that matches `schema`.
    Raises GeminiResponseError on any failure so callers can handle it
    with a single try/except instead of guessing what went wrong.

    Transient errors (Gemini overloaded / rate-limited) are retried a couple
    of times with a short backoff before giving up, since these are usually
    resolved within seconds and shouldn't force the student to redo work.
    """
    client = get_client()

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature,
                ),
            )
            break
        except Exception as e:
            last_error = e
            if attempt < max_retries and _is_transient_error(e):
                time.sleep(1.5 * (attempt + 1))  # 1.5s, then 3s
                continue
            raise GeminiResponseError(f"Gemini API call failed: {e}")
    else:
        raise GeminiResponseError(f"Gemini API call failed: {last_error}")

    raw_text = getattr(response, "text", None)
    if not raw_text:
        raise GeminiResponseError("Gemini returned an empty response.")

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Model occasionally wraps JSON in markdown fences despite instructions.
        cleaned = raw_text.strip().strip("`")
        cleaned = cleaned.replace("json\n", "", 1) if cleaned.startswith("json\n") else cleaned
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise GeminiResponseError("Gemini returned malformed JSON we couldn't parse.")


def call_gemini_json(prompt: str, schema: dict, temperature: float = 0.6) -> dict:
    """
    Public entry point for other modules in this project (e.g. visuals/) to
    get a schema-constrained JSON response from Gemini, reusing the same
    client setup, retry-on-transient-error logic, and JSON parsing as the
    rest of this file - see _call_gemini_json for the details.
    """
    return _call_gemini_json(prompt, schema, temperature=temperature)


# ---------------------------------------------------------------------------
# 1. Student Profile
# ---------------------------------------------------------------------------

def build_student_profile(level: str, language: str, goal: str, time_minutes: int) -> dict:
    """Package the raw form inputs into a single student profile dict."""
    return {
        "level": level,
        "language": language,
        "goal": goal.strip(),
        "time_minutes": time_minutes,
    }


def _profile_summary(profile: dict) -> str:
    """Small helper to describe the student profile inside prompts."""
    return (
        f"Level: {profile['level']}\n"
        f"Preferred language/style: {profile['language']} "
        f"(if Hinglish, mix Hindi and English naturally like a real Indian tutor would)\n"
        f"Learning goal: {profile['goal']}\n"
        f"Time available: {profile['time_minutes']} minutes"
    )


# ---------------------------------------------------------------------------
# RAG grounding helpers (Phase 2 - shared by lesson planning, explaining,
# and question generation when teaching from an uploaded document)
# ---------------------------------------------------------------------------

def _format_context_chunks(context_chunks: list) -> str:
    """Render retrieved chunks as '[Page N] ...' blocks for use inside a prompt."""
    if not context_chunks:
        return ""
    return "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in context_chunks)


GROUNDING_INSTRUCTIONS = """
You must ground your answer in the RETRIEVED DOCUMENT CONTEXT below.
- Prefer facts, definitions, and examples that appear in the retrieved context.
- Do NOT invent facts that are not supported by the context.
- If the context does not contain enough information to fully cover this,
  say so honestly in your explanation rather than making something up.
- You may still explain things in your own words and add a simple example or
  analogy to aid understanding - just don't contradict the source material.
- In "source_pages", list only page numbers that actually appear as [Page N]
  markers below and that you actually drew on. Never invent a page number.
  If you didn't rely on specific pages, leave source_pages empty.
"""


# ---------------------------------------------------------------------------
# Student memory helpers (Phase 3 - shared by lesson planning)
#
# The classification below (weak/developing/strong, resolved/unresolved) is
# already done deterministically in database/student_db.py - Gemini is only
# ever shown the finished buckets, never asked to re-derive them. This is
# the "don't let the LLM be the database" rule from the Phase 3 spec.
# ---------------------------------------------------------------------------

def _format_student_context(history: dict) -> str:
    """
    Render a student's learning history (see database.student_db
    .get_student_history_summary) as a prompt block for the lesson planner.
    """
    if not history or history.get("is_new_student"):
        return "LEARNING HISTORY\nNo previous learning history. This is a first-time student."

    lines = ["LEARNING HISTORY"]

    if history.get("previous_topics"):
        lines.append("\nPreviously studied:")
        lines += [f"- {t}" for t in history["previous_topics"]]

    if history.get("weak_concepts"):
        lines.append("\nWEAK CONCEPTS (prioritize a short revision of these before new material):")
        lines += [f"- {c['concept']}: {c['mastery_score']}%" for c in history["weak_concepts"]]

    if history.get("developing_concepts"):
        lines.append("\nDEVELOPING CONCEPTS (include an extra example or practice on these):")
        lines += [f"- {c['concept']}: {c['mastery_score']}%" for c in history["developing_concepts"]]

    if history.get("strong_concepts"):
        lines.append("\nSTRONG CONCEPTS (avoid repeating these - move toward more advanced material):")
        lines += [f"- {c['concept']}: {c['mastery_score']}%" for c in history["strong_concepts"]]

    if history.get("unresolved_misconceptions"):
        lines.append("\nUNRESOLVED MISCONCEPTIONS (explicitly address these in the lesson):")
        lines += [f"- {m['concept']}: {m['misconception']}" for m in history["unresolved_misconceptions"]]

    if history.get("average_score") is not None:
        lines.append(f"\nRECENT PERFORMANCE\n- Average score across {history['total_sessions']} "
                      f"previous session(s): {history['average_score']}%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Lesson Planner
# ---------------------------------------------------------------------------

LESSON_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "objectives": {"type": "array", "items": {"type": "string"}},
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "summary": {"type": "string"},
                    "source_pages": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Page numbers this concept comes from. Empty for non-document lessons.",
                    },
                },
                "required": ["name", "summary"],
            },
        },
        "estimated_time": {"type": "string"},
    },
    "required": ["topic", "objectives", "concepts", "estimated_time"],
}


def create_lesson_plan(topic: str, profile: dict, student_context: dict = None) -> dict:
    """
    Ask Gemini to break the topic into a short, ordered sequence of concepts
    sized to fit the student's available time and level.

    If `student_context` is given (see database.student_db
    .get_student_history_summary), the plan is personalized: weak concepts
    get prioritized for revision, strong concepts are skipped over, and
    unresolved misconceptions are called out - using the deterministic
    classification already computed by the database layer.
    """
    history_block = _format_student_context(student_context)

    prompt = f"""
You are an expert, patient teacher designing a short lesson plan.

Topic: {topic}

Student profile:
{_profile_summary(profile)}

{history_block}

Design a lesson plan that:
- Breaks the topic into 3 to 5 small, teachable concepts, ordered from
  foundational to more advanced.
- Fits realistically within the student's available time.
- Matches the student's level (do not use jargon for beginners).
- Includes 2-3 short learning objectives for the whole lesson.
- For each concept, write a one-sentence summary of what it covers
  (this is just an internal outline, not the full explanation yet).
- If the learning history above lists weak concepts or unresolved
  misconceptions relevant to this topic, prioritize a short revision of
  those early in the plan rather than treating the student as a total
  beginner. If it lists strong concepts relevant to this topic, don't
  spend time re-teaching them from scratch - move toward deeper material.
  If there's no history, just teach the topic normally.

Respond only with JSON matching the required schema.
"""
    plan = _call_gemini_json(prompt, LESSON_PLAN_SCHEMA)
    if not plan.get("concepts"):
        raise GeminiResponseError("Lesson plan came back with no concepts.")
    return plan


def create_lesson_plan_from_document(document_name: str, profile: dict, overview_chunks: list,
                                      student_context: dict = None) -> dict:
    """
    Same idea as create_lesson_plan(), but the concepts are derived from
    retrieved excerpts of an uploaded document instead of Gemini's own
    knowledge of a topic name. `overview_chunks` should be a broad sample
    of chunks retrieved from the document (see rag/vector_store.py).

    `student_context` personalizes the plan the same way as in
    create_lesson_plan() - see that docstring.
    """
    context_text = _format_context_chunks(overview_chunks)
    history_block = _format_student_context(student_context)

    prompt = f"""
You are an expert, patient teacher designing a short lesson plan based on
material from an uploaded document.

Document: {document_name}

Retrieved excerpts from the document:
{context_text}

Student profile:
{_profile_summary(profile)}

{history_block}

Design a lesson plan that:
- Breaks the material shown in the excerpts above into 3 to 5 small,
  teachable concepts, ordered from foundational to more advanced.
- Is based ONLY on what these excerpts actually cover - do not invent
  topics or sections that aren't represented above.
- Fits realistically within the student's available time.
- Matches the student's level (do not use jargon for beginners).
- Includes 2-3 short learning objectives for the whole lesson.
- For each concept, write a one-sentence summary AND list the page numbers
  (from the [Page N] markers above) where that concept appears, in source_pages.
- Set "topic" to a short descriptive title for what this document teaches.
- If the learning history above lists weak concepts or unresolved
  misconceptions that this document also covers, prioritize a short
  revision of those early in the plan. If it lists relevant strong
  concepts, don't spend time re-teaching them from scratch.

If the excerpts don't contain enough material for a full lesson, do the best
you can with what's there rather than inventing content.

Respond only with JSON matching the required schema.
"""
    plan = _call_gemini_json(prompt, LESSON_PLAN_SCHEMA)
    if not plan.get("concepts"):
        raise GeminiResponseError("Lesson plan came back with no concepts.")
    return plan


# ---------------------------------------------------------------------------
# 3. Teaching Engine
# ---------------------------------------------------------------------------

EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "example": {"type": "string"},
        "source_pages": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Page numbers relied on, if teaching from a document. Empty otherwise.",
        },
    },
    "required": ["explanation", "example"],
}


def explain_concept(concept: dict, profile: dict, misconception: str = None, context_chunks: list = None) -> dict:
    """
    Explain a single concept at the right level, in the student's preferred
    language style. If `misconception` is provided, this is a RE-explanation
    after a wrong/weak answer, so approach it differently (not just repeat).

    If `context_chunks` is provided (retrieved from an uploaded document via
    rag/vector_store.py), the explanation is grounded in that material instead
    of Gemini's own general knowledge - see GROUNDING_INSTRUCTIONS.
    """
    if misconception:
        instruction = f"""
The student previously struggled with this concept. Their likely misconception was:
"{misconception}"

Re-explain the concept in a DIFFERENT way than a first-time explanation would
(different angle, different analogy). Directly address the misconception.
Keep it simpler and more concrete than before.
"""
    else:
        instruction = "This is the student's first time seeing this concept. Teach it clearly from scratch."

    if context_chunks:
        context_block = f"""
RETRIEVED DOCUMENT CONTEXT:
{_format_context_chunks(context_chunks)}

{GROUNDING_INSTRUCTIONS}
"""
    else:
        context_block = ""

    prompt = f"""
You are a warm, encouraging teacher.

Concept to teach: {concept['name']}
Concept outline: {concept['summary']}

Student profile:
{_profile_summary(profile)}

{instruction}
{context_block}

Rules:
- Match the language style ({profile['language']}) the student picked.
- Match the difficulty to their level ({profile['level']}); avoid unnecessary jargon.
- Keep the explanation focused and not too long (a few short paragraphs at most).
- Always include one concrete example or analogy that makes the idea click.

Respond only with JSON matching the required schema.
"""
    return _call_gemini_json(prompt, EXPLANATION_SCHEMA)


# ---------------------------------------------------------------------------
# 3b. Combined Explain + Question (performance optimization)
#
# explain_concept() and generate_question() are kept above as separate,
# independently-usable functions - but calling them back-to-back means two
# full Gemini round trips before the student sees anything, which is the
# main source of perceived slowness when a concept loads. This combines
# both into a single call/response for the hot path (loading or
# re-teaching a concept in app.py), while leaving the standalone functions
# available for anything that only needs one or the other.
# ---------------------------------------------------------------------------

TEACHING_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "example": {"type": "string"},
        "source_pages": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Page numbers relied on, if teaching from a document. Empty otherwise.",
        },
        "question": {"type": "string"},
        "question_type": {
            "type": "string",
            "enum": ["conceptual", "multiple_choice", "short_answer", "application"],
        },
        "options": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Only populated when question_type is multiple_choice.",
        },
    },
    "required": ["explanation", "example", "question", "question_type"],
}


def explain_and_ask(concept: dict, profile: dict, misconception: str = None, context_chunks: list = None) -> dict:
    """
    One Gemini call that both teaches the concept and writes a question
    checking it, instead of two sequential calls. Returns a dict with all
    the fields of explain_concept()'s result plus all of generate_question()'s
    result, so callers can split it into an "explanation" dict and a
    "question" dict however they need.

    Same parameters/behavior as explain_concept() + generate_question()
    called in sequence - this is purely a latency optimization, not a
    behavior change.
    """
    if misconception:
        instruction = f"""
The student previously struggled with this concept. Their likely misconception was:
"{misconception}"

Re-explain the concept in a DIFFERENT way than a first-time explanation would
(different angle, different analogy). Directly address the misconception.
Keep it simpler and more concrete than before.
"""
    else:
        instruction = "This is the student's first time seeing this concept. Teach it clearly from scratch."

    if context_chunks:
        context_block = f"""
RETRIEVED DOCUMENT CONTEXT:
{_format_context_chunks(context_chunks)}

{GROUNDING_INSTRUCTIONS}
(This also governs the question below: base it only on material shown here
or in your own explanation - do not test anything outside this material.)
"""
    else:
        context_block = ""

    prompt = f"""
You are a warm, encouraging teacher doing two things at once: teaching a
concept, then immediately checking whether the student understood it.

Concept to teach: {concept['name']}
Concept outline: {concept['summary']}

Student profile:
{_profile_summary(profile)}

{instruction}
{context_block}

Step 1 - Explain (fields: explanation, example, source_pages):
- Match the language style ({profile['language']}) the student picked.
- Match the difficulty to their level ({profile['level']}); avoid unnecessary jargon.
- Keep the explanation focused and not too long (a few short paragraphs at most).
- Always include one concrete example or analogy that makes the idea click.

Step 2 - Question (fields: question, question_type, options) - based on the
explanation you just wrote in Step 1:
- Write ONE question that checks real understanding (not just recall of a word).
- Pick whichever question type fits best: conceptual, multiple_choice,
  short_answer, or application. If multiple_choice, provide 3-4 options
  (do not reveal which one is correct in the text of the question).
- Write the question in the student's preferred language style ({profile['language']}).

Respond only with JSON matching the required schema.
"""
    result = _call_gemini_json(prompt, TEACHING_STEP_SCHEMA)
    if result.get("question_type") != "multiple_choice":
        result["options"] = []
    return result


# ---------------------------------------------------------------------------
# 4. Question Generator
# ---------------------------------------------------------------------------

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "question_type": {
            "type": "string",
            "enum": ["conceptual", "multiple_choice", "short_answer", "application"],
        },
        "options": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Only populated when question_type is multiple_choice.",
        },
    },
    "required": ["question", "question_type"],
}


def generate_question(concept: dict, explanation: dict, profile: dict, context_chunks: list = None) -> dict:
    """
    Generate one question that checks understanding of what was just taught.
    If `context_chunks` is given, the question is restricted to material that
    actually appears in those chunks, so document-mode questions never test
    something the document never covered.
    """
    if context_chunks:
        context_block = f"""
RETRIEVED DOCUMENT CONTEXT (base the question only on material shown here or
in the explanation below - do not test anything outside this material):
{_format_context_chunks(context_chunks)}
"""
    else:
        context_block = ""

    prompt = f"""
You are a teacher checking whether a student understood what you just taught.

Concept: {concept['name']}
What you just explained: {explanation['explanation']}
Example you used: {explanation['example']}
{context_block}

Student profile:
{_profile_summary(profile)}

Write ONE question that checks real understanding (not just recall of a word).
Pick whichever question type fits best: conceptual, multiple_choice,
short_answer, or application. If multiple_choice, provide 3-4 options
(do not reveal which one is correct in the text of the question).
Write the question in the student's preferred language style ({profile['language']}).

Respond only with JSON matching the required schema.
"""
    question = _call_gemini_json(prompt, QUESTION_SCHEMA)
    if question.get("question_type") != "multiple_choice":
        question["options"] = []
    return question


# ---------------------------------------------------------------------------
# 5. Answer Evaluator
# ---------------------------------------------------------------------------

EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "score": {"type": "integer"},
        "understanding": {"type": "string", "enum": ["strong", "partial", "weak"]},
        "misconception": {"type": "string"},
        "feedback": {"type": "string"},
        "recommended_action": {"type": "string", "enum": ["continue", "reteach"]},
    },
    "required": [
        "correct", "score", "understanding", "misconception",
        "feedback", "recommended_action",
    ],
}


def evaluate_answer(concept: dict, question: dict, student_answer: str, profile: dict) -> dict:
    """
    Judge the student's answer and decide whether to move on or reteach.
    `misconception` should be an empty string when understanding is strong.
    """
    prompt = f"""
You are grading a student's answer to check real understanding, not just
keyword matching. Be fair but honest - do not inflate scores.

Concept being tested: {concept['name']}
Question asked: {question['question']}
Question type: {question['question_type']}
{"Options given: " + ", ".join(question.get("options", [])) if question.get("options") else ""}

Student's answer: "{student_answer}"

Student profile:
{_profile_summary(profile)}

Evaluate the answer and return:
- correct: true/false
- score: 0-100 (partial credit allowed for partially correct reasoning)
- understanding: "strong", "partial", or "weak"
- misconception: the specific likely misconception behind the mistake
  (empty string "" if understanding is strong)
- feedback: 1-2 short encouraging sentences in the student's preferred
  language style ({profile['language']})
- recommended_action: "continue" if understanding is strong or partial-but-acceptable,
  "reteach" if understanding is weak and the concept needs to be re-explained

Respond only with JSON matching the required schema.
"""
    return _call_gemini_json(prompt, EVALUATION_SCHEMA)


# ---------------------------------------------------------------------------
# 6. Adaptive Teacher (orchestration helpers)
# ---------------------------------------------------------------------------

def adapt_after_weak_answer(concept: dict, profile: dict, misconception: str, context_chunks: list = None) -> dict:
    """
    Wraps explain_concept() specifically for the "student struggled" path,
    so the calling code in app.py reads clearly as a teaching decision.

    In document mode, app.py should re-retrieve context_chunks (e.g. querying
    for "<concept name> <misconception>") before calling this, so the
    re-explanation can draw on the most relevant part of the document again.
    """
    return explain_concept(concept, profile, misconception=misconception, context_chunks=context_chunks)


# ---------------------------------------------------------------------------
# 7. Learning Report
# ---------------------------------------------------------------------------

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "concepts_understood": {"type": "array", "items": {"type": "string"}},
        "weak_areas": {"type": "array", "items": {"type": "string"}},
        "overall_score": {"type": "integer"},
        "recommended_revision": {"type": "string"},
        "suggested_next_topic": {"type": "string"},
        "summary_message": {"type": "string"},
    },
    "required": [
        "concepts_understood", "weak_areas", "overall_score",
        "recommended_revision", "suggested_next_topic", "summary_message",
    ],
}


def generate_report(lesson_plan: dict, profile: dict, history: list) -> dict:
    """
    Build the end-of-lesson report from the full history of
    concept -> question -> answer -> evaluation entries.

    `history` is a list of dicts like:
        {
            "concept": "...",
            "question": "...",
            "answer": "...",
            "score": 0-100,
            "understanding": "strong/partial/weak",
            "misconception": "...",
        }
    """
    history_text = "\n".join(
        f"- Concept: {h['concept']} | Understanding: {h['understanding']} | "
        f"Score: {h['score']} | Misconception: {h['misconception'] or 'none'}"
        for h in history
    )

    prompt = f"""
You are summarizing a completed tutoring session into a simple learning report
for the student.

Topic: {lesson_plan['topic']}
Student profile:
{_profile_summary(profile)}

Session history:
{history_text}

Produce:
- concepts_understood: list of concept names the student understood well
- weak_areas: list of concept names that need more work
- overall_score: 0-100 average understanding across the session
- recommended_revision: 1-2 sentences on what to revise and how
- suggested_next_topic: one logical next topic to learn after this one
- summary_message: 2-3 encouraging sentences in the student's preferred
  language style ({profile['language']})

Respond only with JSON matching the required schema.
"""
    return _call_gemini_json(prompt, REPORT_SCHEMA)
