"""
app.py
------
Streamlit UI for the AI Teacher.

Phase 1: teach any typed-in topic (Gemini's own knowledge).
Phase 2: also teach from an uploaded PDF, grounded with RAG (FAISS + Gemini
embeddings) so explanations/questions are based on the document instead of
the model's general knowledge.
Phase 3: remembers each student across sessions (SQLite) and personalizes
future lessons using that history - weak concepts get revised, strong ones
get skipped, unresolved misconceptions get explicitly addressed.
Phase 4: decides whether a visual (equation/graph/table/code/diagram/image)
would help teach the current concept, and shows it. The LLM only ever picks
WHAT visual and WHAT data (visuals/visual_selector.py); rendering is done by
fixed, hardcoded Python/Streamlit calls (visuals/visual_renderer.py) - the
model's output is never treated as executable code, HTML, or JS.
Phase 4.5: a Learning Library. Uploaded PDFs are persisted (database/
document_library.py) so a student can revisit them later without
re-uploading OR re-processing (the FAISS index and extracted chunks are
saved and reloaded, never rebuilt). "My Learning" now groups concept
mastery by topic, and both topics and documents get a [Revise] entry point
into Revision Mode - a lesson built from a DETERMINISTIC priority order
(weak + misconception concepts first, then developing, then a brief strong
review - see database.student_db.build_revision_plan) rather than asking
Gemini to reinvent a plan. Revision reuses the exact same teaching loop as
a normal lesson (explain_and_ask, generate_question, evaluate_answer,
adapt, visuals) - only the concept list and each concept's "summary" hint
are built differently, so nothing about the core loop had to change.

This file only handles UI + session_state. All the "teacher brain" logic
lives in teacher.py. All document/RAG logic lives in rag/. All persistent
student memory lives in database/student_db.py. All document persistence
lives in database/document_library.py. All visual decision/render logic
lives in visuals/. This file never touches SQLite, Gemini, or matplotlib
directly - only calls functions in those modules.

Data flow (Teach a Topic - unchanged from Phase 1, now personalized + visual):

    Student Input -> Student Profile -> [Student History] -> Lesson Planner
    -> Teaching Engine -> [Visual Decision Engine] -> Question Generator
    -> Student Answer -> Answer Evaluator -> Adaptive Teacher -> Next Concept
    -> ... -> Learning Report -> Update Student Memory

Data flow (Teach from PDF - unchanged from Phase 2, now personalized + visual
+ persisted):

    PDF Upload -> [Already processed? Reuse it] -> Extract/Chunk/Embed (if new)
    -> FAISS Index -> Save to Document Library -> [Student History]
    -> [same loop as above] -> Update Student Memory

Data flow (Revision Mode - new in Phase 4.5):

    [Revise] on a topic or document -> Load that topic/document's tracked
    concepts -> build_revision_plan() (deterministic priority order)
    -> [document only: reload existing VectorStore, no reprocessing]
    -> same teaching loop as above, seeded with priority + past
       misconceptions -> Update Student Memory (same tables, new rows/
       updated running averages - history is never overwritten)

Phase 5: an optional "Listen to explanation" voice layer. Explanation text
(never citations, never UI labels) is converted into a natural spoken
script (voice/script_generator.py), then into audio (voice/tts.py, Gemini's
native TTS, reusing the same API key as everything else), cached locally so
identical audio is never regenerated (voice/audio_manager.py). Voice is
lazy - nothing is generated until the student clicks Listen - and entirely
optional: any failure falls back to "keep reading the lesson" and never
blocks the teaching loop. This works unchanged in topic mode, PDF/RAG mode,
and Revision Mode, because all three already share the same
render_lesson_screen() - the Listen button was added there once.
"""

import streamlit as st
from dotenv import load_dotenv

import teacher
from rag.pdf_processor import extract_pages, chunk_pages, PDFProcessingError
from rag.vector_store import VectorStore
from database import student_db as db
from database import document_library as doclib
from visuals.visual_selector import select_visual
from visuals.visual_renderer import render_visual
from voice.script_generator import generate_speech_script
from voice.audio_manager import get_or_create_audio
from voice.tts import TTSUnavailableError

load_dotenv()

MAX_RETEACH_ATTEMPTS = 2  # avoid an infinite reteach loop on one concept
OVERVIEW_CHUNKS_FOR_PLANNING = 8  # how many chunks to sample when planning a document lesson
CONTEXT_CHUNKS_PER_STEP = 5       # how many chunks to retrieve per concept/reteach
DASHBOARD_RECENT_SESSIONS = 15
MAX_STRONG_IN_REVISION = 1        # per spec: strong concepts get brief review only, not a full re-teach
VOICE_OPTIONS = ["Kore", "Puck", "Zephyr", "Aoede", "Charon"]  # a small curated subset, not all 30

st.set_page_config(page_title="AI Teacher", page_icon="📘", layout="centered")


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "student_id": None,     # None until the identify screen is completed
        "student_name": "",
        "stage": "setup",       # setup -> lesson -> report | dashboard | materials
        "mode": "topic",        # "topic" or "document"
        "profile": None,
        "lesson_plan": None,
        "document_name": None,
        "document_id": None,    # persisted database/document_library.py id, when in document mode
        "vector_store": None,   # rag.vector_store.VectorStore, only set in document mode
        "student_context": None,  # this student's history summary, fetched once per lesson
        "revision_mode": False,
        "revision_label": "",
        "concept_index": 0,
        "explanation": None,
        "question": None,
        "visual": None,         # current concept's visual dict (or None), from visual_selector
        "current_context_chunks": [],  # chunks backing the current explanation/question
        "phase": "teaching",    # teaching -> feedback (within a concept)
        "last_evaluation": None,
        "retry_count": 0,
        "history": [],          # one finalized entry per concept, for the report
        "report": None,
        "audio_bytes": None,    # cached/generated audio for the CURRENT explanation, or None
        "audio_error": None,    # user-facing message if voice failed, or None
        "voice_name": "Kore",       # persists across lessons, like a settings preference
        "voice_pace": "normal",     # persists across lessons
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


LESSON_STATE_KEYS = [
    "stage", "mode", "profile", "lesson_plan", "document_name", "document_id", "vector_store",
    "student_context", "revision_mode", "revision_label", "concept_index", "explanation", "question", "visual",
    "current_context_chunks", "phase", "last_evaluation", "retry_count",
    "history", "report", "audio_bytes", "audio_error",
]


def reset_lesson():
    """Reset lesson state so the student can start a new topic or document. Keeps the logged-in student."""
    for key in LESSON_STATE_KEYS:
        if key in st.session_state:
            del st.session_state[key]
    init_state()
    st.session_state.stage = "setup"


def switch_student():
    """Fully log out: clear the student identity, any lesson in progress, AND
    any per-student preferences (voice settings) - these must never leak
    from one student's session into another's (Phase 5 spec section 15)."""
    for key in LESSON_STATE_KEYS + ["student_id", "student_name", "voice_name", "voice_pace"]:
        if key in st.session_state:
            del st.session_state[key]
    init_state()


init_state()


# ---------------------------------------------------------------------------
# Student identification (Phase 3) - gates everything else
# ---------------------------------------------------------------------------

def render_identify_screen():
    st.title("📘 AI Teacher")
    st.caption("Enter a student ID to continue. Using the same ID next time lets the AI remember your progress.")

    with st.form("identify_form"):
        student_id = st.text_input("Student ID", placeholder="e.g. student_001")
        name = st.text_input("Name (optional)", placeholder="e.g. Aisha")
        submitted = st.form_submit_button("Continue", use_container_width=True)

    if not submitted:
        return

    clean_id = "_".join(student_id.strip().split())  # collapse whitespace into underscores
    if not clean_id:
        st.error("Please enter a student ID before continuing.")
        return

    try:
        db.upsert_student_profile(clean_id, name=name.strip() or None)
    except db.StudentDBError as e:
        st.error(f"Couldn't set up your student profile right now: {e}")
        return

    st.session_state.student_id = clean_id
    st.session_state.student_name = name.strip()
    st.rerun()


def render_sidebar():
    with st.sidebar:
        label = st.session_state.student_name or st.session_state.student_id
        st.markdown(f"**Student:** {label}")
        st.caption(f"ID: `{st.session_state.student_id}`")
        st.divider()

        if st.button("🆕 New Lesson", use_container_width=True):
            reset_lesson()
            st.rerun()
        if st.button("📊 My Learning", use_container_width=True):
            st.session_state.stage = "dashboard"
            st.rerun()
        if st.button("📄 My Materials", use_container_width=True):
            st.session_state.stage = "materials"
            st.rerun()

        st.divider()
        if st.button("🔄 Switch Student", use_container_width=True):
            switch_student()
            st.rerun()

        st.divider()
        with st.expander("🔊 Voice settings"):
            st.session_state.voice_name = st.selectbox(
                "Voice", VOICE_OPTIONS,
                index=VOICE_OPTIONS.index(st.session_state.voice_name)
                if st.session_state.voice_name in VOICE_OPTIONS else 0,
            )
            st.session_state.voice_pace = st.selectbox(
                "Speed", ["slow", "normal", "fast"],
                index=["slow", "normal", "fast"].index(st.session_state.voice_pace),
            )


# ---------------------------------------------------------------------------
# Setup screen
# ---------------------------------------------------------------------------

def _radio_index(options, value, fallback=0):
    """Helper: default a radio widget to the student's last-used value, if any."""
    return options.index(value) if value in options else fallback


def render_setup_screen():
    st.title("📘 AI Teacher")
    st.caption("Tell me what you want to learn, and I'll teach it step by step.")

    existing_profile = db.get_student(st.session_state.student_id)

    mode_label = st.radio(
        "Learning Mode",
        ["Teach a Topic", "Teach from PDF"],
        horizontal=True,
        key="learning_mode_choice",
    )

    if mode_label == "Teach a Topic":
        _render_topic_setup_form(existing_profile)
    else:
        _render_document_setup_form(existing_profile)


def _start_lesson_with_history(profile: dict) -> dict:
    """
    Shared Phase 3 step for both modes: save the latest profile, then fetch
    this student's history summary so the lesson planner can personalize.
    Returns {} (no history) if this is a first-time student or the DB has
    a problem - the lesson still proceeds normally either way.
    """
    student_id = st.session_state.student_id
    try:
        db.upsert_student_profile(
            student_id, level=profile["level"], language=profile["language"], goal=profile["goal"],
        )
        return db.get_student_history_summary(student_id)
    except db.StudentDBError as e:
        st.warning(f"Note: couldn't load your learning history right now ({e}). Teaching without personalization.")
        return {}


def _render_topic_setup_form(existing_profile: dict):
    """Phase 1 setup form, now pre-filled with the student's last profile if available."""
    levels = ["Beginner", "Intermediate", "Advanced"]
    languages = ["English", "Hindi", "Hinglish"]

    with st.form("topic_setup_form"):
        topic = st.text_input("Topic", placeholder="e.g. Newton's Laws")

        level = st.radio("Learning Level", levels, horizontal=True,
                          index=_radio_index(levels, existing_profile.get("level")))
        language = st.radio("Language", languages, horizontal=True,
                             index=_radio_index(languages, existing_profile.get("language")))
        time_minutes = st.number_input("Available Time (minutes)", min_value=5, max_value=180, value=15, step=5)
        goal = st.text_input("Learning Goal", value=existing_profile.get("goal") or "",
                              placeholder="e.g. Understand the basics")

        submitted = st.form_submit_button("Start Lesson", use_container_width=True)

    if not submitted:
        return

    if not topic.strip():
        st.error("Please enter a topic before starting.")
        return
    if not goal.strip():
        st.error("Please enter a learning goal before starting.")
        return

    profile = teacher.build_student_profile(level, language, goal, time_minutes)
    student_context = _start_lesson_with_history(profile)

    with st.spinner("Planning your lesson..."):
        try:
            lesson_plan = teacher.create_lesson_plan(topic.strip(), profile, student_context=student_context)
        except teacher.GeminiNotConfiguredError as e:
            st.error(str(e))
            return
        except teacher.GeminiResponseError as e:
            st.error(f"Couldn't create a lesson plan right now: {e}")
            return

    st.session_state.mode = "topic"
    st.session_state.profile = profile
    st.session_state.lesson_plan = lesson_plan
    st.session_state.document_name = None
    st.session_state.document_id = None
    st.session_state.vector_store = None
    st.session_state.student_context = student_context
    st.session_state.revision_mode = False
    st.session_state.revision_label = ""
    st.session_state.concept_index = 0
    st.session_state.stage = "lesson"

    _load_concept(0)
    st.rerun()


def _render_document_setup_form(existing_profile: dict):
    """Phase 2 setup form: upload a PDF, then teach from it - now personalized, and persisted."""
    levels = ["Beginner", "Intermediate", "Advanced"]
    languages = ["English", "Hindi", "Hinglish"]

    with st.form("document_setup_form"):
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        st.caption("Already uploaded this before? It'll be reused instead of reprocessed. "
                    "You can also revisit past uploads anytime from **📄 My Materials**.")

        level = st.radio("Learning Level", levels, horizontal=True,
                          index=_radio_index(levels, existing_profile.get("level")))
        language = st.radio("Language", languages, horizontal=True,
                             index=_radio_index(languages, existing_profile.get("language")))
        time_minutes = st.number_input("Available Time (minutes)", min_value=5, max_value=180, value=15, step=5)
        goal = st.text_input("Learning Goal", value=existing_profile.get("goal") or "",
                              placeholder="e.g. Understand the main concepts")

        submitted = st.form_submit_button("Start Lesson", use_container_width=True)

    if not submitted:
        return

    if uploaded_file is None:
        st.error("Please upload a PDF before starting.")
        return
    if not goal.strip():
        st.error("Please enter a learning goal before starting.")
        return

    profile = teacher.build_student_profile(level, language, goal, time_minutes)
    document_name = uploaded_file.name
    student_id = st.session_state.student_id
    student_context = _start_lesson_with_history(profile)

    status = st.empty()
    try:
        existing_document = doclib.get_document_by_filename(student_id, document_name)
    except db.StudentDBError:
        existing_document = {}

    if existing_document:
        # Phase 4.5: already processed before - reuse it, never re-extract/re-embed.
        status.info(f"Found your previously processed \"{document_name}\" - reusing it (no reprocessing).")
        vector_store = doclib.load_vector_store(existing_document["document_id"], student_id)
        document_id = existing_document["document_id"]
        if vector_store is None:
            status.error("Couldn't load the previously saved version of this document. Try re-uploading it.")
            return
        status.success("Ready.")
    else:
        try:
            status.info("Processing document...")
            pages = extract_pages(uploaded_file.getvalue())
            chunks = chunk_pages(pages, document_name)

            with st.spinner("Reading and understanding your document..."):
                vector_store = VectorStore(chunks)

            status.success("Document processed successfully.")
        except PDFProcessingError as e:
            status.empty()
            st.error(str(e))
            return
        except teacher.GeminiNotConfiguredError as e:
            status.empty()
            st.error(str(e))
            return
        except teacher.GeminiResponseError as e:
            status.empty()
            st.error(f"Couldn't process this document right now: {e}")
            return

        try:
            document_id = doclib.save_document(
                student_id, document_name, page_count=len(pages),
                pdf_bytes=uploaded_file.getvalue(), chunks=chunks, vector_store=vector_store,
            )
        except db.StudentDBError as e:
            document_id = None
            st.warning(f"Note: couldn't save this document to your library right now ({e}). "
                        "Teaching from it this session anyway, but it won't be revisitable later.")

    with st.spinner("Planning your lesson from the document..."):
        try:
            overview_chunks = vector_store.retrieve_relevant_chunks(
                "main topics, key concepts, and overview of this document",
                top_k=OVERVIEW_CHUNKS_FOR_PLANNING,
            )
            lesson_plan = teacher.create_lesson_plan_from_document(
                document_name, profile, overview_chunks, student_context=student_context,
            )
        except teacher.GeminiResponseError as e:
            st.error(f"Couldn't create a lesson plan from this document right now: {e}")
            return

    st.session_state.mode = "document"
    st.session_state.profile = profile
    st.session_state.lesson_plan = lesson_plan
    st.session_state.document_name = document_name
    st.session_state.document_id = document_id
    st.session_state.vector_store = vector_store
    st.session_state.student_context = student_context
    st.session_state.revision_mode = False
    st.session_state.revision_label = ""
    st.session_state.concept_index = 0
    st.session_state.stage = "lesson"

    _load_concept(0)
    st.rerun()


# ---------------------------------------------------------------------------
# Revision Mode (Phase 4.5)
#
# Both entry points below build a lesson_plan dict the exact same shape
# create_lesson_plan()/create_lesson_plan_from_document() would, but the
# concept list is decided DETERMINISTICALLY from student_db.build_revision_plan
# (weak + misconception concepts first, then developing, then a brief
# strong review) instead of asking Gemini to invent a plan - per the spec's
# "use deterministic rules for revision priority" requirement. Once that
# plan exists, _load_concept() and the rest of the teaching loop below run
# completely unchanged - revision isn't a separate code path for teaching,
# only for how the concept list gets built.
# ---------------------------------------------------------------------------

def _priority_note(entry: dict) -> str:
    """
    Turned into each revision concept's "summary" field, which
    explain_and_ask() already reads to guide the explanation - this is how
    revision avoids just repeating the old lesson (spec section 7) without
    needing any changes to teacher.py at all.
    """
    base = f"Revision of a previously studied concept - current mastery is {entry['mastery_score']}% ({entry['status']})."
    if entry["misconception"]:
        return (base + f" The student previously showed this misconception: \"{entry['misconception']}\". "
                "Address it directly, explain the concept in a genuinely NEW way (different angle/analogy "
                "than a first-time explanation), and use a fresh example.")
    if entry["priority"] == "high":
        return base + " Give a concise recap, then explain it differently with a fresh example - don't just repeat a generic explanation."
    if entry["priority"] == "medium":
        return base + " Give a brief recap and one solid new example or practice angle to help it stick."
    return base + " This is just a brief confidence check - keep the recap short."


def _build_revision_lesson_plan(topic: str, entries: list, profile: dict) -> dict:
    concepts = [{"name": e["concept"], "summary": _priority_note(e)} for e in entries]
    return {
        "topic": topic,
        "objectives": [f"Revise weak and developing areas in {topic}"],
        "concepts": concepts,
        "estimated_time": f"{profile['time_minutes']} min",
    }


def _enter_revision_lesson(lesson_plan: dict, mode: str, document_name: str, document_id,
                            vector_store, revision_label: str):
    student_context = _start_lesson_with_history(st.session_state.profile)
    st.session_state.mode = mode
    st.session_state.lesson_plan = lesson_plan
    st.session_state.document_name = document_name
    st.session_state.document_id = document_id
    st.session_state.vector_store = vector_store
    st.session_state.student_context = student_context
    st.session_state.revision_mode = True
    st.session_state.revision_label = revision_label
    st.session_state.concept_index = 0
    st.session_state.history = []
    st.session_state.stage = "lesson"
    _load_concept(0)


def start_topic_revision(topic: str):
    """Revise a previously-studied topic (Phase 4.5 spec sections 5-7, 11)."""
    student_id = st.session_state.student_id
    try:
        concept_rows = db.get_concepts_for_topic(student_id, topic)
        misconceptions = db.get_unresolved_misconceptions(student_id)
    except db.StudentDBError as e:
        st.error(f"Couldn't load revision data for this topic right now: {e}")
        return

    if not concept_rows:
        st.warning(f"No revision data found for \"{topic}\" yet.")
        return

    entries = db.build_revision_plan(concept_rows, misconceptions, max_strong=MAX_STRONG_IN_REVISION)
    if not st.session_state.profile:
        # Revision can be started straight from the dashboard, before any lesson this session.
        existing = db.get_student(student_id)
        st.session_state.profile = teacher.build_student_profile(
            existing.get("level") or "Beginner", existing.get("language") or "English",
            existing.get("goal") or "Revise weak areas", 15,
        )

    lesson_plan = _build_revision_lesson_plan(topic, entries, st.session_state.profile)
    _enter_revision_lesson(lesson_plan, mode="topic", document_name=None, document_id=None,
                            vector_store=None, revision_label=f"Revision: {topic}")
    st.rerun()


def start_document_revision(document_id: int):
    """Revise a previously-uploaded document (Phase 4.5 spec section 8) - never reprocesses the PDF."""
    student_id = st.session_state.student_id
    try:
        document = doclib.get_document(document_id, student_id)
        if not document:
            st.error("Couldn't find this document in your library.")
            return
        vector_store = doclib.load_vector_store(document_id, student_id)
        concept_rows = db.get_concepts_for_document(student_id, document_id)
        misconceptions = db.get_unresolved_misconceptions(student_id)
    except db.StudentDBError as e:
        st.error(f"Couldn't load revision data for this document right now: {e}")
        return

    if vector_store is None:
        st.error("Couldn't reload this document's saved data. Try re-uploading it.")
        return

    if not st.session_state.profile:
        existing = db.get_student(student_id)
        st.session_state.profile = teacher.build_student_profile(
            existing.get("level") or "Beginner", existing.get("language") or "English",
            existing.get("goal") or "Revise weak areas", 15,
        )

    if concept_rows:
        entries = db.build_revision_plan(concept_rows, misconceptions, max_strong=MAX_STRONG_IN_REVISION)
        lesson_plan = _build_revision_lesson_plan(document["filename"], entries, st.session_state.profile)
    else:
        # No concept history for this document yet (e.g. never fully completed) - fall back to a
        # fresh plan from the (already-loaded, NOT re-embedded) document content.
        with st.spinner("Planning your revision..."):
            overview_chunks = vector_store.retrieve_relevant_chunks(
                "main topics, key concepts, and overview of this document", top_k=OVERVIEW_CHUNKS_FOR_PLANNING,
            )
            try:
                lesson_plan = teacher.create_lesson_plan_from_document(
                    document["filename"], st.session_state.profile, overview_chunks,
                    student_context=_start_lesson_with_history(st.session_state.profile),
                )
            except teacher.GeminiResponseError as e:
                st.error(f"Couldn't plan a revision lesson right now: {e}")
                return

    _enter_revision_lesson(lesson_plan, mode="document", document_name=document["filename"],
                            document_id=document_id, vector_store=vector_store,
                            revision_label=f"Revision: {document['filename']}")
    st.rerun()


# ---------------------------------------------------------------------------
# Lesson loop
# ---------------------------------------------------------------------------

def _retrieve_for_concept(concept: dict, extra_query: str = "") -> list:
    """
    Document mode only: retrieve the chunks most relevant to the current
    concept (optionally biased toward a misconception, for reteaching).
    Retrieval happens fresh for every concept/reteach - never just once -
    so different parts of the document surface as the lesson progresses.
    """
    vector_store = st.session_state.vector_store
    if vector_store is None:
        return []
    query = f"{concept['name']}: {concept['summary']}"
    if extra_query:
        query += f" {extra_query}"
    return vector_store.retrieve_relevant_chunks(query, top_k=CONTEXT_CHUNKS_PER_STEP)


def _find_known_misconception(concept_name: str) -> str:
    """
    Phase 3: if this student has an unresolved misconception (from a past
    session) matching the concept we're about to teach, return its text so
    the first explanation of this concept can address it directly - instead
    of waiting for the student to make the same mistake again.
    """
    context = st.session_state.student_context
    if not context or not context.get("unresolved_misconceptions"):
        return None
    concept_lower = concept_name.lower()
    for m in context["unresolved_misconceptions"]:
        stored_concept = m["concept"].lower()
        if stored_concept in concept_lower or concept_lower in stored_concept:
            return m["misconception"]
    return None


def _select_visual_safely(concept: dict, explanation_text: str, context_chunks: list):
    """
    Phase 4: decide whether a visual would help teach this concept. Always
    safe to call - select_visual() itself never raises, but this wraps it
    one more time defensively so an unexpected bug in the visuals package
    can never break the teaching loop (per the Phase 4 spec's hard
    requirement that a broken visual must never crash the lesson).
    """
    try:
        return select_visual(
            concept, explanation_text, st.session_state.profile,
            context_chunks=context_chunks, subject_hint=st.session_state.lesson_plan.get("topic"),
        )
    except Exception:
        return None


def _load_concept(index: int):
    """Generate the explanation + first question for a fresh concept."""
    concept = st.session_state.lesson_plan["concepts"][index]
    profile = st.session_state.profile
    is_document_mode = st.session_state.mode == "document"
    known_misconception = _find_known_misconception(concept["name"])

    try:
        context_chunks = _retrieve_for_concept(concept) if is_document_mode else None
        result = teacher.explain_and_ask(
            concept, profile, misconception=known_misconception, context_chunks=context_chunks,
        )
    except teacher.GeminiResponseError as e:
        st.session_state.explanation = None
        st.session_state.question = None
        st.session_state["_load_error"] = str(e)
        return

    explanation = {
        "explanation": result["explanation"],
        "example": result["example"],
        "source_pages": result.get("source_pages", []),
    }
    question = {
        "question": result["question"],
        "question_type": result["question_type"],
        "options": result.get("options", []),
    }

    st.session_state.explanation = explanation
    st.session_state.question = question
    st.session_state.visual = _select_visual_safely(concept, explanation["explanation"], context_chunks)
    st.session_state.current_context_chunks = context_chunks or []
    st.session_state.phase = "teaching"
    st.session_state.last_evaluation = None
    st.session_state.retry_count = 0
    st.session_state.audio_bytes = None
    st.session_state.audio_error = None
    st.session_state.pop("_load_error", None)


def render_lesson_screen():
    lesson_plan = st.session_state.lesson_plan
    concepts = lesson_plan["concepts"]
    index = st.session_state.concept_index
    concept = concepts[index]

    st.title(f"📘 {lesson_plan['topic']}")
    if st.session_state.mode == "document" and st.session_state.document_name:
        st.caption(f"📄 From: {st.session_state.document_name}")
    if st.session_state.get("revision_mode"):
        st.info(f"🔁 {st.session_state.get('revision_label') or 'Revision Mode'}")
    st.progress((index) / len(concepts), text=f"Concept {index + 1} of {len(concepts)}: {concept['name']}")

    if st.session_state.get("_load_error"):
        st.error(f"Something went wrong generating this concept: {st.session_state['_load_error']}")
        if st.button("Retry"):
            _load_concept(index)
            st.rerun()
        return

    explanation = st.session_state.explanation
    question = st.session_state.question
    if explanation is None or question is None:
        st.info("Loading your lesson...")
        return

    st.subheader(concept["name"])
    st.write(explanation["explanation"])
    st.markdown(f"**Example:** {explanation['example']}")
    _render_source_pages()

    if st.session_state.get("visual"):
        render_visual(st.session_state.visual)

    _render_listen_button(concept["name"], explanation["explanation"], st.session_state.profile)

    st.divider()
    st.markdown(f"**Question:** {question['question']}")

    if st.session_state.phase == "teaching":
        _render_answer_input(question)
    else:
        _render_feedback(concept, index, len(concepts))


def _render_source_pages():
    """Document mode only: show which pages the current explanation drew on."""
    if st.session_state.mode != "document":
        return
    pages = sorted({c["page"] for c in st.session_state.current_context_chunks})
    if pages:
        page_list = ", ".join(str(p) for p in pages)
        st.caption(f"Source: Page {page_list}")


def _render_listen_button(concept_name: str, explanation_text: str, profile: dict):
    """
    Phase 5: an optional "Listen to explanation" button. Nothing is
    generated until the student actually clicks it (per spec section 7 -
    speech is only generated for real teaching content, on demand, never
    automatically for every piece of UI text).

    `explanation_text` is the plain explanation only - never the "Example:"
    markdown, never source-page captions, never any UI label - so citations
    and navigation text can never end up in the spoken script (spec
    section 12). The visual's one-line `visual_explanation` (not its raw
    data) is passed along so the script can describe the same visual the
    student is looking at, never a different one (spec section 3/8).

    Any failure anywhere in the pipeline - script generation or TTS -
    is caught here and turned into a plain, honest message. The lesson
    (question, answer, evaluation) is completely unaffected either way.
    """
    listen_key = f"listen_{st.session_state.concept_index}_{st.session_state.retry_count}"

    if st.button("🔊 Listen to explanation", key=listen_key):
        with st.spinner("Preparing audio..."):
            try:
                visual = st.session_state.get("visual")
                visual_explanation = visual.get("visual_explanation") if visual else None
                visual_type = visual.get("visual_type") if visual else None
                script = generate_speech_script(
                    explanation_text, concept_name, profile.get("level", "Beginner"),
                    language=profile.get("language", "English"),
                    visual_explanation=visual_explanation,
                    visual_type=visual_type,
                )
                st.session_state.audio_bytes = get_or_create_audio(
                    script, language=profile.get("language", "English"),
                    voice=st.session_state.voice_name, pace=st.session_state.voice_pace,
                )
                st.session_state.audio_error = None
            except (teacher.GeminiResponseError, TTSUnavailableError):
                st.session_state.audio_bytes = None
                st.session_state.audio_error = "Voice is temporarily unavailable. You can continue reading the lesson."
            except Exception:
                # Belt-and-suspenders: nothing in the voice pipeline may ever crash the lesson.
                st.session_state.audio_bytes = None
                st.session_state.audio_error = "Voice is temporarily unavailable. You can continue reading the lesson."

    if st.session_state.audio_bytes:
        st.audio(st.session_state.audio_bytes, format="audio/wav")
    elif st.session_state.audio_error:
        st.caption(st.session_state.audio_error)


def _save_attempt_to_memory(concept_name: str, evaluation: dict):
    """
    Phase 3: record this single answer attempt into persistent student
    memory - every submission, not just the ones the student continues
    past, since "attempts" and "correct_attempts" should reflect reality.
    Never blocks the lesson if the database has a problem.

    Phase 4.5: also tags the concept with the current topic and (in
    document mode) document_id, so it can be found again for topic- or
    document-scoped revision later.
    """
    if not st.session_state.student_id:
        return
    try:
        db.update_concept_mastery(
            st.session_state.student_id, concept_name, evaluation["score"], evaluation["correct"],
            topic=st.session_state.lesson_plan.get("topic"),
            document_id=st.session_state.get("document_id"),
        )
        if evaluation.get("misconception"):
            db.save_misconception(st.session_state.student_id, concept_name, evaluation["misconception"])
        if evaluation["understanding"] == "strong":
            db.resolve_misconceptions_for_concept(st.session_state.student_id, concept_name)
    except db.StudentDBError as e:
        st.warning(f"Note: couldn't save this to your learning history right now ({e}).")


def _render_answer_input(question):
    answer_key = f"answer_input_{st.session_state.concept_index}_{st.session_state.retry_count}"

    if question["question_type"] == "multiple_choice" and question.get("options"):
        answer = st.radio("Your answer:", question["options"], key=answer_key, index=None)
    else:
        answer = st.text_area("Your answer:", key=answer_key)

    if st.button("Submit Answer", use_container_width=True):
        if not answer or not str(answer).strip():
            st.error("Please enter an answer before submitting.")
            return

        concept = st.session_state.lesson_plan["concepts"][st.session_state.concept_index]

        with st.spinner("Checking your answer..."):
            try:
                evaluation = teacher.evaluate_answer(
                    concept, question, str(answer).strip(), st.session_state.profile,
                )
            except teacher.GeminiResponseError as e:
                st.error(f"Couldn't evaluate your answer right now: {e}")
                return

        _save_attempt_to_memory(concept["name"], evaluation)

        st.session_state.last_evaluation = evaluation
        st.session_state.last_answer = str(answer).strip()
        st.session_state.phase = "feedback"
        st.rerun()


def _render_feedback(concept, index, total_concepts):
    evaluation = st.session_state.last_evaluation

    if evaluation["understanding"] == "strong":
        st.success(evaluation["feedback"])
    elif evaluation["understanding"] == "partial":
        st.warning(evaluation["feedback"])
    else:
        st.error(evaluation["feedback"])

    st.caption(f"Score: {evaluation['score']}/100 · Understanding: {evaluation['understanding']}")

    struggling = evaluation["recommended_action"] == "reteach"
    can_retry_more = st.session_state.retry_count < MAX_RETEACH_ATTEMPTS

    if struggling and evaluation.get("misconception"):
        st.info(f"Likely misconception: {evaluation['misconception']}")

    if struggling and can_retry_more:
        if st.button("Explain differently & try again", use_container_width=True):
            with st.spinner("Re-explaining..."):
                try:
                    is_document_mode = st.session_state.mode == "document"
                    context_chunks = (
                        _retrieve_for_concept(concept, extra_query=evaluation.get("misconception", ""))
                        if is_document_mode else None
                    )
                    result = teacher.explain_and_ask(
                        concept, st.session_state.profile, misconception=evaluation.get("misconception", ""),
                        context_chunks=context_chunks,
                    )
                except teacher.GeminiResponseError as e:
                    st.error(f"Couldn't re-explain right now: {e}")
                    return
            st.session_state.explanation = {
                "explanation": result["explanation"],
                "example": result["example"],
                "source_pages": result.get("source_pages", []),
            }
            st.session_state.question = {
                "question": result["question"],
                "question_type": result["question_type"],
                "options": result.get("options", []),
            }
            st.session_state.visual = _select_visual_safely(concept, result["explanation"], context_chunks)
            st.session_state.current_context_chunks = context_chunks or []
            st.session_state.retry_count += 1
            st.session_state.phase = "teaching"
            st.session_state.audio_bytes = None
            st.session_state.audio_error = None
            st.rerun()
    else:
        button_label = "Continue to Next Concept" if index + 1 < total_concepts else "Finish Lesson & See Report"
        if st.button(button_label, use_container_width=True):
            st.session_state.history.append({
                "concept": concept["name"],
                "question": st.session_state.question["question"],
                "answer": st.session_state.get("last_answer", ""),
                "score": evaluation["score"],
                "understanding": evaluation["understanding"],
                "misconception": evaluation.get("misconception", ""),
            })

            next_index = index + 1
            if next_index >= total_concepts:
                _generate_final_report()
            else:
                st.session_state.concept_index = next_index
                _load_concept(next_index)
            st.rerun()


# ---------------------------------------------------------------------------
# Report screen
# ---------------------------------------------------------------------------

def _save_session_to_memory(overall_score):
    """Phase 3: record the completed lesson as one row in the sessions table."""
    if not st.session_state.student_id:
        return
    try:
        db.save_session(
            student_id=st.session_state.student_id,
            mode=st.session_state.mode,
            topic=st.session_state.lesson_plan["topic"],
            document_name=st.session_state.document_name,
            available_time=st.session_state.profile["time_minutes"],
            final_score=overall_score,
        )
    except db.StudentDBError as e:
        st.warning(f"Note: couldn't save this session to your learning history right now ({e}).")


def _generate_final_report():
    with st.spinner("Putting together your learning report..."):
        try:
            report = teacher.generate_report(
                st.session_state.lesson_plan, st.session_state.profile, st.session_state.history
            )
            st.session_state.report = report
            overall_score = report.get("overall_score")
        except teacher.GeminiResponseError as e:
            st.session_state.report = {"error": str(e)}
            scores = [h["score"] for h in st.session_state.history]
            overall_score = round(sum(scores) / len(scores)) if scores else None

    _save_session_to_memory(overall_score)
    st.session_state.stage = "report"


def render_report_screen():
    st.title("📊 Your Learning Report")

    report = st.session_state.report
    if not report or report.get("error"):
        st.error(f"Couldn't generate the full report: {report.get('error') if report else 'unknown error'}")
        st.caption(
            "This is usually a temporary issue on Gemini's side (high demand), not something "
            "wrong with your session. Your answers - and your progress in the database - are saved either way."
        )
        if st.button("Retry Generating Report", use_container_width=True):
            _generate_final_report()
            st.rerun()
        st.write("Here's your raw session history in the meantime:")
        st.json(st.session_state.history)
    else:
        st.write(report["summary_message"])
        st.metric("Overall Score", f"{report['overall_score']}/100")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Concepts Understood**")
            for c in report["concepts_understood"] or ["None yet"]:
                st.write(f"- {c}")
        with col2:
            st.markdown("**⚠️ Weak Areas**")
            for c in report["weak_areas"] or ["None"]:
                st.write(f"- {c}")

        st.markdown("**Recommended Revision**")
        st.write(report["recommended_revision"])

        st.markdown("**Suggested Next Topic**")
        st.write(report["suggested_next_topic"])

    st.divider()
    if st.button("Start a New Lesson", use_container_width=True):
        reset_lesson()
        st.rerun()


# ---------------------------------------------------------------------------
# Dashboard screen (Phase 3, upgraded in Phase 4.5)
# ---------------------------------------------------------------------------

def render_dashboard_screen():
    st.title("📊 My Learning")

    student_id = st.session_state.student_id
    try:
        summary = db.get_student_history_summary(student_id, recent_limit=DASHBOARD_RECENT_SESSIONS)
        all_concepts = db.get_concept_progress(student_id)
        topics = db.get_topics_summary(student_id)
    except db.StudentDBError as e:
        st.error(f"Couldn't load your learning history right now: {e}")
        return

    if summary["is_new_student"]:
        st.info("No learning history yet — finish a lesson and your progress will show up here.")
        return

    st.subheader("Overall Progress")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Average score", f"{summary['average_score']}%" if summary["average_score"] is not None else "-")
    col2.metric("Concepts studied", len(all_concepts))
    col3.metric("Strong", len(summary["strong_concepts"]))
    col4.metric("Developing", len(summary["developing_concepts"]))
    col5.metric("Weak", len(summary["weak_concepts"]))

    st.subheader("Topics")
    if topics:
        for t in topics:
            status_emoji = {"weak": "🔴", "developing": "🟡", "strong": "🟢"}.get(t["status"], "")
            with st.expander(f"{status_emoji} {t['topic']} — {t['mastery']}% ({t['status'].capitalize()})"):
                st.caption(
                    f"Last studied: {t['last_studied'][:10]} · {t['concept_count']} concept(s) tracked · "
                    f"{t['attempts']} attempt(s) total"
                )
                topic_concepts = db.get_concepts_for_topic(student_id, t["topic"])
                st.dataframe(
                    [
                        {"Concept": c["concept"], "Mastery": f"{c['mastery_score']}%",
                         "Status": c["status"].capitalize(), "Attempts": c["attempts"]}
                        for c in topic_concepts
                    ],
                    use_container_width=True, hide_index=True,
                )
                if st.button("Start Revision", key=f"revise_topic_{t['topic']}", use_container_width=True):
                    start_topic_revision(t["topic"])
    else:
        st.caption("No topics tracked yet.")

    st.subheader("Learning History")
    if summary["recent_sessions"]:
        for s in summary["recent_sessions"]:
            label = s["document_name"] or s["topic"]
            score_text = f"{s['final_score']}%" if s["final_score"] is not None else "—"
            date_text = s["created_at"][:10]
            row_col1, row_col2 = st.columns([4, 1])
            with row_col1:
                st.write(f"**{label}** — {score_text} · {date_text}")
            with row_col2:
                if st.button("Revise", key=f"revise_session_{s['session_id']}", use_container_width=True):
                    if s["mode"] == "document":
                        existing_doc = doclib.get_document_by_filename(student_id, s["document_name"])
                        if existing_doc:
                            start_document_revision(existing_doc["document_id"])
                        else:
                            st.warning("The original document for this session is no longer available.")
                    else:
                        start_topic_revision(s["topic"])
    else:
        st.caption("No completed sessions yet.")

    if summary["unresolved_misconceptions"]:
        st.subheader("Open Misconceptions")
        for m in summary["unresolved_misconceptions"]:
            st.warning(f"**{m['concept']}**: {m['misconception']}")


# ---------------------------------------------------------------------------
# My Materials screen (Phase 4.5 - Document Library)
# ---------------------------------------------------------------------------

def render_materials_screen():
    st.title("📄 My Materials")
    st.caption("Documents you've uploaded. Revisiting one never re-processes the file.")

    student_id = st.session_state.student_id
    try:
        documents = doclib.get_documents_for_student(student_id)
    except db.StudentDBError as e:
        st.error(f"Couldn't load your document library right now: {e}")
        return

    if not documents:
        st.info("No documents uploaded yet. Upload a PDF from **🆕 New Lesson → Teach from PDF**.")
        return

    for doc in documents:
        with st.expander(f"📄 {doc['filename']}"):
            st.caption(
                f"Uploaded: {doc['upload_date'][:10]} · {doc['page_count'] or '?'} page(s) · "
                f"{doc['chunk_count'] or 0} chunk(s) · status: {doc['processed_status']}"
            )

            try:
                doc_concepts = db.get_concepts_for_document(student_id, doc["document_id"])
            except db.StudentDBError:
                doc_concepts = []

            if doc_concepts:
                st.markdown("**Concepts studied from this document:**")
                st.dataframe(
                    [
                        {"Concept": c["concept"], "Mastery": f"{c['mastery_score']}%",
                         "Status": c["status"].capitalize()}
                        for c in doc_concepts
                    ],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("No concepts studied from this document yet.")

            col1, col2 = st.columns(2)
            with col1:
                try:
                    pdf_bytes = doclib.get_pdf_bytes(doc["document_id"], student_id)
                except db.StudentDBError:
                    pdf_bytes = None
                if pdf_bytes:
                    st.download_button(
                        "⬇️ Open / Download", data=pdf_bytes, file_name=doc["filename"],
                        mime="application/pdf", use_container_width=True,
                        key=f"download_{doc['document_id']}",
                    )
                else:
                    st.caption("Original file unavailable.")
            with col2:
                if st.button("🔁 Revise", key=f"revise_doc_{doc['document_id']}", use_container_width=True):
                    start_document_revision(doc["document_id"])


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if st.session_state.student_id is None:
    render_identify_screen()
else:
    render_sidebar()
    if st.session_state.stage == "setup":
        render_setup_screen()
    elif st.session_state.stage == "lesson":
        render_lesson_screen()
    elif st.session_state.stage == "report":
        render_report_screen()
    elif st.session_state.stage == "dashboard":
        render_dashboard_screen()
    elif st.session_state.stage == "materials":
        render_materials_screen()
