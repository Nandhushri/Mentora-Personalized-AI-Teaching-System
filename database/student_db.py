"""
database/student_db.py
-----------------------
Persistent student memory for the AI Teacher (Phase 3), using Python's
built-in sqlite3 - no ORM, no external database server.

This module is the single source of truth for a student's factual learning
state (scores, mastery, misconceptions) and their session-level learning
history. It never calls Gemini and never makes teaching decisions - it just
stores and retrieves facts, and (Phase 4.5) deterministically prioritizes
what to revise. teacher.py is responsible for turning those facts into a
lesson (see the "Important Architecture Rule" in the Phase 3 spec: the LLM
interprets state, the database remembers it).

Phase 4.5 adds: each concept_progress row now also remembers which topic
(and, if applicable, which persisted document_id - see
database/document_library.py) it was taught under, so "My Learning" can
group concepts by topic and Revision Mode can pull "the weak concepts for
THIS topic/document" instead of the student's entire history. This is a
foreign-key-style link, not a merge - document metadata and vector-store
data still live entirely in document_library.py's own tables (per the
Phase 4.5 spec's "keep STUDENT MEMORY and DOCUMENT LIBRARY separate" rule).

The database file and all tables are created automatically the first time
this module is used - nothing to set up by hand. New columns are added to
existing tables via a small guarded migration so upgrading from an earlier
phase never loses existing data.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "ai_teacher.db")

# Deterministic mastery thresholds (spec section 3). Change these two
# numbers to retune what counts as "weak" / "developing" / "strong".
WEAK_MAX = 49          # 0-49   -> weak
DEVELOPING_MAX = 79    # 50-79  -> developing
                        # 80-100 -> strong


class StudentDBError(Exception):
    """Raised when a database operation fails."""
    pass


def status_from_mastery(mastery_score: int) -> str:
    """Deterministic status label - not left up to the LLM to decide."""
    if mastery_score <= WEAK_MAX:
        return "weak"
    if mastery_score <= DEVELOPING_MAX:
        return "developing"
    return "strong"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connect():
    """
    Open a connection, make sure all tables exist, yield it, commit on
    success, and always close it. Every public function in this module
    goes through here, so the database is created automatically on first
    use - no separate init step required.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        _create_tables(conn)
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        raise StudentDBError(f"Database error: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, coltype: str):
    """Small guarded migration helper: adds a column only if it doesn't already exist."""
    existing_columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing_columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def _create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT,
            level TEXT,
            language TEXT,
            goal TEXT,
            teaching_style TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            mode TEXT NOT NULL,              -- 'topic' or 'document'
            topic TEXT NOT NULL,
            document_name TEXT,
            available_time INTEGER,
            final_score INTEGER,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS concept_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            concept TEXT NOT NULL,
            mastery_score INTEGER NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            correct_attempts INTEGER NOT NULL DEFAULT 0,
            last_studied TEXT NOT NULL,
            status TEXT NOT NULL,
            UNIQUE (student_id, concept)
        );

        CREATE TABLE IF NOT EXISTS misconceptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            concept TEXT NOT NULL,
            misconception TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'unresolved',
            first_detected TEXT NOT NULL,
            resolved_at TEXT
        );
    """)
    # Phase 4.5: link each concept to the topic (and, if applicable, the
    # persisted document_id from database/document_library.py) it was
    # taught under. Guarded so this is a no-op on a database that already
    # has these columns from a previous run.
    _add_column_if_missing(conn, "concept_progress", "topic", "TEXT")
    _add_column_if_missing(conn, "concept_progress", "document_id", "INTEGER")

    # Phase 5: voice settings are per-student, not a global app setting -
    # persisting them here (rather than only in session_state) is what
    # keeps one student's voice/speed choice from leaking into another
    # student's session after a student switch.
    _add_column_if_missing(conn, "students", "preferred_voice", "TEXT")
    _add_column_if_missing(conn, "students", "voice_pace", "TEXT")


# ---------------------------------------------------------------------------
# Student profile
# ---------------------------------------------------------------------------

def upsert_student_profile(student_id: str, level: str = None, language: str = None,
                            goal: str = None, name: str = None) -> dict:
    """
    Create the student if they don't exist yet, or update whichever fields
    are provided. Called whenever a lesson starts, so the profile always
    reflects the student's latest stated level/language/goal.
    """
    student_id = (student_id or "").strip()
    if not student_id:
        raise StudentDBError("student_id cannot be empty.")

    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO students (student_id, name, level, language, goal, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (student_id, name, level, language, goal, _now()),
        )
        # Only overwrite fields that were actually provided.
        updates, values = [], []
        for column, value in [("name", name), ("level", level), ("language", language), ("goal", goal)]:
            if value is not None:
                updates.append(f"{column} = ?")
                values.append(value)
        if updates:
            values.append(student_id)
            conn.execute(f"UPDATE students SET {', '.join(updates)} WHERE student_id = ?", values)

        row = conn.execute("SELECT * FROM students WHERE student_id = ?", (student_id,)).fetchone()
        return dict(row) if row else {}


def get_student(student_id: str) -> dict:
    """Returns the student's profile dict, or {} if they've never been seen."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM students WHERE student_id = ?", (student_id,)).fetchone()
        return dict(row) if row else {}


def save_voice_preferences(student_id: str, voice_name: str, voice_pace: str):
    """
    Persist this student's voice/speed choice so it's remembered next time
    they log in, and - just as importantly - so it never leaks into a
    different student's session (each student's preference lives on their
    own row, not in a shared global).
    """
    student_id = (student_id or "").strip()
    if not student_id:
        return
    with _connect() as conn:
        conn.execute(
            "UPDATE students SET preferred_voice = ?, voice_pace = ? WHERE student_id = ?",
            (voice_name, voice_pace, student_id),
        )


def get_voice_preferences(student_id: str) -> dict:
    """Returns {'preferred_voice': ..., 'voice_pace': ...}, with safe defaults for a new student."""
    student = get_student(student_id)
    return {
        "preferred_voice": student.get("preferred_voice") or "Kore",
        "voice_pace": student.get("voice_pace") or "normal",
    }


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def save_session(student_id: str, mode: str, topic: str, document_name: str,
                  available_time: int, final_score) -> int:
    """Record one completed lesson. Returns the new session_id."""
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO sessions (student_id, mode, topic, document_name, available_time, final_score, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (student_id, mode, topic, document_name, available_time, final_score, _now()),
        )
        return cursor.lastrowid


def get_recent_sessions(student_id: str, limit: int = 5) -> list:
    """Most recent sessions first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE student_id = ? ORDER BY created_at DESC, session_id DESC LIMIT ?",
            (student_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Concept-level progress
# ---------------------------------------------------------------------------

def update_concept_mastery(student_id: str, concept: str, score: int, correct: bool,
                            topic: str = None, document_id: int = None) -> dict:
    """
    Fold one new evaluation into this student's running mastery for a
    concept (a simple running average, not just "last score wins"), and
    recompute the deterministic status. Creates the row if this is the
    student's first attempt at the concept.

    `topic` and `document_id` (Phase 4.5) tag which topic/document this
    concept was most recently taught under, so it can be found again for
    topic- or document-scoped revision. They're updated on every call (not
    just insert), so if a concept later gets taught under a different
    topic, the tag stays current.
    """
    with _connect() as conn:
        existing = conn.execute(
            "SELECT * FROM concept_progress WHERE student_id = ? AND concept = ?",
            (student_id, concept),
        ).fetchone()

        if existing is None:
            new_mastery = max(0, min(100, int(round(score))))
            attempts = 1
            correct_attempts = 1 if correct else 0
        else:
            attempts = existing["attempts"] + 1
            correct_attempts = existing["correct_attempts"] + (1 if correct else 0)
            # Running average keeps history from being overwritten by a single attempt.
            new_mastery = int(round((existing["mastery_score"] * existing["attempts"] + score) / attempts))
            new_mastery = max(0, min(100, new_mastery))

        status = status_from_mastery(new_mastery)
        last_studied = _now()

        conn.execute(
            """
            INSERT INTO concept_progress
                (student_id, concept, mastery_score, attempts, correct_attempts, last_studied, status,
                 topic, document_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (student_id, concept) DO UPDATE SET
                mastery_score = excluded.mastery_score,
                attempts = excluded.attempts,
                correct_attempts = excluded.correct_attempts,
                last_studied = excluded.last_studied,
                status = excluded.status,
                topic = COALESCE(excluded.topic, concept_progress.topic),
                document_id = COALESCE(excluded.document_id, concept_progress.document_id)
            """,
            (student_id, concept, new_mastery, attempts, correct_attempts, last_studied, status,
             topic, document_id),
        )

        row = conn.execute(
            "SELECT * FROM concept_progress WHERE student_id = ? AND concept = ?",
            (student_id, concept),
        ).fetchone()
        return dict(row)


def get_concept_progress(student_id: str) -> list:
    """All tracked concepts for this student, most recently studied first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM concept_progress WHERE student_id = ? ORDER BY last_studied DESC",
            (student_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_concepts_for_topic(student_id: str, topic: str) -> list:
    """All tracked concepts previously taught to this student under a specific topic."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM concept_progress WHERE student_id = ? AND topic = ? ORDER BY mastery_score ASC",
            (student_id, topic),
        ).fetchall()
        return [dict(r) for r in rows]


def get_concepts_for_document(student_id: str, document_id: int) -> list:
    """All tracked concepts previously taught to this student from a specific document."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM concept_progress WHERE student_id = ? AND document_id = ? ORDER BY mastery_score ASC",
            (student_id, document_id),
        ).fetchall()
        return [dict(r) for r in rows]


def get_topics_summary(student_id: str) -> list:
    """
    One row per distinct topic this student has concept-level history for,
    with an aggregate mastery (average of its concepts' mastery scores),
    a deterministic status derived from that average, how many concepts
    are tracked under it, and when it was last studied. This is what
    powers the "Topics" list in My Learning (Phase 4.5 spec section 1).
    Topics taught purely from a document (topic still gets set) are
    included too - "My Materials" additionally lists them by filename.
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT topic,
                   ROUND(AVG(mastery_score)) AS avg_mastery,
                   SUM(attempts) AS total_attempts,
                   COUNT(*) AS concept_count,
                   MAX(last_studied) AS last_studied
            FROM concept_progress
            WHERE student_id = ? AND topic IS NOT NULL AND topic != ''
            GROUP BY topic
            ORDER BY last_studied DESC
            """,
            (student_id,),
        ).fetchall()

    summaries = []
    for r in rows:
        avg_mastery = int(r["avg_mastery"]) if r["avg_mastery"] is not None else 0
        summaries.append({
            "topic": r["topic"],
            "mastery": avg_mastery,
            "status": status_from_mastery(avg_mastery),
            "attempts": r["total_attempts"],
            "concept_count": r["concept_count"],
            "last_studied": r["last_studied"],
        })
    return summaries


def build_revision_plan(concepts: list, unresolved_misconceptions: list, max_strong: int = 1) -> list:
    """
    Deterministically decide which concepts to revise and in what order
    (Phase 4.5 spec section 6 - "use deterministic rules", not left to the
    LLM). Given this student's tracked concepts for one topic/document:

      1. Concepts with an unresolved misconception, or status "weak" -> highest priority.
      2. Concepts with status "developing" -> medium priority.
      3. Concepts with status "strong" -> at most `max_strong` included, for a brief review only.

    Returns an ordered list of {"concept", "mastery_score", "status",
    "priority", "misconception"} dicts - callers turn this into the
    lesson plan's concept list.
    """
    misconception_by_concept = {}
    for m in unresolved_misconceptions:
        misconception_by_concept.setdefault(m["concept"], m["misconception"])

    weak_or_flagged, developing, strong = [], [], []
    for c in concepts:
        entry = {
            "concept": c["concept"],
            "mastery_score": c["mastery_score"],
            "status": c["status"],
            "misconception": misconception_by_concept.get(c["concept"]),
        }
        if entry["misconception"] or c["status"] == "weak":
            entry["priority"] = "high"
            weak_or_flagged.append(entry)
        elif c["status"] == "developing":
            entry["priority"] = "medium"
            developing.append(entry)
        else:
            entry["priority"] = "brief"
            strong.append(entry)

    # Weakest-first within each bucket, so the most urgent gets taught first.
    weak_or_flagged.sort(key=lambda e: e["mastery_score"])
    developing.sort(key=lambda e: e["mastery_score"])

    return weak_or_flagged + developing + strong[:max_strong]


# ---------------------------------------------------------------------------
# Misconceptions
# ---------------------------------------------------------------------------

def save_misconception(student_id: str, concept: str, misconception_text: str):
    """
    Record a newly detected misconception, unless an identical unresolved
    one for this student+concept already exists (avoids duplicate spam
    across attempts on the same concept in one lesson).
    """
    misconception_text = (misconception_text or "").strip()
    if not misconception_text:
        return

    with _connect() as conn:
        duplicate = conn.execute(
            "SELECT id FROM misconceptions WHERE student_id = ? AND concept = ? "
            "AND misconception = ? AND status = 'unresolved'",
            (student_id, concept, misconception_text),
        ).fetchone()
        if duplicate:
            return

        conn.execute(
            "INSERT INTO misconceptions (student_id, concept, misconception, status, first_detected) "
            "VALUES (?, ?, ?, 'unresolved', ?)",
            (student_id, concept, misconception_text, _now()),
        )


def resolve_misconceptions_for_concept(student_id: str, concept: str) -> int:
    """
    Mark all unresolved misconceptions for this concept as resolved (called
    when the student demonstrates strong understanding). Returns how many
    were resolved.
    """
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE misconceptions SET status = 'resolved', resolved_at = ? "
            "WHERE student_id = ? AND concept = ? AND status = 'unresolved'",
            (_now(), student_id, concept),
        )
        return cursor.rowcount


def get_unresolved_misconceptions(student_id: str) -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM misconceptions WHERE student_id = ? AND status = 'unresolved' "
            "ORDER BY first_detected DESC",
            (student_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Aggregate history (this is what reaches the Lesson Planner and Dashboard)
# ---------------------------------------------------------------------------

def get_student_history_summary(student_id: str, recent_limit: int = 5) -> dict:
    """
    One call that gathers everything the Lesson Planner (and the dashboard)
    need: previously studied topics, concepts bucketed by deterministic
    status, unresolved misconceptions, and recent session scores.

    Only relevant, already-summarized history is returned here - never the
    whole database - so this is safe to drop straight into a Gemini prompt.
    """
    sessions = get_recent_sessions(student_id, limit=recent_limit)
    concepts = get_concept_progress(student_id)
    misconceptions = get_unresolved_misconceptions(student_id)

    is_new_student = not sessions and not concepts

    previous_topics = []
    for s in sessions:
        label = s["document_name"] or s["topic"]
        if label not in previous_topics:
            previous_topics.append(label)

    weak = [c for c in concepts if c["status"] == "weak"]
    developing = [c for c in concepts if c["status"] == "developing"]
    strong = [c for c in concepts if c["status"] == "strong"]

    all_sessions = get_recent_sessions(student_id, limit=1000)
    scored_sessions = [s["final_score"] for s in all_sessions if s["final_score"] is not None]
    average_score = round(sum(scored_sessions) / len(scored_sessions)) if scored_sessions else None

    return {
        "is_new_student": is_new_student,
        "previous_topics": previous_topics,
        "weak_concepts": weak,
        "developing_concepts": developing,
        "strong_concepts": strong,
        "unresolved_misconceptions": misconceptions,
        "recent_sessions": sessions,
        "average_score": average_score,
        "total_sessions": len(all_sessions),
    }
