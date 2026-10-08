"""
voice/audio_manager.py
------------------------
Local audio caching so the same explanation is never sent to the TTS API
twice. This is the only module app.py needs to call for voice - it owns
the cache-or-generate decision, so callers never have to think about
hashing or file paths.
"""

import hashlib
import os

from voice.tts import generate_speech, TTSUnavailableError, SUPPORTED_LANGUAGES, DEFAULT_VOICE

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio_cache")


def _cache_key(text: str, language: str, voice: str, pace: str) -> str:
    """
    Deterministic key from every input that changes the actual audio
    output: the words, the language, the voice, AND the pace - if any one
    of these changes, this must be a different key so a fresh generation
    happens rather than silently reusing audio spoken in the wrong style.
    """
    raw = f"{text}|{language}|{voice}|{pace}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.wav")


def _read_cached(path: str):
    """Returns the cached bytes, or None if the file is missing/empty/unreadable (self-heals by regenerating)."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        return data if data else None
    except OSError:
        return None


def get_or_create_audio(text: str, language: str = "English", voice: str = None,
                         pace: str = "normal") -> bytes:
    """
    Return WAV audio bytes for `text`, generating and caching it only if
    this exact (text, language, voice, pace) combination hasn't been
    spoken before.

    Falls back to a supported language/voice if given something
    unrecognized, rather than failing outright (per spec: "if the selected
    language or voice is unsupported, automatically fall back to a
    supported configuration").

    Raises TTSUnavailableError on failure (propagated from voice.tts) -
    callers must handle this gracefully, never let it crash the lesson.
    """
    if language not in SUPPORTED_LANGUAGES:
        language = "English"
    voice = voice if voice else DEFAULT_VOICE

    key = _cache_key(text, language, voice, pace)
    path = _cache_path(key)

    cached = _read_cached(path)
    if cached is not None:
        return cached

    try:
        audio_bytes = generate_speech(text, language=language, voice=voice, pace=pace)
    except TTSUnavailableError:
        raise
    except Exception as e:
        # Defense-in-depth: voice.tts.generate_speech already wraps its own
        # failures as TTSUnavailableError, but this guarantees that even an
        # unanticipated error here can never surface as a raw exception to
        # app.py - voice failures must always be this one catchable type.
        raise TTSUnavailableError(f"Speech generation failed: {e}")

    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as f:
            f.write(audio_bytes)
    except OSError:
        # Caching is a nice-to-have; failing to write to disk should never
        # stop the student from hearing the audio they already generated.
        pass

    return audio_bytes
