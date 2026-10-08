"""
voice/tts.py
-------------
Converts plain speech text into audio bytes (WAV), using Gemini's native
TTS capability.

Reuses the existing Gemini client/API key setup from teacher.py (teacher
.get_client()) rather than introducing a second provider or a second set
of credentials - per the Phase 5 spec, no new API infrastructure is
created here. The only new environment variables are optional overrides
for the model/voice, not a new secret (see .env.example).
"""

import io
import os
import wave

from google.genai import types

from teacher import get_client

# Overridable via .env, same pattern as GEMINI_MODEL in teacher.py.
TTS_MODEL = os.environ.get("TTS_MODEL", "gemini-3.1-flash-tts-preview")
DEFAULT_VOICE = os.environ.get("TTS_VOICE", "Kore")

# The three language styles this app's setup form offers. Gemini's TTS
# models auto-detect spoken language from the text itself (see below), so
# this list exists only so audio_manager.py can validate/fall back to a
# known-good value rather than caching audio under an unrecognized key.
SUPPORTED_LANGUAGES = ["English", "Hindi", "Hinglish"]

# Gemini TTS returns raw PCM audio at a fixed format - documented as
# 24kHz, mono, 16-bit. We wrap it in a standard WAV header ourselves
# (stdlib `wave` module, no extra dependency) so it plays in any browser
# via Streamlit's native audio player.
_PCM_CHANNELS = 1
_PCM_RATE = 24000
_PCM_SAMPLE_WIDTH = 2


class TTSUnavailableError(Exception):
    """Raised when speech audio couldn't be generated for any reason."""
    pass


def _pcm_to_wav_bytes(pcm_data: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(_PCM_CHANNELS)
        wf.setsampwidth(_PCM_SAMPLE_WIDTH)
        wf.setframerate(_PCM_RATE)
        wf.writeframes(pcm_data)
    return buffer.getvalue()


def generate_speech(text: str, language: str = "English", voice: str = None,
                     pace: str = "normal") -> bytes:
    """
    Synthesize `text` as speech and return WAV audio bytes.

    `language` is informational only - Gemini's TTS models auto-detect the
    spoken language from the text itself, so no per-language model/voice
    switching is needed (this keeps the language system extensible: adding
    a new language doesn't require any code change here).

    Raises TTSUnavailableError on any failure (bad/missing API key, model
    unavailable, malformed response, etc.) - callers must treat this as
    "voice isn't available right now" and fall back to text, never crash.
    """
    text = (text or "").strip()
    if not text:
        raise TTSUnavailableError("No text was provided to speak.")

    pace_instruction = {
        "slow": "Speak clearly and a little slower than normal, for careful listening.",
        "fast": "Speak at a slightly brisker, energetic pace.",
    }.get(pace, "Speak at a normal, natural pace.")

    prompt = f"Say warmly and clearly, like a friendly, encouraging teacher. {pace_instruction}\n\n{text}"

    try:
        client = get_client()
        response = client.models.generate_content(
            model=TTS_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice or DEFAULT_VOICE,
                        )
                    )
                ),
            ),
        )
        data = response.candidates[0].content.parts[0].inline_data.data
        if isinstance(data, str):
            # Defensive: some SDK paths may hand back base64 text instead of
            # already-decoded bytes - handle both without assuming either.
            import base64
            data = base64.b64decode(data)
        if not data:
            raise TTSUnavailableError("TTS response contained no audio data.")
        return _pcm_to_wav_bytes(data)
    except TTSUnavailableError:
        raise
    except Exception as e:
        raise TTSUnavailableError(f"Speech generation failed: {e}")
