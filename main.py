import os
import io
import re
import wave
import secrets
import asyncio
import logging
from math import gcd

import httpx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import Response
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("desk-buddy")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEVICE_TOKEN = os.getenv("DEVICE_TOKEN")

if not GROQ_API_KEY or not DEVICE_TOKEN:
    raise RuntimeError("Missing GROQ_API_KEY or DEVICE_TOKEN")

app = FastAPI(title="Desk Buddy AI Backend")

GROQ_BASE = "https://api.groq.com/openai/v1"
HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}"}

STT_MODEL = "whisper-large-v3"
CHAT_MODEL = "openai/gpt-oss-120b"
TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "troy"

TARGET_RATE = 16000
MAX_UPLOAD_BYTES = 1_000_000
MAX_SPEECH_CHARS = 190   # Orpheus accepts up to 200 characters

SYSTEM_PROMPT = """
You are Desk Buddy, a friendly desktop voice assistant.

Rules:
1. Always answer in natural English, even if the question was asked in Hindi or Hinglish.
2. Your answer will be spoken aloud, so keep it to one to three short sentences.
3. Do not use lists, markdown, emojis or symbols.
4. Be friendly, accurate and helpful.
5. Do not claim to have performed actions you cannot perform.
"""

# Short-term memory per device. Resets when the server restarts.
conversations = {}

# Convert typographic characters to plain ones so speech sounds clean.
PUNCT_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
    "\u2026": "...",
})


class ProviderError(Exception):
    pass


def authenticate(token: str):
    if not token or not secrets.compare_digest(
        token.encode(), DEVICE_TOKEN.encode()
    ):
        raise HTTPException(status_code=401, detail="Unauthorized device")


def validate_wav(data: bytes):
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio is empty or too large")

    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if wav.getnchannels() != 1:
                raise ValueError("mono audio required")
            if wav.getsampwidth() != 2:
                raise ValueError("16-bit audio required")
            if wav.getframerate() != TARGET_RATE:
                raise ValueError("16 kHz audio required")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid WAV: {exc}")


async def groq_post(client: httpx.AsyncClient, endpoint: str, **kwargs) -> httpx.Response:
    """Calls Groq. Retries once on network errors and temporary server errors."""
    last_problem = "unknown"

    for attempt in range(2):
        try:
            response = await client.post(
                f"{GROQ_BASE}/{endpoint}",
                headers=HEADERS,
                **kwargs
            )
        except httpx.HTTPError as exc:
            last_problem = f"network error: {exc!r}"
        else:
            if response.status_code == 200:
                return response
            last_problem = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code not in (429, 500, 502, 503, 504):
                break

        if attempt == 0:
            await asyncio.sleep(1.5)

    raise ProviderError(f"{endpoint} failed: {last_problem}")


async def transcribe(client: httpx.AsyncClient, wav_bytes: bytes) -> str:
    response = await groq_post(
        client,
        "audio/transcriptions",
        files={"file": ("recording.wav", wav_bytes, "audio/wav")},
        data={
            "model": STT_MODEL,
            "response_format": "json",
            "temperature": "0",
        },
    )
    return (response.json().get("text") or "").strip()


async def ask_llm(client: httpx.AsyncClient, messages: list) -> str:
    response = await groq_post(
        client,
        "chat/completions",
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "max_completion_tokens": 600,
            "reasoning_effort": "low",
        },
    )
    choice = response.json()["choices"][0]
    content = (choice.get("message") or {}).get("content") or ""
    return content.strip()


def clean_for_speech(text: str) -> str:
    text = text.translate(PUNCT_MAP)
    text = re.sub(r"[*_`#>|~\[\]{}]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= MAX_SPEECH_CHARS:
        return text

    # Cut at the last full sentence when possible, otherwise at a word.
    cut = text[:MAX_SPEECH_CHARS]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if end > 40:
        return cut[:end + 1]

    return cut.rsplit(" ", 1)[0].rstrip(",;:") + "."


async def synthesize(client: httpx.AsyncClient, text: str) -> bytes:
    response = await groq_post(
        client,
        "audio/speech",
        json={
            "model": TTS_MODEL,
            "input": text,
            "voice": TTS_VOICE,
            "response_format": "wav",
        },
    )
    return to_device_wav(response.content)


def to_device_wav(source_wav: bytes) -> bytes:
    """Converts any WAV from the speech service to 16 kHz mono 16-bit."""
    try:
        audio, rate = sf.read(
            io.BytesIO(source_wav),
            dtype="float32",
            always_2d=True
        )
    except Exception as exc:
        raise ProviderError(f"speech audio unreadable: {exc}")

    if audio.size == 0:
        raise ProviderError("speech audio is empty")

    audio = audio.mean(axis=1)

    if rate != TARGET_RATE:
        divisor = gcd(rate, TARGET_RATE)
        audio = resample_poly(audio, TARGET_RATE // divisor, rate // divisor)

    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype("<i2")

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(TARGET_RATE)
        wav.writeframes(pcm.tobytes())

    return output.getvalue()


def build_messages(conv_key: str, transcript: str) -> list:
    history = conversations.get(conv_key, [])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": transcript})
    return messages


def remember(conv_key: str, transcript: str, answer: str):
    history = conversations.get(conv_key, [])
    history.extend([
        {"role": "user", "content": transcript},
        {"role": "assistant", "content": answer},
    ])
    conversations[conv_key] = history[-10:]


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"device": "Desk Buddy", "status": "online"}


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "healthy"}


@app.post("/talk")
async def talk(
    file: UploadFile = File(...),
    conversation_id: str = Header(default="default"),
    x_device_token: str = Header(default=""),
):
    authenticate(x_device_token)

    audio_data = await file.read()
    validate_wav(audio_data)

    conv_key = conversation_id[:64]

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0)
        ) as client:

            # 1. Speech to text
            transcript = await transcribe(client, audio_data)
            log.info("Heard: %r", transcript)

            # 2. AI answer
            if transcript:
                answer = await ask_llm(client, build_messages(conv_key, transcript))
                log.info("Answer: %r", answer)

                speech = clean_for_speech(answer) if answer else ""

                if speech:
                    remember(conv_key, transcript, answer)
                else:
                    speech = "Sorry, I could not think of an answer. Please ask again."
            else:
                speech = "Sorry, I did not catch that. Please try again."

            # 3. Text to speech
            wav = await synthesize(client, speech)

    except ProviderError as exc:
        log.error("Provider problem: %s", exc)
        raise HTTPException(status_code=503, detail="AI service temporarily unavailable")
    except Exception:
        log.exception("Unexpected error while handling /talk")
        raise HTTPException(status_code=503, detail="Internal error, please try again")

    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )
