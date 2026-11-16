
import os
import io
import wave
import secrets
import asyncio

import httpx
import numpy as np
import soundfile as sf

from scipy.signal import resample_poly
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import Response
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEVICE_TOKEN = os.getenv("DEVICE_TOKEN")

if not GROQ_API_KEY or not DEVICE_TOKEN:
    raise RuntimeError("Missing GROQ_API_KEY or DEVICE_TOKEN")

app = FastAPI(title="Desk Buddy AI Backend")

GROQ_BASE = "https://api.groq.com/openai/v1"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}"
}

SYSTEM_PROMPT = """
You are Desk Buddy, a friendly, intelligent desktop AI assistant.

Your rules:
1. Always respond in natural English.
2. Understand questions transcribed from both Hindi and English.
3. Be friendly, helpful and concise.
4. Prefer answers that can be spoken naturally.
5. Avoid unnecessarily long responses.
6. Do not claim to have performed actions you cannot perform.
7. You are an educational and productivity assistant.
"""

# Short-term conversation memory.
# This is in-process memory and resets when the server restarts.
conversations = {}


def authenticate(token):
    if not token or not secrets.compare_digest(
        token, DEVICE_TOKEN
    ):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized device"
        )


async def groq_request(client, endpoint, **kwargs):
    response = await client.post(
        f"{GROQ_BASE}/{endpoint}",
        headers=HEADERS,
        **kwargs
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"AI provider error: {response.text[:300]}"
        )

    return response


def normalize_wav(audio_bytes):
    audio, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32",
        always_2d=True
    )

    # Convert to mono.
    audio = np.mean(audio, axis=1)

    if sample_rate != 16000:
        from math import gcd
        divisor = gcd(sample_rate, 16000)

        audio = resample_poly(
            audio,
            16000 // divisor,
            sample_rate // divisor
        )

    audio = np.clip(audio, -1.0, 1.0)

    pcm = (audio * 32767).astype("<i2")

    output = io.BytesIO()

    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm.tobytes())

    return output.getvalue()


@app.get("/")
async def root():
    return {
        "device": "Desk Buddy",
        "status": "online"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/talk")
async def talk(
    file: UploadFile = File(...),
    conversation_id: str = Header(default="default"),
    x_device_token: str = Header(default="")
):
    authenticate(x_device_token)

    if file.content_type not in (
        "audio/wav",
        "audio/x-wav",
        "application/octet-stream"
    ):
        raise HTTPException(
            status_code=415,
            detail="WAV audio required"
        )

    audio_data = await file.read()

    if not audio_data or len(audio_data) > 1_000_000:
        raise HTTPException(
            status_code=413,
            detail="Audio is empty or too large"
        )

    try:
        with wave.open(io.BytesIO(audio_data), "rb") as wav:
            if wav.getnchannels() != 1:
                raise ValueError("Mono audio required")
            if wav.getsampwidth() != 2:
                raise ValueError("16-bit audio required")
            if wav.getframerate() != 16000:
                raise ValueError("16 kHz audio required")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid WAV: {exc}"
        )

    async with httpx.AsyncClient(timeout=90.0) as client:

        # 1. Speech recognition.
        transcription_response = await groq_request(
            client,
            "audio/transcriptions",
            files={
                "file": (
                    "recording.wav",
                    audio_data,
                    "audio/wav"
                )
            },
            data={
                "model": "whisper-large-v3",
                "response_format": "json",
                "temperature": "0"
            }
        )

        transcript = transcription_response.json().get("text", "").strip()

        if not transcript:
            raise HTTPException(
                status_code=422,
                detail="No speech detected"
            )

        # 2. Retrieve recent conversation.
        history = conversations.get(conversation_id, [])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        messages.extend(history[-6:])

        messages.append({
            "role": "user",
            "content": transcript
        })

        # 3. Generate AI response.
        chat_response = await groq_request(
            client,
            "chat/completions",
            json={
                "model": "openai/gpt-oss-120b",
                "messages": messages,
                "max_completion_tokens": 160
            }
        )

        answer = (
            chat_response.json()["choices"][0]
            ["message"]["content"]
        ).strip()

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="Empty AI response"
            )

        # Orpheus currently supports 200 input characters.
        speech_text = answer[:195]

        # 4. Generate English speech.
        tts_response = await groq_request(
            client,
            "audio/speech",
            json={
                "model": "canopylabs/orpheus-v1-english",
                "input": speech_text,
                "voice": "troy",
                "response_format": "wav"
            }
        )

        # 5. Normalize the WAV for ESP32.
        try:
            final_audio = normalize_wav(tts_response.content)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Audio conversion failed: {exc}"
            )

        # Update short-term conversation history.
        history.extend([
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": answer}
        ])

        conversations[conversation_id] = history[-10:]

        return Response(
            content=final_audio,
            media_type="audio/wav",
            headers={
                "X-Transcript": transcript[:200],
                "X-Assistant-Text": speech_text[:195],
                "Cache-Control": "no-store"
            }
        )
