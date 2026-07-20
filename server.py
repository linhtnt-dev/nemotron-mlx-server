"""
Nemotron ASR Server
====================

Single-file FastAPI server that loads a local Nemotron ASR model once
(via mlx-audio) and exposes:

  * POST /v1/audio/transcriptions           - upload a full audio file
  * WS   /v1/audio/transcriptions/stream    - realtime streaming audio (raw PCM16)
  * GET  /health                            - health check
  * GET  /v1/models                         - OpenAI-style model listing stub

IMPORTANT - read before running:
  1. Set MODEL_PATH below to your local Nemotron checkpoint directory.
  2. The exact mlx-audio Nemotron loading/inference call may differ by
     version. The `load_model` / `model.transcribe` calls are the most
     likely API shape as of mid-2025, but verify against the version you
     installed (`pip show mlx-audio`, check its README/source).
  3. The streaming endpoint here still re-transcribes the *entire*
     accumulated buffer on every threshold hit. That is NOT true
     incremental streaming - it works but adds latency and repeats
     compute. See the "Real streaming" note near the bottom of this
     file for what a proper RNN-T / cache-based implementation needs.
  4. This server is OpenAI-*shaped* for the basic JSON response
     ({"text": ...}), but it is NOT a drop-in OpenAI-compatible
     endpoint. Missing: `model` field handling, `response_format`
     (json/text/srt/vtt/verbose_json), auth header, and OpenAI error
     envelope. The websocket stream protocol is custom - OpenAI's own
     Realtime API uses a different event-based protocol
     (session.created, input_audio_buffer.append, etc.), which this
     does not replicate.
"""

import io
import json
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.environ.get("NEMOTRON_MODEL_PATH", "/path/to/your/nemotron-model")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

# Streaming assumptions - raw PCM16 mono audio pushed over the websocket.
STREAM_SAMPLE_RATE = int(os.environ.get("STREAM_SAMPLE_RATE", "16000"))
STREAM_CHANNELS = 1
STREAM_CHUNK_TRIGGER_BYTES = int(os.environ.get("STREAM_CHUNK_TRIGGER_BYTES", "32000"))  # ~1s at 16kHz/16-bit

# ============================================================
# MODEL
# ============================================================

print(f"Loading Nemotron ASR from: {MODEL_PATH}")

# Adjust this import/API depending on the exact mlx-audio
# Nemotron implementation you installed.
from mlx_audio.stt import load_model

model = load_model(MODEL_PATH)

print("Nemotron ASR loaded successfully.")


def _extract_text(result) -> str:
    """Normalize whatever mlx-audio returns into a plain string."""
    if hasattr(result, "text"):
        return result.text
    if isinstance(result, dict):
        return result.get("text", "")
    return str(result)


def _pcm16_bytes_to_wav_file(pcm_bytes: bytes, path: str, sample_rate: int, channels: int) -> None:
    """Wrap raw PCM16 bytes in a valid WAV container and write to `path`.

    Writing raw bytes straight into a file with a `.wav` suffix (without a
    real WAV header) is invalid and most decoders will reject it. This
    builds a minimal, correct RIFF/WAVE header instead of depending on an
    extra numpy/soundfile round-trip.
    """
    import wave

    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Nemotron ASR Server",
    version="1.0.0",
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
    }


# ============================================================
# OpenAI-style model listing stub (cosmetic compatibility only)
# ============================================================

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "nemotron-asr",
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


# ============================================================
# NORMAL TRANSCRIPTION
# OpenAI-shaped endpoint (see module docstring for caveats)
# ============================================================

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
):
    """
    Example:

    curl http://localhost:8000/v1/audio/transcriptions \\
      -X POST \\
      -F "file=@audio.wav"
    """

    audio_bytes = await file.read()

    if not audio_bytes:
        return JSONResponse(status_code=400, content={"error": "empty file"})

    suffix = os.path.splitext(file.filename or ".wav")[1] or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
        temp_file.write(audio_bytes)
        temp_file.flush()

        try:
            result = model.transcribe(temp_file.name)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return JSONResponse(content={"text": _extract_text(result)})


# ============================================================
# REALTIME WEBSOCKET STREAMING
# ============================================================

@app.websocket("/v1/audio/transcriptions/stream")
async def transcribe_stream(websocket: WebSocket):
    """Streaming protocol (custom, NOT OpenAI Realtime API compatible):

    Client -> server:
      - binary frames: raw PCM16 mono audio at STREAM_SAMPLE_RATE
      - text frame: {"type": "stop"}  -> finalize and close

    Server -> client:
      - {"type": "partial", "text": "..."}  after each chunk threshold
      - {"type": "final", "text": "..."}    on stop
      - {"type": "error", "error": "..."}   on failure
    """

    await websocket.accept()
    print("WebSocket client connected.")

    audio_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()

            # ------------------------------------------------
            # BINARY AUDIO CHUNK
            # ------------------------------------------------
            if "bytes" in message and message["bytes"] is not None:
                chunk = message["bytes"]
                audio_buffer.extend(chunk)

                # NOTE: this re-transcribes the whole buffer each time.
                # See module docstring - this is a placeholder, not true
                # incremental streaming.
                if len(audio_buffer) > STREAM_CHUNK_TRIGGER_BYTES:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                        _pcm16_bytes_to_wav_file(
                            bytes(audio_buffer),
                            temp_file.name,
                            STREAM_SAMPLE_RATE,
                            STREAM_CHANNELS,
                        )
                        try:
                            result = model.transcribe(temp_file.name)
                            await websocket.send_json(
                                {"type": "partial", "text": _extract_text(result)}
                            )
                        except Exception as e:
                            await websocket.send_json({"type": "error", "error": str(e)})

            # ------------------------------------------------
            # JSON CONTROL MESSAGE
            # ------------------------------------------------
            elif "text" in message and message["text"] is not None:
                data = json.loads(message["text"])

                if data.get("type") == "stop":
                    text = ""
                    if audio_buffer:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                            _pcm16_bytes_to_wav_file(
                                bytes(audio_buffer),
                                temp_file.name,
                                STREAM_SAMPLE_RATE,
                                STREAM_CHANNELS,
                            )
                            try:
                                result = model.transcribe(temp_file.name)
                                text = _extract_text(result)
                            except Exception as e:
                                await websocket.send_json({"type": "error", "error": str(e)})

                    await websocket.send_json({"type": "final", "text": text})
                    await websocket.close()
                    break

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")


# ============================================================
# "Real streaming" note
# ============================================================
# For low-latency production streaming you do not want to re-run
# model.transcribe() on the whole growing buffer. Nemotron's
# RNN-T/streaming variants expose (depending on the exact mlx-audio
# release) something like an encoder cache / decoder state object that
# you feed fixed-size audio frames into incrementally, pulling out
# partial hypotheses per frame instead of per full-buffer decode. Check
# the specific model card / mlx-audio source for a `streaming_transcribe`,
# `StreamingState`, or similar API before shipping this to production -
# the code above intentionally keeps the naive "re-decode the buffer"
# approach so it runs today with a stock mlx-audio Nemotron build.


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
