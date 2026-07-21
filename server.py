"""
Nemotron ASR Server (OpenAI-compatible /v1/audio/transcriptions)
=================================================================

Single-file FastAPI server that loads a local Nemotron 3.5 ASR model once
(via mlx-audio's Nemotron support) and exposes:

  * GET  /health                            - health check
  * GET  /v1/models                         - OpenAI-style model listing
  * POST /v1/audio/transcriptions           - OpenAI-compatible transcription
  * WS   /v1/audio/transcriptions/stream    - realtime streaming audio (custom protocol)

Install:
    pip install "git+https://github.com/Blaizzy/mlx-audio.git" fastapi uvicorn python-multipart

  Nemotron support isn't in a PyPI release of mlx-audio yet (latest at time
  of writing is 0.4.3) - it's merged into `main`, hence the git install.
  Once it ships in a release, `pip install -U mlx-audio` will work instead.

Before running:
  1. Set MODEL_PATH below (or the NEMOTRON_MODEL_PATH env var) to your local
     Nemotron checkpoint directory, or a mlx-community repo id such as
     "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit" (mlx-audio's
     `load()` will download it from the Hub if it's not a local path).
  2. Verify the mlx-audio API shape against the version you installed -
     confirmed against the mlx-community model card as of this writing:
     `from mlx_audio.stt import load; model = load(path);
     model.generate(audio_path, language=...).text`

OpenAI compatibility notes:
  * POST /v1/audio/transcriptions matches OpenAI's request shape
    (multipart `file`, `model`, `language`, `prompt`, `response_format`,
    `temperature`) and default JSON response `{"text": ...}`, so
    `openai` SDK's `client.audio.transcriptions.create(...)` works
    against this server unmodified.
  * `verbose_json` returns `segments: []` here - this server does not
    produce word/segment timestamps, so if your client depends on those,
    they won't be populated.
  * There's no auth check (any API key is accepted) and no OpenAI error
    envelope (`{"error": {...}}`) - errors are plain FastAPI/HTTPException
    JSON.
  * This is NOT the OpenAI Realtime API. The /stream websocket below is a
    custom protocol (raw PCM16 binary frames + a `{"type": "stop"}`
    control message) - OpenAI's actual real-time transcription protocol
    (`session.created`, `input_audio_buffer.append`, etc.) is not
    implemented. `client.chat.completions.create(...)` also has no
    meaning against an ASR-only server like this one.
"""

import json
import os
import secrets
import tempfile
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse

from mlx_audio.stt import load

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = os.environ.get("NEMOTRON_MODEL_PATH", "/absolute/path/to/your/nemotron-model")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

MODEL_ID = "nemotron-asr"

# --------------------------------------------------------------
# Hardcoded API key (dev/local use only!)
#
# Override with an env var before exposing this server beyond
# localhost:
#   export NEMOTRON_API_KEY="something-you-generate-yourself"
# --------------------------------------------------------------
API_KEY = os.environ.get("NEMOTRON_API_KEY", "sk-nemotron-local-9f3a1c7d4e2b8f01")

# Streaming websocket assumptions - raw PCM16 mono audio.
STREAM_SAMPLE_RATE = int(os.environ.get("STREAM_SAMPLE_RATE", "16000"))
STREAM_CHANNELS = 1
STREAM_CHUNK_TRIGGER_BYTES = int(os.environ.get("STREAM_CHUNK_TRIGGER_BYTES", "32000"))  # ~1s at 16kHz/16-bit

# ============================================================
# LOAD MODEL ONCE
# ============================================================

print(f"Loading model from: {MODEL_PATH}")

model = load(MODEL_PATH)

print("Model loaded successfully.")


def _openai_error(message: str, err_type: str = "invalid_request_error", code: Optional[str] = None) -> dict:
    return {"error": {"message": message, "type": err_type, "code": code}}


async def verify_api_key(authorization: Optional[str] = Header(default=None)) -> None:
    """OpenAI-style bearer auth: `Authorization: Bearer <API_KEY>`."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail=_openai_error(
                "You didn't provide an API key. You need to provide your API key in an "
                "Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY).",
                code="missing_api_key",
            ),
        )

    token = authorization[len("Bearer "):].strip()
    if not secrets.compare_digest(token, API_KEY):
        raise HTTPException(
            status_code=401,
            detail=_openai_error("Incorrect API key provided.", code="invalid_api_key"),
        )


def _pcm16_bytes_to_wav_file(pcm_bytes: bytes, path: str, sample_rate: int, channels: int) -> None:
    """Wrap raw PCM16 bytes in a valid WAV container and write to `path`."""
    import wave

    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Nemotron ASR OpenAI-Compatible Server",
    version="1.0.0",
)


@app.exception_handler(StarletteHTTPException)
async def openai_style_exception_handler(request: Request, exc: StarletteHTTPException):
    """Make every HTTPException in this app render as OpenAI's {"error": {...}} envelope."""
    detail = exc.detail
    content = detail if isinstance(detail, dict) and "error" in detail else _openai_error(str(detail))
    return JSONResponse(status_code=exc.status_code, content=content)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
    }


# ============================================================
# OPENAI-COMPATIBLE: GET /v1/models
# ============================================================

@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


# ============================================================
# OPENAI-COMPATIBLE:
#
# POST /v1/audio/transcriptions
#
# Compatible with:
#
# client.audio.transcriptions.create(
#     model="nemotron-asr",
#     file=audio_file,
# )
# ============================================================

@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_api_key)])
async def create_transcription(
    file: UploadFile = File(...),
    model_name: str = Form(MODEL_ID, alias="model"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    # --------------------------------------------------------
    # Validate model
    # --------------------------------------------------------
    if model_name != MODEL_ID:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    # --------------------------------------------------------
    # Validate response format
    # --------------------------------------------------------
    supported_formats = {"json", "text", "verbose_json"}
    if response_format not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{response_format}'",
        )

    # --------------------------------------------------------
    # Save uploaded audio temporarily
    # --------------------------------------------------------
    suffix = os.path.splitext(file.filename or ".wav")[1]
    audio_data = await file.read()

    if not audio_data:
        raise HTTPException(status_code=400, detail="empty file")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(audio_data)

    try:
        # ----------------------------------------------------
        # RUN NEMOTRON ASR
        # ----------------------------------------------------
        generate_kwargs = {
            "audio": temp_path,
            "temperature": temperature,
        }

        # Nemotron 3.5 ASR supports language-ID prompt conditioning,
        # e.g. language="en-US", "vi-VN", "zh-CN". Omit for auto-detect.
        if language:
            generate_kwargs["language"] = language

        result = model.generate(**generate_kwargs)
        text = result.text

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    # ========================================================
    # OPENAI RESPONSE FORMATS
    # ========================================================

    if response_format == "text":
        return JSONResponse(content=text, media_type="text/plain")

    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": language,
            "duration": None,
            "text": text,
            "segments": [],  # not populated - see module docstring
        }

    # Default: json
    return {"text": text}


# ============================================================
# REALTIME WEBSOCKET STREAMING (custom protocol, not OpenAI Realtime API)
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

    NOTE: this re-decodes the whole accumulated buffer on every threshold
    hit rather than doing true incremental decoding. Nemotron 3.5 is a
    cache-aware streaming FastConformer-RNNT model, so a production
    implementation should feed fixed-size frames into its streaming
    cache/state (check mlx-audio's Nemotron streaming API - e.g. an
    `att_context_size` / cache object) and emit partials per frame instead
    of re-running `generate()` over the whole growing buffer.
    """

    # Auth: accept either `?api_key=...` or an `Authorization: Bearer ...`
    # header on the handshake (browsers can't set custom headers on WS
    # connections, so the query param is the practical option for JS clients).
    supplied_key = websocket.query_params.get("api_key")
    if not supplied_key:
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            supplied_key = auth_header[len("Bearer "):].strip()

    if not supplied_key or not secrets.compare_digest(supplied_key, API_KEY):
        await websocket.close(code=4401, reason="invalid or missing api_key")
        return

    await websocket.accept()
    print("WebSocket client connected.")

    audio_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                chunk = message["bytes"]
                audio_buffer.extend(chunk)

                if len(audio_buffer) > STREAM_CHUNK_TRIGGER_BYTES:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                        _pcm16_bytes_to_wav_file(
                            bytes(audio_buffer),
                            temp_file.name,
                            STREAM_SAMPLE_RATE,
                            STREAM_CHANNELS,
                        )
                        try:
                            result = model.generate(audio=temp_file.name)
                            await websocket.send_json({"type": "partial", "text": result.text})
                        except Exception as e:
                            await websocket.send_json({"type": "error", "error": str(e)})

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
                                result = model.generate(audio=temp_file.name)
                                text = result.text
                            except Exception as e:
                                await websocket.send_json({"type": "error", "error": str(e)})

                    await websocket.send_json({"type": "final", "text": text})
                    await websocket.close()
                    break

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
