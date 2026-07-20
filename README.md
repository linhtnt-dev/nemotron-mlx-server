# Nemotron ASR Server

Single-file FastAPI server that loads a local NVIDIA Nemotron ASR model (via
`mlx-audio`) once and serves it over HTTP (file upload) and WebSocket
(streaming).

## Endpoints

- `GET /health` — health check
- `GET /v1/models` — OpenAI-style model listing stub
- `POST /v1/audio/transcriptions` — upload an audio file, get back `{"text": ...}`
- `WS /v1/audio/transcriptions/stream` — stream raw PCM16 audio, get partial/final text

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit `server.py` and set `MODEL_PATH` to your local Nemotron checkpoint
directory, or export it as an env var:

```bash
export NEMOTRON_MODEL_PATH=/path/to/your/nemotron-model
```

## Run

```bash
python server.py
```

Server listens on `http://0.0.0.0:8000` by default (override with `HOST`/`PORT` env vars).

## Test

Health check:

```bash
curl http://localhost:8000/health
```

Transcribe a file:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav"
```

Streaming (requires a 16kHz mono 16-bit PCM WAV file):

```bash
pip install websockets
python client_stream_example.py path/to/audio_16k_mono.wav
```

## Is this OpenAI-compatible?

Short answer: **partially, and only for the non-streaming endpoint.**

What matches OpenAI's `/v1/audio/transcriptions`:
- Same path and method (`POST /v1/audio/transcriptions`)
- Multipart file upload via a `file` field
- Default JSON response shape `{"text": "..."}`

What's missing or different, so this is not a true drop-in replacement:
- No handling of OpenAI's other request params (`model`, `language`, `prompt`,
  `temperature`, `response_format` for `text`/`srt`/`vtt`/`verbose_json`)
- No OpenAI-style error envelope or auth (`Authorization: Bearer ...`) handling
- No timestamps/segments (`verbose_json` output) unless you add them
- The streaming endpoint is a **custom protocol** (raw PCM16 binary frames +
  a `{"type": "stop"}` control message). OpenAI's actual real-time
  transcription is a different, event-based WebSocket protocol
  (`session.created`, `input_audio_buffer.append`,
  `conversation.item.input_audio_transcription.completed`, etc.) — this
  server does not implement that protocol.

If you need real OpenAI SDK/client compatibility (e.g. so `openai-python`'s
`client.audio.transcriptions.create(...)` works against this server
unmodified), you'd need to add the missing request params and match their
response schema more closely. The streaming side would need a full rewrite
to speak OpenAI's Realtime API event protocol rather than this raw-PCM
approach.

## Known limitations / next steps

- The streaming endpoint currently **re-transcribes the entire accumulated
  buffer** every ~1s of audio rather than doing true incremental decoding.
  This works but wastes compute and adds latency that grows with buffer
  size. For production-grade low-latency streaming, use Nemotron's
  streaming/RNN-T state-caching API (if exposed by your `mlx-audio` version)
  to feed fixed-size frames incrementally and emit partials per frame
  instead of per full-buffer re-decode.
- `model.transcribe(...)` and `load_model(...)` calls are best-effort based
  on the `mlx-audio` STT API shape; verify against the version you have
  installed (`pip show mlx-audio`) since exact function names/signatures
  can change between releases.
