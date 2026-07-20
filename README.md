# Nemotron ASR Server

Single-file FastAPI server that loads a local NVIDIA Nemotron 3.5 ASR model
(via `mlx-audio`'s Nemotron support) once and serves it over HTTP (OpenAI-
compatible transcription endpoint) and WebSocket (custom streaming protocol).

## Endpoints

- `GET /health` — health check
- `GET /v1/models` — OpenAI-style model listing
- `POST /v1/audio/transcriptions` — OpenAI-compatible transcription (`model`, `language`, `prompt`, `response_format`, `temperature`)
- `WS /v1/audio/transcriptions/stream` — stream raw PCM16 audio, get partial/final text (custom protocol, see below)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nemotron support isn't in a PyPI release of `mlx-audio` yet (latest is
0.4.3) — it's merged into `main`, hence the git install in
`requirements.txt`:

```
pip install "git+https://github.com/Blaizzy/mlx-audio.git"
```

Once it ships in a release, `pip install -U mlx-audio` will work instead.

Set `MODEL_PATH` in `server.py`, or export it as an env var — either a
local checkpoint directory or a Hub repo id such as
`mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` (mlx-audio's `load()`
downloads it if it's not a local path):

```bash
export NEMOTRON_MODEL_PATH=mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit
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
curl http://localhost:8000/v1/audio/transcriptions \
  -X POST \
  -F "file=@audio.wav" \
  -F "model=nemotron-asr"
```

With a specific language (Nemotron's language-ID prompt conditioning,
e.g. `en-US`, `vi-VN`, `zh-CN`):

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -X POST \
  -F "file=@audio.wav" \
  -F "model=nemotron-asr" \
  -F "language=vi-VN"
```

Plain-text response:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -X POST \
  -F "file=@audio.wav" \
  -F "model=nemotron-asr" \
  -F "response_format=text"
```

With the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")

with open("audio.wav", "rb") as audio:
    result = client.audio.transcriptions.create(model="nemotron-asr", file=audio)

print(result.text)
```

Streaming (requires a 16kHz mono 16-bit PCM WAV file):

```bash
pip install websockets
python client_stream_example.py path/to/audio_16k_mono.wav
```

## Is this OpenAI-compatible?

**Yes, for `/v1/audio/transcriptions` — no, for streaming.**

What matches OpenAI's Audio Transcriptions API:
- Same path/method, multipart `file` upload
- `model`, `language`, `prompt`, `temperature`, `response_format`
  (`json` / `text` / `verbose_json`) request params
- Default JSON response shape `{"text": "..."}`
- Works unmodified with `openai-python`'s
  `client.audio.transcriptions.create(model="nemotron-asr", file=...)`

What's still different from the real API:
- No auth check — any API key is accepted, there's no real key validation
- No OpenAI-style error envelope (`{"error": {...}}`) — errors are plain
  FastAPI/HTTPException JSON
- `verbose_json` returns `segments: []` — no word/segment timestamps
- `client.chat.completions.create(...)` has no meaning here — this only
  implements the transcription endpoint, not Chat Completions
- The WebSocket stream is a **custom protocol** (raw PCM16 binary frames +
  a `{"type": "stop"}` control message), not OpenAI's Realtime API, which
  uses a different event-based protocol (`session.created`,
  `input_audio_buffer.append`,
  `conversation.item.input_audio_transcription.completed`, etc.)

## Known limitations / next steps

- The streaming endpoint currently **re-transcribes the entire accumulated
  buffer** every ~1s of audio rather than doing true incremental decoding.
  This works but wastes compute and adds latency that grows with buffer
  size. Nemotron 3.5 is a cache-aware streaming FastConformer-RNNT model,
  so a production implementation should feed fixed-size frames into its
  streaming cache/state (check mlx-audio's Nemotron streaming API for an
  `att_context_size` / cache object) and emit partials per frame instead
  of per full-buffer re-decode.
- `model.generate(...)` and `load(...)` calls follow the API documented on
  the [mlx-community model card](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit)
  as of this writing; re-check against the `mlx-audio` version you have
  installed since it's a pre-release API on `main`.
