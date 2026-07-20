"""
Minimal example client for the streaming websocket endpoint.

Streams a local 16kHz mono WAV file's raw PCM16 samples to the server
in chunks, prints partial/final results.

Usage:
    python client_stream_example.py path/to/audio_16k_mono.wav
"""

import asyncio
import json
import sys
import wave

import websockets

WS_URL = "ws://localhost:8000/v1/audio/transcriptions/stream"
CHUNK_FRAMES = 3200  # ~0.2s at 16kHz


async def stream_file(path: str):
    with wave.open(path, "rb") as wf:
        if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            print(
                "Warning: server assumes 16kHz mono 16-bit PCM. "
                f"This file is {wf.getframerate()}Hz, {wf.getnchannels()}ch, "
                f"{wf.getsampwidth() * 8}-bit. Convert it first (e.g. with ffmpeg)."
            )

        async with websockets.connect(WS_URL) as ws:
            while True:
                frames = wf.readframes(CHUNK_FRAMES)
                if not frames:
                    break
                await ws.send(frames)

                try:
                    reply = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    print(json.loads(reply))
                except asyncio.TimeoutError:
                    pass

            await ws.send(json.dumps({"type": "stop"}))
            final = await ws.recv()
            print("FINAL:", json.loads(final))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python client_stream_example.py <16k_mono.wav>")
        sys.exit(1)
    asyncio.run(stream_file(sys.argv[1]))
