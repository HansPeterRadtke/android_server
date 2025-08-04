import asyncio
import websockets
import json
import torch
import time
import numpy as np
import traceback
from datetime import datetime
print("Importing transformers ...")
from transformers import pipeline
print("Importing transformers DONE")

print("creating pipeline ...")
asr_pipe = pipeline("automatic-speech-recognition", model="../../models/whisper-small")
print("creating pipeline DONE")

clients = set()
audio_buffers = {}
timers = {}


async def transcribe_and_send(websocket, audio_data):
  print("[DEBUG] Starting transcription with", len(audio_data), "bytes")
  start = time.time()

  try:
    waveform = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    result = asr_pipe(waveform)
    end = time.time()

    print("[DEBUG] Transcription took", round((end - start) * 1000), "ms")
    print("[DEBUG] Transcribed text:", result["text"])
    await websocket.send(json.dumps({"transcription": result["text"]}))

  except Exception as e:
    print("[ERROR] Transcription failed:", e)
    traceback.print_exc()


async def handle_client(websocket):
  print("[INFO] WebSocket Connected")
  clients.add(websocket)
  audio_buffers[websocket] = bytearray()
  timers[websocket] = None

  async def reset_timer():
    if timers[websocket]:
      timers[websocket].cancel()
    timers[websocket] = asyncio.get_event_loop().call_later(1.5, lambda: asyncio.create_task(finish_audio(websocket)))

  async def finish_audio(ws):
    if audio_buffers[ws]:
      await transcribe_and_send(ws, bytes(audio_buffers[ws]))
      audio_buffers[ws].clear()

  try:
    async for message in websocket:
      if isinstance(message, str):
        try:
          data = json.loads(message)
          if data.get("audio"):
            print("[INFO] Started recording.")
        except:
          print("[ERROR] Could not parse JSON string.")
      else:
        audio_buffers[websocket].extend(message)
        await reset_timer()

  except websockets.exceptions.ConnectionClosed:
    print("[INFO] Client disconnected")
  finally:
    await finish_audio(websocket)
    clients.discard(websocket)
    audio_buffers.pop(websocket, None)
    if timers[websocket]:
      timers[websocket].cancel()
    timers.pop(websocket, None)


async def main():
  async with websockets.serve(handle_client, "0.0.0.0", 8888, max_size=(2**23), reuse_address=True):
    print("[INFO] Server started on port 8888")
    await asyncio.Future()  # run forever

if __name__ == "__main__":
  asyncio.run(main())
