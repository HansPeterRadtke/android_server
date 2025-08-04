import asyncio
import websockets
import json
import torch
import time
from datetime import datetime
from transformers import pipeline

asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0)

clients = set()
audio_buffers = {}
timers = {}


async def transcribe_and_send(websocket, audio_data):
  print("[DEBUG] Starting transcription with", len(audio_data), "bytes")
  start_pre = time.time()
  input_values = asr_pipe.feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt").input_values.to(0)
  end_pre = time.time()
  print("[DEBUG] Preprocessing took", round((end_pre - start_pre) * 1000), "ms")

  with torch.inference_mode():
    start_gen = time.time()
    prediction = asr_pipe.model.generate(input_values, num_beams=1)
    end_gen = time.time()

  decoded = asr_pipe.tokenizer.batch_decode(prediction, skip_special_tokens=True)[0]
  end_total = time.time()
  print("[DEBUG] Model.generate() took", round((end_gen - start_gen) * 1000), "ms")
  print("[DEBUG] Total model processing took", round((end_total - start_pre) * 1000), "ms")

  await websocket.send(json.dumps({"transcription": decoded}))


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
  async with websockets.serve(handle_client, "0.0.0.0", 8888, max_size = (2**23), reuse_address = True):
    print("[INFO] Server started on port 8888")
    await asyncio.Future()  # run forever

if __name__ == "__main__":
  asyncio.run(main())
