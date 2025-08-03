print("[BOOT] TOP OF SCRIPT REACHED")
import asyncio
import websockets
import json
import traceback
import sys
import os
import torch
import numpy as np
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration

PORT = 8888

print("[BOOT] WebSocket server script loaded.")
sys.stdout.flush()

try:
  print("[BOOT] Loading model and processor...")
  sys.stdout.flush()
  model_dir  = "/home/hans/dev/GPT/models/whisper-base"
  processor  = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
  model      = WhisperForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
  model.eval()
  device     = "cuda" if torch.cuda.is_available() else "cpu"
  model      = model.to(device)
  print("[BOOT] Model loaded and ready on device:", device)
  sys.stdout.flush()
except Exception as e:
  print("[ERROR] Model loading failed:", str(e))
  traceback.print_exc()
  sys.stdout.flush()
  sys.exit(1)

async def handle_client(websocket):
  print("[INFO] Entered handle_client()")
  sys.stdout.flush()
  print("[INFO] New client connected.")
  sys.stdout.flush()
  try:
    mode = None
    audio_buffer = bytearray()
    expected_audio_size = 0

    while True:
      msg = await websocket.recv()

      if isinstance(msg, str):
        print("[DEBUG] Received text message:", msg)
        sys.stdout.flush()
        try:
          data = json.loads(msg)
          if "msg" in data:
            reversed_msg = data["msg"][::-1]
            await websocket.send(json.dumps({"result": reversed_msg}))

          elif "size" in data:
            size = int(data["size"])
            received = 0
            next_log = 100000
            while received < size:
              chunk = await websocket.recv()
              received += len(chunk)
              if received >= next_log:
                print(f"[DEBUG] Received {received} bytes")
                sys.stdout.flush()
                next_log += 100000
            chunk = b"x" * 8192
            sent = 0
            while sent < size:
              remaining = size - sent
              to_send = chunk if remaining >= len(chunk) else b"x" * remaining
              await websocket.send(to_send)
              sent += len(to_send)
            print("[INFO] Bandwidth test completed.")
            sys.stdout.flush()

          elif "audio" in data and data["audio"] is True:
            mode = "audio"
            expected_audio_size = int(data.get("length", 0))
            audio_buffer.clear()
            print("[INFO] Ready to receive audio data")
            sys.stdout.flush()

          else:
            await websocket.send(json.dumps({"error": "Unknown command"}))

        except Exception as e:
          print("[ERROR] Failed to parse text message:", str(e))
          traceback.print_exc()
          sys.stdout.flush()

      elif isinstance(msg, bytes) and mode == "audio":
        audio_buffer.extend(msg)
        print(f"[DEBUG] Received audio chunk ({len(msg)} bytes), total: {len(audio_buffer)}")
        sys.stdout.flush()
        if expected_audio_size and len(audio_buffer) >= expected_audio_size:
          print("[INFO] Full audio data received. Starting transcription...")
          sys.stdout.flush()
          try:
            model_start = time.time()
            waveform = torch.frombuffer(audio_buffer, dtype=torch.int16).float() / 32768.0
            waveform = waveform.unsqueeze(0)

            prep_start = time.time()
            inputs = processor(
              waveform[0],
              sampling_rate=16000,
              return_tensors="pt",
              language="en"
            )
            input_features = inputs.input_features.to(device)
            attention_mask = inputs.get("attention_mask")
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
            prep_end = time.time()

            gen_start = time.time()
            if attention_mask is not None:
              attention_mask = attention_mask.to(device)
              predicted_ids = model.generate(input_features, attention_mask=attention_mask, forced_decoder_ids=forced_decoder_ids)
            else:
              predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            gen_end = time.time()

            model_end = time.time()
            print(f"[INFO] Pre-processing took {prep_end - prep_start:.3f} seconds")
            print(f"[INFO] Model.generate() took {gen_end - gen_start:.3f} seconds")
            print(f"[INFO] Total model processing took {model_end - model_start:.3f} seconds")
            sys.stdout.flush()

            result = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print("[INFO] Transcription:", result)
            await websocket.send(json.dumps({"transcription": result}))
          except Exception as e:
            print("[ERROR] Audio processing failed:", str(e))
            traceback.print_exc()
            await websocket.send(json.dumps({"error": "Transcription failed"}))
          mode = None

  except websockets.exceptions.ConnectionClosedError as e:
    print("[INFO] Client disconnected with error:", e)
  except websockets.exceptions.ConnectionClosedOK as e:
    print("[INFO] Client disconnected cleanly:", e)
  except asyncio.CancelledError:
    print("[INFO] handle_client cancelled.")
  except Exception as e:
    print("[ERROR] Connection handler exception:", str(e))
    traceback.print_exc()
  sys.stdout.flush()

async def input_monitor(shutdown_event):
  print("[BOOT] Input monitor started. Type 'exit', 'quit', or 'q' to stop.")
  sys.stdout.flush()
  loop = asyncio.get_running_loop()
  while not shutdown_event.is_set():
    try:
      line = await loop.run_in_executor(None, sys.stdin.readline)
      cmd = line.strip().lower()
      if cmd in ("exit", "quit", "q"):
        print("[INFO] Shutdown command received.")
        sys.stdout.flush()
        shutdown_event.set()
        break
      else:
        print(f"[ERROR] Unknown input command: {cmd}")
        sys.stdout.flush()
    except Exception as e:
      print("[ERROR] Input monitor exception:", str(e))
      traceback.print_exc()
      sys.stdout.flush()
    await asyncio.sleep(0.1)

async def main():
  print("[INFO] WebSocket server starting on port", PORT)
  sys.stdout.flush()
  shutdown_event = asyncio.Event()
  try:
    async with websockets.serve(handle_client, "", PORT):
      print("[INFO] Server is listening for clients...")
      sys.stdout.flush()
      await asyncio.gather(
        input_monitor(shutdown_event),
        shutdown_event.wait()
      )
      print("[INFO] Shutdown signal processed. Server stopping...")
      sys.stdout.flush()
  except asyncio.CancelledError:
    print("[INFO] Server main task cancelled.")
  except Exception as e:
    print("[ERROR] Failed to start server:", str(e))
    traceback.print_exc()
  sys.stdout.flush()

try:
  asyncio.run(main())
except KeyboardInterrupt:
  print("[INFO] Server shutdown via keyboard interrupt.")
except Exception as e:
  print("[ERROR] Server error:", str(e))
  traceback.print_exc()
sys.stdout.flush()