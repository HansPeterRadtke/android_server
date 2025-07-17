print      ("[BOOT] TOP OF SCRIPT REACHED")
import asyncio
import websockets
import json
import traceback
import sys

PORT = 8888

print      ("[BOOT] WebSocket server script loaded.")
sys       .stdout.flush()

async def handle_client(websocket):
  print      ("[INFO] Entered handle_client()")
  sys       .stdout.flush()
  print      ("[INFO] New client connected.")
  sys       .stdout.flush()
  try:
    async for message in websocket:
      print      ("[DEBUG] Received message:" , message)
      sys       .stdout.flush()
      try:
        data           = json.loads      (message)
        if  ("msg"  in data):
          msg         = data["msg"]
          reversed_msg= msg[::-1]
          print      ("[INFO] Reversing message:", msg , "->" , reversed_msg)
          sys       .stdout.flush()
          await websocket.send(json.dumps({"result": reversed_msg}))
          print      ("[DEBUG] Sent reversed message.")
          sys       .stdout.flush()

        elif("size" in data):
          size      = int(data["size"])
          print      ("[INFO] Starting bandwidth test for" , size , "bytes")
          sys       .stdout.flush()

          received  = 0
          while(received < size):
            chunk   = await websocket.recv()
            received+= len(chunk)
            print    (f"[DEBUG] Received {len(chunk)} bytes, total: {received}")
            sys     .stdout.flush()

          chunk     = b"x" * 8192
          sent      = 0
          while(sent < size):
            remaining= size - sent
            to_send = chunk if remaining >= len(chunk) else b"x" * remaining
            await websocket.send(to_send)
            sent   += len(to_send)
            print  (f"[DEBUG] Sent {len(to_send)} bytes, total: {sent}")
            sys     .stdout.flush()

          print      ("[INFO] Bandwidth test completed.")
          sys       .stdout.flush()

        else:
          print      ("[WARN] Unknown command received.")
          await websocket.send(json.dumps({"error": "Unknown command"}))
          sys       .stdout.flush()

      except Exception as e:
        print      ("[ERROR] Failed processing message:", str(e))
        traceback  .print_exc()
        sys       .stdout.flush()

  except websockets.exceptions.ConnectionClosedError as e:
    print      ("[INFO] Client disconnected with error:" , e)
  except websockets.exceptions.ConnectionClosedOK as e:
    print      ("[INFO] Client disconnected cleanly:" , e)
  except asyncio.CancelledError:
    print      ("[INFO] handle_client cancelled.")
  except Exception as e:
    print      ("[ERROR] Connection handler exception:", str(e))
    traceback  .print_exc()
  sys       .stdout.flush()

async def input_monitor(shutdown_event):
  print      ("[BOOT] Input monitor started. Type 'exit', 'quit', or 'q' to stop.")
  sys       .stdout.flush()
  loop       = asyncio.get_running_loop()
  while not shutdown_event.is_set():
    try:
      line   = await loop.run_in_executor(None, sys.stdin.readline)
      cmd    = line.strip().lower()
      if  (cmd in ("exit" , "quit" , "q")):
        print      ("[INFO] Shutdown command received.")
        sys       .stdout.flush()
        shutdown_event.set()
        break
      else:
        print      (f"[ERROR] Unknown input command: {cmd}")
        sys       .stdout.flush()
    except Exception as e:
      print      ("[ERROR] Input monitor exception:", str(e))
      traceback  .print_exc()
      sys       .stdout.flush()
    await asyncio.sleep(0.1)

async def main():
  print      ("[INFO] WebSocket server starting on port" , PORT)
  sys       .stdout.flush()
  shutdown_event = asyncio.Event()
  try:
    async with websockets.serve(handle_client , "" , PORT):
      print      ("[INFO] Server is listening for clients...")
      sys       .stdout.flush()
      await asyncio.gather(
        input_monitor    (shutdown_event),
        shutdown_event   .wait()
      )
      print      ("[INFO] Shutdown signal processed. Server stopping...")
      sys       .stdout.flush()
  except asyncio.CancelledError:
    print      ("[INFO] Server main task cancelled.")
  except Exception as e:
    print      ("[ERROR] Failed to start server:", str(e))
    traceback  .print_exc()
  sys       .stdout.flush()

try:
  asyncio     .run      (main())
except KeyboardInterrupt:
  print      ("[INFO] Server shutdown via keyboard interrupt.")
except Exception as e:
  print      ("[ERROR] Server error:" , str(e))
  traceback  .print_exc()
sys       .stdout.flush()