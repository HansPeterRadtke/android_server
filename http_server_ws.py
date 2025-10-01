import asyncio
import websockets
import traceback
import sys
import threading

clients = set()

async def handle_client(websocket):
  print("[INFO] WebSocket Connected")
  sys.stdout.flush()
  clients.add(websocket)

  try:
    async for message in websocket:
      print("[DEBUG] Received message of type:", type(message))
      sys.stdout.flush()
      if isinstance(message, bytes):
        try:
          print("[DEBUG] Echoing audio chunk, length:", len(message))
          sys.stdout.flush()
          await websocket.send(message)
        except Exception as e:
          print("[ERROR] Failed to echo audio chunk:", e)
          traceback.print_exc()
          sys.stdout.flush()
  except websockets.exceptions.ConnectionClosed as e:
    print("[INFO] Client disconnected:", e)
    sys.stdout.flush()
  finally:
    clients.discard(websocket)
    print("[DEBUG] Client removed from clients set")
    sys.stdout.flush()

async def main():
  print("[INFO] Server launching on port 8888")
  sys.stdout.flush()
  async with websockets.serve(handle_client, "0.0.0.0", 8888, max_size=(2**23), reuse_address=True):
    print("[INFO] Server started and accepting connections")
    sys.stdout.flush()
    await asyncio.Future()

def input_thread():
  while True:
    try:
      cmd = input().strip()
      if cmd == "exit":
        print("[INFO] Shutting down server via input thread")
        sys.stdout.flush()
        sys.exit(0)
      else:
        print(f"[DEBUG] Unknown command: {cmd}")
        sys.stdout.flush()
    except Exception as e:
      print("[ERROR] Input thread error:", e)
      sys.stdout.flush()

if __name__ == "__main__":
  threading.Thread(target=input_thread, daemon=True).start()
  asyncio.run(main())