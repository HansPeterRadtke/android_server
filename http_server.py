import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

class MyHandler(http.server.BaseHTTPRequestHandler):
  def do_GET(self):
    parsed_path = urlparse(self.path)
    query = parse_qs(parsed_path.query)
    print("[DEBUG] Path:", self.path)
    print("[DEBUG] Parsed query:", query)
    print("[DEBUG] Keys:", list(query.keys()))

    if "msg" in query:
      msg = query["msg"][0]
      reversed_msg = msg[::-1]
      print("[INFO] Reverting message:", msg, "->", reversed_msg)
      self.send_response(200)
      self.send_header("Content-type", "text/plain")
      self.end_headers()
      self.wfile.write(reversed_msg.encode("utf-8"))

    elif "size" in query:
      try:
        size = int(query["size"][0])
        print("[INFO] Bandwidth test: sending", size, "bytes")
        self.send_response(200)
        self.send_header("Content-type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        chunk = b"x" * 8192
        sent = 0
        while sent < size:
          remaining = size - sent
          to_send = chunk if remaining >= len(chunk) else b"x" * remaining
          self.wfile.write(to_send)
          self.wfile.flush()
          sent += len(to_send)
          print("[DEBUG] Sent chunk up to byte", sent)
        print("[INFO] Done sending", sent, "bytes")
      except Exception as e:
        print("[ERROR] Exception during size send:", e)
        self.send_error(500, "Internal server error")
    else:
      print("[WARN] No recognized query parameter. Query:", query)
      self.send_error(404, "Invalid query")

class ReusableTCPServer(socketserver.TCPServer):
  allow_reuse_address = True

PORT = 8888
httpd = ReusableTCPServer(("", PORT), MyHandler)
print("[INFO] Server started on port", PORT)
try:
  httpd.serve_forever()
except KeyboardInterrupt:
  print("[INFO] Server stopped")
  httpd.server_close()