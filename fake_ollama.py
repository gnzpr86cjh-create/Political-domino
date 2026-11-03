"""A tiny stand-in for the Ollama HTTP API, used ONLY by the test-suite.

It lets the tests exercise the real client, streaming, JSON parsing, retrieval,
validation and storage code paths without a GPU or a model download. The
responses it returns are written by each test; the application itself never
contains canned analytical content.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class FakeOllama:
    def __init__(self, models=("qwen3:4b",), responder=None, chat_status=200):
        self.models = list(models)
        self.responder = responder          # callable(stage, payload) -> str
        self.chat_status = chat_status
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/api/tags":
                    body = json.dumps({"models": [{"name": m} for m in outer.models]}).encode()
                    self.send_response(200); self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                else:
                    self.send_response(404); self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(payload)
                if outer.chat_status != 200:
                    body = b'{"error":"boom"}'
                    self.send_response(outer.chat_status); self.send_header("Content-Length", str(len(body)))
                    self.end_headers(); self.wfile.write(body); return
                system = payload["messages"][0]["content"]
                stage = "map" if "extract the causal structure" in system else "gap"
                content = outer.responder(stage, payload)
                self.send_response(200); self.send_header("Content-Type", "application/x-ndjson"); self.end_headers()
                step = max(1, len(content) // 3)
                for i in range(0, len(content), step):
                    line = {"message": {"role": "assistant", "content": content[i:i + step]}, "done": False}
                    self.wfile.write((json.dumps(line) + "\n").encode()); self.wfile.flush()
                self.wfile.write((json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop"}) + "\n").encode())

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
