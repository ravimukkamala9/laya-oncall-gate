"""A tiny fake Ollama server for tests/screenshots (NOT a real LLM): answers with the keyword stand-in's
numbers as JSON, with a delay, so the plumbing (HTTP, schema, retry, metrics) can be tested without Ollama."""
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from oncall_gate.engine import DemoEngine


class H(BaseHTTPRequestHandler):
    delay = 0.15
    broken_every = 0            # set >0 to return malformed JSON for every Nth call (tests the retry/error path)
    calls = 0

    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"models": [{"name": "llama3.2:latest"}, {"name": "qwen2.5:7b"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        H.calls += 1
        time.sleep(self.delay)
        text = re.search(r"\n  text: (.*)\n", body["messages"][1]["content"]).group(1)
        a = DemoEngine().predict({"alert": text})["answers"]
        out = {"category": a["category"]["choice"], "customer_impact": a["customer_impact"]["noul"],
               "self_recovering": a["self_recovering"]["noul"], "sustained": a["sustained"]["noul"],
               "business_risk": a["business_risk"]["noul"], "severity": a["severity"]["score"]}
        content = json.dumps(out)
        if H.broken_every and H.calls % H.broken_every == 0:
            content = "Sure! Here is the JSON: {oops"
        self._send({"message": {"role": "assistant", "content": content}, "total_duration": int(self.delay * 1e9),
                    "load_duration": 0, "prompt_eval_count": 310, "eval_count": 58, "eval_duration": int(self.delay * 0.8e9)})


def start(port=11999):
    srv = HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
