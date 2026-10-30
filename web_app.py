#!/usr/bin/env python3
"""
TACD & PISA-1200: Interactive Browser Application.

A lightweight, zero-dependency browser interface built with Python standard library.
Enables visual inspection of scene topology graphs and real-time contrastive decoding.
"""

from __future__ import annotations
import os
import sys
import json
import base64
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from core.extractor import StructuralTopologyExtractor
from core.decoder import TACDDecoder, TACDConfig


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TACD & PISA-1200: Mitigating VLM Hallucination</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.2);
      --text: #f8fafc;
      --muted: #94a3b8;
      --danger: #f87171;
      --success: #4ade80;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      text-align: center;
      margin-bottom: 32px;
      padding: 24px;
      background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
      border-radius: 16px;
      border: 1px solid var(--border);
    }
    h1 { font-size: 1.8rem; font-weight: 700; color: #fff; margin-bottom: 8px; }
    .badge {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 0.8rem;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid var(--accent);
      margin-bottom: 12px;
    }
    .subtitle { color: var(--muted); font-size: 0.95rem; max-width: 800px; margin: 0 auto; }
    
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    
    .panel {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
    }
    .panel h2 { font-size: 1.15rem; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
    
    .image-preview {
      width: 100%;
      height: 240px;
      border-radius: 8px;
      border: 2px dashed var(--border);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      margin-bottom: 16px;
      background: #0b1120;
    }
    .image-preview img { width: 100%; height: 100%; object-fit: contain; }
    
    .form-group { margin-bottom: 16px; }
    label { display: block; font-size: 0.85rem; color: var(--muted); margin-bottom: 6px; }
    input[type="text"], select {
      width: 100%;
      padding: 10px 14px;
      background: #0b1120;
      border: 1px solid var(--border);
      border-radius: 6px;
      color: #fff;
      font-size: 0.9rem;
    }
    .slider-row { display: flex; align-items: center; gap: 12px; }
    input[type="range"] { flex: 1; }
    .slider-val { font-weight: 600; width: 45px; text-align: right; color: var(--accent); }
    
    button.btn-primary {
      width: 100%;
      padding: 12px;
      background: #0284c7;
      color: white;
      border: none;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }
    button.btn-primary:hover { background: #0369a1; }
    
    .results-card {
      padding: 16px;
      border-radius: 8px;
      margin-bottom: 16px;
      border: 1px solid var(--border);
    }
    .results-card.baseline { background: rgba(248, 113, 113, 0.08); border-color: rgba(248, 113, 113, 0.3); }
    .results-card.tacd { background: rgba(74, 222, 128, 0.08); border-color: rgba(74, 222, 128, 0.3); }
    
    .card-title {
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
      font-weight: 700;
    }
    .card-title.baseline { color: var(--danger); }
    .card-title.tacd { color: var(--success); }
    
    .response-text { font-size: 1rem; color: #fff; margin-bottom: 8px; }
    .status-tag {
      font-size: 0.75rem;
      padding: 2px 8px;
      border-radius: 4px;
      font-weight: 600;
      display: inline-block;
    }
    .status-tag.danger { background: rgba(248, 113, 113, 0.2); color: var(--danger); }
    .status-tag.success { background: rgba(74, 222, 128, 0.2); color: var(--success); }
    
    .topology-box {
      font-family: monospace;
      font-size: 0.8rem;
      background: #0b1120;
      padding: 12px;
      border-radius: 6px;
      max-height: 180px;
      overflow-y: auto;
      border: 1px solid var(--border);
      color: #38bdf8;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge">Research Reference Demo</div>
      <h1>Beyond Pixel Grounding (PISA-1200 & TACD)</h1>
      <p class="subtitle">Topology-Aware Contrastive Decoding for mitigating cross-modal hallucination in Vision-Language Models.</p>
    </header>

    <div class="grid">
      <!-- Input Panel -->
      <div class="panel">
        <h2>📷 Input Image & Query</h2>
        <div class="image-preview">
          <img id="preview" src="/sample.jpg" alt="Sample Scene">
        </div>

        <div class="form-group">
          <label>Evaluation Query / Question:</label>
          <input type="text" id="query" value="Is there a computer mouse next to the laptop on the desk?">
        </div>

        <div class="form-group">
          <label>Contrastive Penalty (&alpha;):</label>
          <div class="slider-row">
            <input type="range" id="alpha" min="0.0" max="1.5" step="0.1" value="0.6" oninput="document.getElementById('alpha-val').innerText = this.value">
            <span class="slider-val" id="alpha-val">0.6</span>
          </div>
        </div>

        <div class="form-group">
          <label>Adaptive Truncation Cutoff (&beta;):</label>
          <div class="slider-row">
            <input type="range" id="beta" min="0.01" max="0.5" step="0.01" value="0.1" oninput="document.getElementById('beta-val').innerText = this.value">
            <span class="slider-val" id="beta-val">0.1</span>
          </div>
        </div>

        <button class="btn-primary" onclick="runInference()">Run TACD Contrastive Decoding</button>
      </div>

      <!-- Output Panel -->
      <div class="panel">
        <h2>📊 Decoding Comparison</h2>

        <div class="results-card baseline">
          <div class="card-title baseline">Baseline VLM (Ungrounded Language Prior)</div>
          <div class="response-text" id="base-response">Click "Run TACD" to evaluate...</div>
          <span class="status-tag danger" id="base-tag">Vulnerable to Hallucination</span>
        </div>

        <div class="results-card tacd">
          <div class="card-title tacd">TACD Mitigated (Topology-Grounded)</div>
          <div class="response-text" id="tacd-response">Click "Run TACD" to evaluate...</div>
          <span class="status-tag success" id="tacd-tag">Verified Grounded</span>
        </div>

        <h3 style="font-size: 0.95rem; margin-top: 20px; margin-bottom: 8px;">Verified Structural Topology Graph G = (O, R):</h3>
        <div class="topology-box" id="topology-view">Loading scene topology...</div>
      </div>
    </div>
  </div>

  <script>
    async function loadTopology() {
      const res = await fetch('/api/topology');
      const data = await res.json();
      document.getElementById('topology-view').innerText = data.topology;
    }

    async function runInference() {
      const query = document.getElementById('query').value;
      const alpha = parseFloat(document.getElementById('alpha').value);
      const beta = parseFloat(document.getElementById('beta').value);

      document.getElementById('base-response').innerText = "Generating baseline response...";
      document.getElementById('tacd-response').innerText = "Computing contrastive logits...";

      const res = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, alpha, beta })
      });
      const data = await res.json();

      document.getElementById('base-response').innerText = data.baseline.response;
      document.getElementById('base-tag').innerText = data.baseline.hallucination_detected ? "HALLUCINATION DETECTED" : "NO HALLUCINATION";

      document.getElementById('tacd-response').innerText = data.tacd.response;
      document.getElementById('tacd-tag').innerText = data.tacd.hallucination_detected ? "HALLUCINATION DETECTED" : "CLEAN / FACTUALLY GROUNDED";
    }

    loadTopology();
    runInference();
  </script>
</body>
</html>
"""


class TACDBrowserHandler(BaseHTTPRequestHandler):
    extractor = StructuralTopologyExtractor()
    decoder = TACDDecoder()
    sample_img_path = os.path.join(repo_root, "assets", "sample.jpg")

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif self.path == "/sample.jpg":
            if os.path.exists(self.sample_img_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                with open(self.sample_img_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Sample image not found")
        elif self.path == "/api/topology":
            graph = self.extractor.extract_from_image(self.sample_img_path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"topology": graph.to_string("canonical")}).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/compare":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)

            query = data.get("query", "Is there a computer mouse next to the laptop on the desk?")
            alpha = float(data.get("alpha", 0.6))
            beta = float(data.get("beta", 0.1))

            comparison = self.decoder.compare(
                image=self.sample_img_path,
                query=query,
                alpha=alpha,
                beta=beta,
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(comparison).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")


def run_server(port: int = 8000):
    server = HTTPServer(("localhost", port), TACDBrowserHandler)
    print(f"\n" + "=" * 70)
    print(f"  TACD Browser Web App running at: http://localhost:{port}")
    print(f"  Open http://localhost:{port} in your browser (Safari, Chrome, etc.)")
    print(f"  Press Ctrl+C to stop.")
    print("=" * 70 + "\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run TACD interactive browser app")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    args = parser.parse_args()
    run_server(port=args.port)
