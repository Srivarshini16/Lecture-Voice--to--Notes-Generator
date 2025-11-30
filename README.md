🎧 Lecture Notes AI — Local Inference Engine

Lecture Notes AI is a lightweight, high-performance system for transcribing and summarizing audio lectures locally. It includes a real-time telemetry dashboard for monitoring CPU usage, inference latency, and processing throughput.

🚀 Features

Local Whisper Inference (PyTorch, offline)

Telemetry Dashboard for CPU load & latency

Fast Summarization with fallback logic

Clean FastAPI Backend + Tailwind frontend

🛠️ Tech Stack

Backend: Python, FastAPI, Uvicorn

ML: PyTorch, OpenAI Whisper

Monitoring: psutil

Frontend: HTML5, Tailwind CSS

⚡ Quick Start
pip install -r requirements.txt
python app.py

Open: http://127.0.0.1:8000
