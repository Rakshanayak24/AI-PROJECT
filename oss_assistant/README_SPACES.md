---
title: OSS Personal Assistant (Qwen2.5)
emoji: 🤖
colorFrom: cyan
colorTo: teal
sdk: gradio
sdk_version: "4.40.0"
app_file: app.py
pinned: false
license: mit
---

# OSS Personal Assistant — Qwen2.5-0.5B-Instruct

A personal assistant built on the open-source Qwen2.5-0.5B-Instruct model.

## Features
- Multi-turn conversation with sliding-window memory
- Rule-based guardrails (input filter + output scanner)
- Structured logging for observability

## Setup (Local)
```bash
pip install -r requirements.txt
HF_TOKEN=your_token python app.py
```

## Deployment on HF Spaces
1. Fork this repo to your HuggingFace account
2. Set `HF_TOKEN` in your Space's Secrets (Settings → Repository Secrets)
3. Push to `main` — Spaces will auto-build and deploy

## Architecture
- **Model**: Qwen/Qwen2.5-0.5B-Instruct via HF Inference API
- **UI**: Gradio
- **Memory**: Sliding window (10 turns)
- **Safety**: Rule-based input/output filtering
