#!/usr/bin/env bash
set -e

echo "🤖 Starting OSS Assistant (Qwen2.5-0.5B-Instruct)"
echo "   Port: 7860"
echo ""

if [ -z "$HF_TOKEN" ]; then
  echo "⚠️  HF_TOKEN not set. Some models may require authentication."
  echo "   Get a free token at: https://huggingface.co/settings/tokens"
  echo ""
fi

cd "$(dirname "$0")/../oss_assistant"
pip install -q -r requirements.txt
python app.py
