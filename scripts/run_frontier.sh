#!/usr/bin/env bash
set -e

echo "🚀 Starting Frontier Assistant (Claude Sonnet)"
echo "   Port: 7861"
echo ""

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "❌ ANTHROPIC_API_KEY not set."
  echo "   Get an API key at: https://console.anthropic.com"
  echo ""
  exit 1
fi

cd "$(dirname "$0")/../frontier_assistant"
pip install -q -r requirements.txt
python app.py
