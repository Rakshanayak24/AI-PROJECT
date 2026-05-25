#!/usr/bin/env bash
set -e

echo "📊 Running AI Assistant Evaluation Suite"
echo ""

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "❌ ANTHROPIC_API_KEY required for LLM-as-judge + Frontier model."
  exit 1
fi

if [ -z "$HF_TOKEN" ]; then
  echo "⚠️  HF_TOKEN not set — OSS model calls may fail. Set it or use --skip-oss"
fi

cd "$(dirname "$0")/../evaluation"
pip install -q -r requirements.txt

# Run full evaluation (all 60 prompts)
# Add --limit 5 for a quick smoke test
# Add --skip-oss if you don't have HF_TOKEN
python eval_runner.py "$@"

echo ""
echo "✅ Results saved to evaluation/results/"
