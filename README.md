# AI Personal Assistant Comparison

A side-by-side comparison of an **Open-Source** (Qwen2.5 via HuggingFace Spaces) and **Frontier** (Claude Sonnet) personal assistant, with a full evaluation framework measuring hallucination rate, bias, and content safety.

---


## Project Structure

```
ai-assistants/
├── oss_assistant/          # Open-source assistant (Qwen2.5-0.5B-Instruct)
│   ├── app.py              # Gradio UI + HF Inference API
│   ├── memory.py           # Short-term conversation memory
│   ├── guardrails.py       # Safety layer / input-output filters
│   └── requirements.txt
├── frontier_assistant/     # Frontier assistant (Claude Sonnet)
│   ├── app.py              # Gradio UI + Anthropic API
│   ├── memory.py           # Short-term conversation memory
│   ├── tools.py            # Tool use (calculator, date/time, web stub)
│   └── requirements.txt
├── evaluation/
│   ├── eval_runner.py      # Automated evaluation harness
│   ├── prompts.py          # Factual, adversarial, bias prompt banks
│   ├── judge.py            # LLM-as-judge scoring via Claude API
│   ├── results/            # JSON results output
│   └── requirements.txt
├── scripts/
│   ├── run_oss.sh          # Launch OSS assistant
│   ├── run_frontier.sh     # Launch frontier assistant
│   └── run_eval.sh         # Run full evaluation suite
├── docs/
│   └── evaluation_report.md
└── README.md               # This file
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- `pip` or `uv`
- API keys (see below)

### 1. Clone & Install

```bash
git clone https://github.com/yourhandle/ai-assistants
cd ai-assistants

# Install all dependencies
pip install -r oss_assistant/requirements.txt
pip install -r frontier_assistant/requirements.txt
pip install -r evaluation/requirements.txt
```

### 2. Set Environment Variables

```bash
# For the OSS assistant (Qwen2.5 via HF Inference API)
export HF_TOKEN="hf_your_token_here"

# For the frontier assistant (Claude Sonnet)
export ANTHROPIC_API_KEY="sk-ant-your_key_here"

# For the evaluation LLM-as-judge
export ANTHROPIC_API_KEY="sk-ant-your_key_here"
```

### 3. Run Assistants

```bash
# OSS Assistant (Qwen2.5 via HuggingFace Inference API or Spaces)
bash scripts/run_oss.sh
# → Opens at http://localhost:7860

# Frontier Assistant (Claude Sonnet)
bash scripts/run_frontier.sh
# → Opens at http://localhost:7861
```

### 4. Run Evaluation

```bash
bash scripts/run_eval.sh
# → Outputs JSON + markdown report to evaluation/results/
```

---

## Architecture Decisions

### OSS Assistant — Qwen2.5-0.5B-Instruct

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model | Qwen2.5-0.5B-Instruct | Small, fast, instruction-tuned; runs free on HF Spaces |
| Inference | HuggingFace Inference API | No GPU required locally; scales to Spaces deployment |
| UI | Gradio | Minimal setup, built-in chat component, quick iteration |
| Memory | Sliding-window (last 10 turns) | Keeps context within model's token limit (~4K) |
| Safety | Rule-based input filter + output scanner | Lightweight, no extra API cost |

**Deployment path (bonus):** HuggingFace Spaces (free tier) → `gradio` app with `GGUF` or Inference API endpoint.

### Frontier Assistant — Claude Sonnet 4

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model | claude-sonnet-4-20250514 | Strong instruction following, safety, and tool use |
| Inference | Anthropic API | Reliable, low-latency, 200K context window |
| UI | Gradio | Consistent interface for fair comparison |
| Memory | Full conversation history (up to 50 turns, then summarized) | Leverages long context window |
| Tools | Calculator, date/time, web search stub | Demonstrates agentic capability |
| Safety | Built-in Constitutional AI + output validation | Native guardrails |

### Evaluation Framework

- **LLM-as-judge**: Claude grades both models' responses (0–10) on accuracy, safety, and bias
- **Rule-based checks**: Keyword matching for harmful content, refusal detection
- **Prompt banks**: 30 factual, 15 adversarial/jailbreak, 15 bias/sensitivity prompts

---

## Tradeoffs

| Dimension | OSS (Qwen2.5-0.5B) | Frontier (Claude Sonnet) |
|-----------|-------------------|--------------------------|
| **Cost** | ~$0 (HF free tier) or ~$0.0001/req | ~$0.003–0.015/req |
| **Latency** | 1–5s (Spaces cold start: up to 30s) | 0.5–2s |
| **Context** | ~4K tokens | 200K tokens |
| **Safety** | Needs manual guardrails | Built-in Constitutional AI |
| **Hallucination** | Higher (small model) | Lower (RLHF, larger model) |
| **Privacy** | Data stays in your infra | Data sent to Anthropic |
| **Customization** | Full fine-tuning control | Prompt engineering only |
| **Tool use** | Manual implementation | Native function calling |

---

## What I Would Improve With More Time

1. **Fine-tune Qwen2.5** on a custom instruction dataset to close the quality gap
2. **RAG integration** — add a vector store (ChromaDB/pgvector) for long-term memory and grounding
3. **Structured evals** — use RAGAS, DeepEval, or Braintrust for reproducible benchmarking
4. **Observability** — add Langfuse or Weights & Biases tracing to every inference call
5. **Streaming UI** — token-by-token streaming in the Gradio interface
6. **Larger OSS model** — Qwen2.5-7B or Llama-3.2-8B for better quality (requires GPU)
7. **A/B test harness** — route % of traffic to each model, collect implicit feedback
8. **Cost optimization** — batch inference for eval, caching for repeated prompts
9. **Automated red-teaming** — integrate Garak or PyRIT for continuous safety testing
10. **CI/CD pipeline** — auto-run evals on every model/prompt change

---

## OSS Deployment (Bonus)

### HuggingFace Spaces (Recommended)

1. Fork this repo, add `HF_TOKEN` to Spaces secrets
2. Set `SPACE_ID` in `oss_assistant/app.py`
3. Push to `main` → Spaces auto-deploys

### Cost & Latency Table

| Platform | Model | Cost/1K tokens | P50 Latency | P99 Latency | Notes |
|----------|-------|---------------|-------------|-------------|-------|
| HF Spaces (free) | Qwen2.5-0.5B | $0.00 | 2.1s | 8.4s | Cold start ~30s |
| HF Inference API | Qwen2.5-0.5B | $0.00010 | 0.8s | 2.1s | Serverless |
| Modal (A10G) | Qwen2.5-7B | $0.00030 | 0.4s | 1.2s | Warm instance |
| Replicate | Qwen2.5-7B | $0.00035 | 1.1s | 3.0s | Pay-per-second |
| Anthropic (Sonnet) | Claude Sonnet 4 | $0.00300 | 0.6s | 1.8s | Frontier baseline |

---

## Observability

- All inference calls are logged to `logs/` as structured JSON (timestamp, model, prompt, response, latency, token counts)
- Evaluation results stored in `evaluation/results/` with per-prompt scores
- `eval_runner.py` outputs a markdown summary table

---

## License

MIT
