"""
OSS Personal Assistant — Qwen2.5-0.5B-Instruct
Uses the HuggingFace Inference API (serverless) so no local GPU is needed.
Set HF_TOKEN environment variable before running.

Usage:
    python app.py
    # or: HF_TOKEN=hf_xxx python app.py
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import List, Tuple, Generator

import gradio as gr
from huggingface_hub import InferenceClient

from memory import ConversationMemory
from guardrails import check_input, check_output, REFUSAL_MESSAGE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID   = "Qwen/Qwen2.5-0.5B-Instruct"
HF_TOKEN   = os.getenv("HF_TOKEN", "")
MAX_TOKENS = 512
TEMPERATURE = 0.7
TOP_P       = 0.9
MAX_TURNS   = 10        # sliding-window memory

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful, harmless, and honest personal assistant.
You assist users with everyday tasks: answering questions, writing, analysis, brainstorming, and more.
You are concise and friendly. If you don't know something, say so clearly.
You never make up facts. You never produce harmful, illegal, or unethical content."""

# ---------------------------------------------------------------------------
# HuggingFace Inference client
# ---------------------------------------------------------------------------
client = InferenceClient(
    model=MODEL_ID,
    token=HF_TOKEN or None,
)

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def log_turn(user_msg: str, assistant_msg: str, latency_ms: float, flagged: bool) -> None:
    record = {
        "ts": time.time(),
        "model": MODEL_ID,
        "user": user_msg,
        "assistant": assistant_msg,
        "latency_ms": round(latency_ms, 1),
        "flagged": flagged,
    }
    log_file = LOG_DIR / "oss_log.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")

# ---------------------------------------------------------------------------
# Core chat function
# ---------------------------------------------------------------------------
def chat(
    user_message: str,
    history: List[Tuple[str, str]],
    memory: ConversationMemory,
) -> Tuple[str, List[Tuple[str, str]]]:
    """Single turn: returns (assistant_reply, updated_history)."""

    # 1. Input guardrail
    ok, refusal = check_input(user_message)
    if not ok:
        history = history + [(user_message, refusal)]
        log_turn(user_message, refusal, 0.0, flagged=True)
        return refusal, history

    # 2. Add user turn to memory
    memory.add_user(user_message)
    messages = memory.to_messages()

    # 3. Call HF Inference API
    t0 = time.perf_counter()
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        assistant_reply = response.choices[0].message.content
    except Exception as e:
        logger.error("Inference error: %s", e)
        assistant_reply = f"Sorry, I encountered an error: {e}"

    latency_ms = (time.perf_counter() - t0) * 1000

    # 4. Output guardrail
    assistant_reply = check_output(assistant_reply)

    # 5. Update memory & history
    memory.add_assistant(assistant_reply)
    history = history + [(user_message, assistant_reply)]

    log_turn(user_message, assistant_reply, latency_ms, flagged=False)
    logger.info("OSS | %.0fms | user=%s...", latency_ms, user_message[:40])

    return assistant_reply, history

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    memory = ConversationMemory(max_turns=MAX_TURNS, system_prompt=SYSTEM_PROMPT)

    with gr.Blocks(
        title="OSS Assistant — Qwen2.5",
        theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="teal",
            font=gr.themes.GoogleFont("JetBrains Mono"),
        ),
        css="""
        #chatbot { height: 520px; }
        .gr-button-primary { background: #059669 !important; }
        footer { display: none !important; }
        """,
    ) as demo:

        gr.Markdown(
            """
            # 🤖 OSS Personal Assistant
            **Model:** `Qwen2.5-0.5B-Instruct` · HuggingFace Inference API  
            **Memory:** Sliding window (last 10 turns) · **Guardrails:** Rule-based input/output filter
            """
        )

        chatbot = gr.Chatbot(elem_id="chatbot", label="Conversation", bubble_full_width=False)
        state   = gr.State([])              # Gradio history list

        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Ask me anything…",
                show_label=False,
                scale=8,
                lines=1,
            )
            send_btn  = gr.Button("Send",  variant="primary", scale=1)
            clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        gr.Markdown(
            "<sub>⚠️ OSS model — responses may be less reliable than frontier models. "
            "Guardrails are rule-based and may not catch all unsafe content.</sub>"
        )

        # --- event handlers ---
        def respond(user_msg: str, history: list):
            if not user_msg.strip():
                return "", history, history
            reply, new_history = chat(user_msg, history, memory)
            return "", new_history, new_history

        def clear_chat():
            memory.clear()
            return [], []

        send_btn.click(respond,  [msg_box, state], [msg_box, chatbot, state])
        msg_box.submit(respond,  [msg_box, state], [msg_box, chatbot, state])
        clear_btn.click(clear_chat, outputs=[chatbot, state])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_port=7860, share=False)
