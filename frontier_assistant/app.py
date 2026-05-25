"""
Frontier Personal Assistant — Claude Sonnet 4
Uses the Anthropic API with tool use and long-context memory.
Set ANTHROPIC_API_KEY environment variable before running.

Usage:
    python app.py
    # or: ANTHROPIC_API_KEY=sk-ant-xxx python app.py
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import List, Tuple

import anthropic
import gradio as gr

from memory import ConversationMemory
from tools import TOOLS, execute_tool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID      = "claude-sonnet-4-20250514"
MAX_TOKENS    = 1024
TEMPERATURE   = 0.7
MAX_PAIRS     = 50

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful, honest, and thoughtful personal assistant powered by Claude.
You assist users with everyday tasks: answering questions, writing, analysis, brainstorming, coding help, and more.
You have access to tools: a calculator, current date/time lookup, and a web search stub.
Use tools when they would provide a better answer (e.g. use the calculator for arithmetic, not mental math).
Be concise but thorough. If you're uncertain, say so.
Never fabricate facts, URLs, or citations."""

# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------
api_key = os.getenv("ANTHROPIC_API_KEY", "")
client  = anthropic.Anthropic(api_key=api_key) if api_key else None

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def log_turn(user_msg: str, assistant_msg: str, latency_ms: float, tools_used: list) -> None:
    record = {
        "ts": time.time(),
        "model": MODEL_ID,
        "user": user_msg,
        "assistant": assistant_msg,
        "latency_ms": round(latency_ms, 1),
        "tools_used": tools_used,
    }
    log_file = LOG_DIR / "frontier_log.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")

# ---------------------------------------------------------------------------
# Core chat function (with agentic tool loop)
# ---------------------------------------------------------------------------
def chat(
    user_message: str,
    history: List[Tuple[str, str]],
    memory: ConversationMemory,
) -> Tuple[str, List[Tuple[str, str]]]:

    if not client:
        error = "⚠️ ANTHROPIC_API_KEY not set. Please set it and restart."
        history = history + [(user_message, error)]
        return error, history

    memory.add_user(user_message)
    messages = memory.to_messages()
    system   = memory.get_system()

    t0 = time.perf_counter()
    tools_used = []
    final_reply = ""

    try:
        # Agentic loop: keep calling until no more tool_use blocks
        while True:
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=system,
                tools=TOOLS,
                messages=messages,
            )

            # Collect text blocks
            text_parts = [b.text for b in response.content if b.type == "text"]
            tool_uses  = [b for b in response.content if b.type == "tool_use"]

            if not tool_uses:
                # Final answer — no more tool calls
                final_reply = " ".join(text_parts).strip()
                break

            # Handle tool calls
            tool_results = []
            for tu in tool_uses:
                result = execute_tool(tu.name, tu.input)
                tools_used.append(tu.name)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result,
                })

            # Append assistant message + tool results to messages list
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            # If there was text before the tool call, it's preamble; ignore for now
            # (we'll use the final answer after all tools are done)

    except anthropic.APIError as e:
        final_reply = f"API error: {e}"
    except Exception as e:
        logger.exception("Unexpected error")
        final_reply = f"Unexpected error: {e}"

    latency_ms = (time.perf_counter() - t0) * 1000

    if tools_used:
        tool_note = f"\n\n<sub>🔧 Tools used: {', '.join(tools_used)}</sub>"
        final_reply += tool_note

    memory.add_assistant(final_reply)
    history = history + [(user_message, final_reply)]

    log_turn(user_message, final_reply, latency_ms, tools_used)
    logger.info("Frontier | %.0fms | tools=%s | user=%s…", latency_ms, tools_used, user_message[:40])

    return final_reply, history

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    memory = ConversationMemory(max_pairs=MAX_PAIRS, system_prompt=SYSTEM_PROMPT)

    with gr.Blocks(
        title="Frontier Assistant — Claude Sonnet",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="purple",
            font=gr.themes.GoogleFont("IBM Plex Mono"),
        ),
        css="""
        #chatbot { height: 520px; }
        .gr-button-primary { background: #7c3aed !important; }
        footer { display: none !important; }
        """,
    ) as demo:

        gr.Markdown(
            """
            # 🚀 Frontier Personal Assistant
            **Model:** `claude-sonnet-4-20250514` · Anthropic API  
            **Memory:** Full history (50 turns, with auto-summarization) · **Tools:** Calculator · DateTime · Web search
            """
        )

        chatbot   = gr.Chatbot(elem_id="chatbot", label="Conversation", bubble_full_width=False)
        state     = gr.State([])

        with gr.Row():
            msg_box   = gr.Textbox(
                placeholder="Ask me anything — I can do math, remember our conversation, and more…",
                show_label=False,
                scale=8,
                lines=1,
            )
            send_btn  = gr.Button("Send",  variant="primary", scale=1)
            clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        gr.Markdown(
            "<sub>✅ Powered by Constitutional AI — built-in safety, honesty, and helpfulness.</sub>"
        )

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
    app.launch(server_port=7861, share=False)
