#!/usr/bin/env python3
"""
Obelysk — Multi-Turn Conversation Generator

Loads a HuggingFace model (e.g. Qwen3-14B) via transformers, runs multi-turn
conversation inference, and writes a conversation.json that prove-model can
consume for ZK proof capture.

Usage:
    python3 generate_conversation.py \
        --model-dir /root/.obelysk/models/qwen3-14b \
        --topic "Explain quantum computing" \
        --turns 3 \
        --output /tmp/conversation.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages():
    """Auto-install transformers/torch/accelerate if missing."""
    required = ["torch", "transformers", "accelerate"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return

    print(f"[INFO] Installing missing packages: {', '.join(missing)}", file=sys.stderr)
    cmd = [sys.executable, "-m", "pip", "install"] + missing
    try:
        subprocess.check_call(cmd, stdout=sys.stderr, stderr=sys.stderr)
    except subprocess.CalledProcessError:
        # PEP 668 fallback: --break-system-packages
        print("[WARN] pip install failed, retrying with --break-system-packages", file=sys.stderr)
        subprocess.check_call(cmd + ["--break-system-packages"], stdout=sys.stderr, stderr=sys.stderr)


ensure_packages()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_follow_up(tokenizer, model, conversation_history, device):
    """Generate a follow-up question based on conversation so far."""
    follow_up_messages = conversation_history + [
        {
            "role": "user",
            "content": (
                "Based on our conversation so far, ask one specific follow-up "
                "question that digs deeper into an important aspect we haven't "
                "fully explored. Just ask the question, nothing else."
            ),
        }
    ]
    text = tokenizer.apply_chat_template(
        follow_up_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=128, temperature=0.7, do_sample=True
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    question = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    # Take only the first sentence/question if model outputs extra
    for sep in ["?", "\n"]:
        idx = question.find(sep)
        if idx > 0:
            question = question[: idx + 1]
            break
    return question


def generate_response(tokenizer, model, messages, device, temperature, max_tokens):
    """Generate a model response for the given messages."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    full_context_tokens = inputs["input_ids"][0].tolist()
    last_token_id = full_context_tokens[-1] if full_context_tokens else 0

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
        )
    gen_time_ms = int((time.time() - t0) * 1000)

    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    response_tokens = new_ids.tolist()
    response_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    return {
        "full_context_tokens": full_context_tokens,
        "last_token_id": last_token_id,
        "response_text": response_text,
        "response_tokens": response_tokens,
        "generation_time_ms": gen_time_ms,
    }


def build_first_question(topic):
    """Build the first user question from the topic."""
    topic_lower = topic.lower().strip()
    # If the topic is already phrased as a question, use it directly
    if topic_lower.endswith("?") or topic_lower.startswith(("what ", "how ", "why ", "when ", "where ", "who ", "explain ", "describe ", "derive ")):
        return topic
    return f"What is {topic} and why is it important?"


def main():
    parser = argparse.ArgumentParser(description="Generate multi-turn conversation for ZK proving")
    parser.add_argument("--model-dir", required=True, type=Path, help="HuggingFace model directory")
    parser.add_argument("--topic", required=True, help="Conversation topic")
    parser.add_argument("--turns", type=int, default=3, help="Number of Q&A turns (default: 3)")
    parser.add_argument("--output", required=True, type=Path, help="Output conversation.json path")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens per response")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        print(f"[ERR] Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    # Detect model name from config.json
    config_path = model_dir / "config.json"
    model_name = model_dir.name
    if config_path.is_file():
        with open(config_path) as f:
            cfg = json.load(f)
            model_name = cfg.get("_name_or_path", model_name)

    print(f"[INFO] Loading model: {model_name} from {model_dir}", file=sys.stderr)
    print(f"[INFO] Topic: {args.topic}", file=sys.stderr)
    print(f"[INFO] Turns: {args.turns}", file=sys.stderr)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[INFO] Loading model to {device} ({dtype})...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    print("[INFO] Model loaded.", file=sys.stderr)

    # Generate conversation ID
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:4]
    conversation_id = f"conv_{ts}_{short_id}"

    system_message = {
        "role": "system",
        "content": "You are a knowledgeable assistant. Give clear, detailed answers.",
    }
    conversation_history = [system_message]
    turns = []

    for i in range(args.turns):
        # Build user question
        if i == 0:
            user_question = build_first_question(args.topic)
        else:
            user_question = generate_follow_up(
                tokenizer, model, conversation_history, device
            )

        print(f"[TURN {i}] Q: {user_question[:80]}{'...' if len(user_question) > 80 else ''}", file=sys.stderr)

        # Add user message and generate response
        conversation_history.append({"role": "user", "content": user_question})
        result = generate_response(
            tokenizer, model, conversation_history,
            device, args.temperature, args.max_tokens,
        )

        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": result["response_text"]})

        resp_len = len(result["response_tokens"])
        print(
            f"[TURN {i}] A: {resp_len} tokens ({result['generation_time_ms']}ms)",
            file=sys.stderr,
        )

        turns.append({
            "turn_index": i,
            "role": "user",
            "content": user_question,
            "full_context_tokens": result["full_context_tokens"],
            "last_token_id": result["last_token_id"],
            "response": {
                "content": result["response_text"],
                "tokens": result["response_tokens"],
                "generation_time_ms": result["generation_time_ms"],
            },
        })

    # Build output JSON
    conversation = {
        "version": "1",
        "conversation_id": conversation_id,
        "topic": args.topic,
        "model_name": model_name,
        "model_dir": str(model_dir),
        "turns": turns,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_turns": len(turns),
            "temperature": args.temperature,
            "max_new_tokens": args.max_tokens,
            "generator": "obelysk-converse/0.1.0",
        },
    }

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(conversation, f, indent=2)

    print(f"[OK] Saved: {args.output} ({len(turns)} turns, id={conversation_id})", file=sys.stderr)

    # Machine-readable output for shell parsing
    print(f"CONVERSATION_FILE={args.output}")
    print(f"CONVERSATION_ID={conversation_id}")
    print(f"CONVERSATION_TURNS={len(turns)}")


if __name__ == "__main__":
    main()
