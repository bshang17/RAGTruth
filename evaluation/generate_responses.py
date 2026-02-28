#!/usr/bin/env python3
"""Generate responses from a local HuggingFace model,
using the same prompts from the RAGTruth test set."""

import argparse
import json
import os
import signal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from prompts import load_source_map, load_responses

_cancel = False


def _handle_signal(sig, frame):
    global _cancel
    _cancel = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-info", default="../dataset/source_info.jsonl")
    p.add_argument("--responses", default="../dataset/response.jsonl")
    p.add_argument("--local-model", required=True,
                    help="Path to a local HF model directory")
    p.add_argument("--output", default="output/generated_responses.jsonl")
    p.add_argument("--model-name", default="llama2-7b-unlearned",
                    help="Name tag for this model in output records")
    p.add_argument("--filter-model", default=None,
                    help="Only generate for source_ids that originally used this model")
    p.add_argument("--temperatures", default="0.7",
                    help="Comma-separated temperatures")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-source-ids", type=int, default=None,
                    help="Limit to first N source_ids (for quick testing)")
    p.add_argument("--split", default="test")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def generate_one(model, tokenizer, source_id, prompt, task_type, model_name,
                  temperature, args):
    """Generate a single response using a locally-loaded HF model."""
    B_INST, E_INST = "[INST]", "[/INST]"
    input_prompt = f"{B_INST} {prompt.strip()} {E_INST}"

    inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            top_k=40,
        )
    # Decode only the new tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "source_id": source_id,
        "task_type": task_type,
        "model": model_name,
        "temperature": temperature,
        "response": answer.strip(),
        "prompt": prompt,
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    source_map = load_source_map(args.source_info)
    responses = load_responses(args.responses, split=args.split,
                                filter_model=args.filter_model)

    # Collect unique source_ids from the test set
    source_ids = sorted(set(r["source_id"] for r in responses))
    if args.max_source_ids:
        source_ids = source_ids[:args.max_source_ids]
    temperatures = [float(t) for t in args.temperatures.split(",")]

    # Resume support
    done_keys = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                d = json.loads(line)
                done_keys.add((d["source_id"], d["temperature"]))

    tasks_to_run = []
    for sid in source_ids:
        for temp in temperatures:
            if (sid, temp) not in done_keys:
                tasks_to_run.append((sid, temp))

    print(f"Total source_ids: {len(source_ids)}, temperatures: {temperatures}")
    print(f"Tasks to run: {len(tasks_to_run)} ({len(done_keys)} already done)")

    # Load model
    print(f"Loading model from {args.local_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.local_model)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.local_model, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {model.device} ({dtype})")

    # Generate
    results = []
    pbar = tqdm(total=len(tasks_to_run), desc="Generating")
    for sid, temp in tasks_to_run:
        if _cancel:
            break
        si = source_map[sid]
        try:
            r = generate_one(model, tokenizer, sid, si["prompt"],
                              si["task_type"], args.model_name, temp, args)
            results.append(r)
        except Exception as e:
            results.append({
                "source_id": sid, "task_type": si["task_type"],
                "model": args.model_name, "temperature": temp,
                "response": None, "prompt": si["prompt"], "error": str(e),
            })
        pbar.update(1)
    pbar.close()

    # Write results
    mode = "a" if args.resume and done_keys else "w"
    with open(args.output, mode) as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    ok = sum(1 for r in results if r.get("response"))
    err = sum(1 for r in results if r.get("error"))
    print(f"Done. Success: {ok}, Errors: {err}")


if __name__ == "__main__":
    main()
