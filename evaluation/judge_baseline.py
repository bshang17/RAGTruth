#!/usr/bin/env python3
"""Baseline detector hallucination judge.

Uses the trained RAGTruth hallucination detector (fine-tuned Llama-2 served
via TGI) to judge responses. Adapted from baseline/predict_and_evaluate.py."""

import argparse
import asyncio
import json
import logging
import os
import sys

from huggingface_hub import AsyncInferenceClient
from transformers import AutoTokenizer
from tqdm import tqdm

# Import process_dialog_to_single_turn from the baseline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline"))
from dataset import process_dialog_to_single_turn

from prompts import load_source_map, load_responses, load_generated, enrich_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

B_INST, E_INST = "[INST]", "[/INST]"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                    help="Path to generated_responses.jsonl or dataset/response.jsonl")
    p.add_argument("--source-info", default="../dataset/source_info.jsonl")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--tgi-url", default="http://127.0.0.1:8300")
    p.add_argument("--tokenizer", default="meta-llama/Llama-2-13b-hf")
    p.add_argument("--filter-model", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--concurrency", type=int, default=80)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--is-generated", action="store_true",
                    help="Set if input is generated responses (not original dataset)")
    return p.parse_args()


async def judge_one(client, data, tokenizer, sem, pbar):
    """Judge one record using the baseline detector via TGI."""
    input_prompt = process_dialog_to_single_turn(
        data, tokenizer, return_prompt=True, train=False
    )
    input_prompt = f"{B_INST} {input_prompt.strip()} {E_INST}"

    async with sem:
        for attempt in range(10):
            try:
                answer = await client.text_generation(
                    input_prompt,
                    max_new_tokens=512,
                    stream=False,
                    do_sample=True,
                    temperature=0.05,
                    top_p=0.95,
                    top_k=40,
                )
                answer = answer.strip()
                pred = json.loads(answer)
                pbar.update(1)
                return pred
            except Exception:
                if attempt == 9:
                    pbar.update(1)
                    return {"hallucination list": [], "error": "inference_failed"}
                await asyncio.sleep(min(2 ** attempt, 30))


async def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    source_map = load_source_map(args.source_info)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.is_generated:
        records = load_generated(args.input)
        records = [r for r in records if r.get("response")]
    else:
        records = load_responses(args.input, split=args.split,
                                  filter_model=args.filter_model)

    # Enrich records with reference/question/task_type
    for r in records:
        enrich_record(r, source_map)

    # Resume support
    done_keys = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                d = json.loads(line)
                done_keys.add(d.get("_key"))
        logger.info(f"Resuming: {len(done_keys)} already done")

    pending = []
    for r in records:
        key = f"{r['source_id']}_{r.get('model', '')}_{r.get('temperature', '')}"
        if key not in done_keys:
            r["_key"] = key
            pending.append(r)

    logger.info(f"Judging {len(pending)} records ({len(done_keys)} skipped)")

    client = AsyncInferenceClient(model=args.tgi_url, timeout=100)
    sem = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(pending), desc="Judging")

    mode = "a" if args.resume and done_keys else "w"
    with open(args.output, mode) as f:

        async def _judge_and_write(record):
            pred = await judge_one(client, record, tokenizer, sem, pbar)

            si = source_map[record["source_id"]]
            result = {
                "_key": record["_key"],
                "source_id": record["source_id"],
                "model": record.get("model", ""),
                "task_type": si["task_type"],
                "judge": "baseline_detector",
                "pred": pred,
                "pred_halu": len(pred.get("hallucination list", [])) > 0,
            }
            if "labels" in record:
                result["is_halu"] = len(record["labels"]) > 0
            f.write(json.dumps(result) + "\n")
            f.flush()

        tasks = [asyncio.create_task(_judge_and_write(r)) for r in pending]
        await asyncio.gather(*tasks)

    pbar.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
