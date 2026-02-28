#!/usr/bin/env python3
"""GPT-based hallucination judge.

Sends (source context, response) pairs to GPT-4 and asks it to identify
hallucinated spans. Works on both generated and original responses."""

import argparse
import asyncio
import json
import logging
import os
import signal

from openai import (AsyncOpenAI, RateLimitError, APITimeoutError,
                     APIConnectionError, BadRequestError)
from tqdm import tqdm

from prompts import load_source_map, load_responses, load_generated, build_gpt_judge_messages

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_cancel = False


def _handle_signal(sig, frame):
    global _cancel
    _cancel = True
    logger.info("Shutting down gracefully...")


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                    help="Path to generated_responses.jsonl or dataset/response.jsonl")
    p.add_argument("--source-info", default="../dataset/source_info.jsonl")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--model", default="gpt-4o", help="GPT model to use")
    p.add_argument("--filter-model", default=None,
                    help="Only judge records from this model (for original responses)")
    p.add_argument("--split", default="test",
                    help="Split filter (only for original response.jsonl input)")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--is-generated", action="store_true",
                    help="Set if input is generated responses (not original dataset)")
    return p.parse_args()


async def judge_one(client, record, source_map, args, sem):
    """Call GPT to judge one response."""
    messages = build_gpt_judge_messages(record, source_map)

    async with sem:
        for attempt in range(args.max_retries):
            if _cancel:
                return None
            try:
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content.strip()
                try:
                    pred = json.loads(raw)
                except json.JSONDecodeError:
                    pred = {"hallucination list": [], "parse_error": raw}
                return pred
            except (RateLimitError, APITimeoutError, APIConnectionError):
                wait = min(2 ** attempt, 60)
                logger.warning(f"Retry {attempt+1}/{args.max_retries}, waiting {wait}s")
                await asyncio.sleep(wait)
            except BadRequestError as e:
                logger.error(f"BadRequest for {record.get('source_id', '?')}: {e}")
                return {"hallucination list": [], "error": str(e)}

    logger.error(f"Exhausted retries for {record.get('source_id', '?')}")
    return {"hallucination list": [], "error": "exhausted_retries"}


async def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    source_map = load_source_map(args.source_info)

    if args.is_generated:
        records = load_generated(args.input)
        # Filter out error records
        records = [r for r in records if r.get("response")]
    else:
        records = load_responses(args.input, split=args.split,
                                  filter_model=args.filter_model)

    # Resume support
    done_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                d = json.loads(line)
                done_ids.add(d.get("_key", d.get("source_id")))
        logger.info(f"Resuming: {len(done_ids)} already done")

    pending = []
    for r in records:
        key = f"{r['source_id']}_{r.get('model', '')}_{r.get('temperature', '')}"
        if key not in done_ids:
            r["_key"] = key
            pending.append(r)

    logger.info(f"Judging {len(pending)} records ({len(done_ids)} skipped)")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(pending), desc="Judging")

    mode = "a" if args.resume and done_ids else "w"
    with open(args.output, mode) as f:

        async def _judge_and_write(record):
            pred = await judge_one(client, record, source_map, args, sem)
            pbar.update(1)
            if pred is None:
                return

            si = source_map[record["source_id"]]
            result = {
                "_key": record["_key"],
                "source_id": record["source_id"],
                "model": record.get("model", ""),
                "task_type": si["task_type"],
                "judge": args.model,
                "pred": pred,
                "pred_halu": len(pred.get("hallucination list", [])) > 0,
            }
            # Include ground truth labels if available (original dataset)
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
