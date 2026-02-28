#!/usr/bin/env python3
"""Remove Evident Baseless Info hallucination spans from RAGTruth responses
and use GPT to polish the transition points."""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from difflib import SequenceMatcher

from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError, BadRequestError
from tqdm import tqdm

try:
    from rapidfuzz.distance import Levenshtein
    def edit_distance(a, b):
        return Levenshtein.distance(a, b)
except ImportError:
    def edit_distance(a, b):
        return sum(1 for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b).get_opcodes()
                   if op != 'equal' for _ in range(max(i2 - i1, j2 - j1)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a text editor. You will receive a passage that has had some phrases \
removed, leaving awkward transitions. Your ONLY job is to minimally fix the \
grammar and flow at each removal junction. Rules:

1. Keep ALL existing words verbatim. Do not rephrase, reorder, or paraphrase.
2. You may ONLY: delete redundant punctuation (e.g. ", ,"), delete dangling \
conjunctions/prepositions left by removal, or add at most 1-2 small \
connecting words (articles, conjunctions, prepositions) to restore grammar.
3. Do not add new factual content. Do not elaborate.
4. If a sentence becomes empty or nonsensical after removal, delete it entirely.
5. Return ONLY the corrected passage, with no commentary or explanation."""

# Graceful shutdown
_cancel = False

def _handle_signal(sig, frame):
    global _cancel
    _cancel = True
    logger.info("Received signal, finishing in-flight tasks...")

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="dataset/response.jsonl")
    p.add_argument("--output-dir", default="cleaning/output")
    p.add_argument("--model", default="gpt-5.2-2025-12-11")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--edit-threshold", type=float, default=0.10)
    p.add_argument("--split", default="all", choices=["train", "test", "all"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Skip GPT calls, output naive removal only")
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def load_and_filter(input_path, split):
    """Load response.jsonl and filter to records with Evident Baseless Info."""
    records = []
    with open(input_path) as f:
        for line in f:
            rec = json.loads(line)
            if split != "all" and rec["split"] != split:
                continue
            ebi = [l for l in rec["labels"] if l["label_type"] == "Evident Baseless Info"]
            if ebi:
                records.append(rec)
    return records


def merge_intervals(spans):
    """Merge overlapping spans sorted by start."""
    if not spans:
        return []
    merged = [{"start": spans[0]["start"], "end": spans[0]["end"]}]
    for s in spans[1:]:
        if s["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append({"start": s["start"], "end": s["end"]})
    return merged


def naive_remove(response, merged_spans):
    """Remove spans and return naive text, marked text with [JUNCTION], and junction positions."""
    pieces_naive = []
    pieces_marked = []
    junction_positions = []
    prev_end = 0

    for span in merged_spans:
        segment = response[prev_end:span["start"]]
        pieces_naive.append(segment)
        pieces_marked.append(segment)
        if span["start"] > 0 and span["end"] < len(response):
            pieces_marked.append("[JUNCTION]")
            junction_positions.append(sum(len(p) for p in pieces_naive))
        prev_end = span["end"]

    tail = response[prev_end:]
    pieces_naive.append(tail)
    pieces_marked.append(tail)

    return "".join(pieces_naive), "".join(pieces_marked), junction_positions


def validate(naive_text, polished_text, junction_positions, threshold):
    """Validate GPT output. Returns (edit_dist, edit_ratio, flag)."""
    if not naive_text:
        return 0, 0.0, None

    dist = edit_distance(naive_text, polished_text)
    ratio = dist / len(naive_text)
    flag = None

    if ratio > threshold:
        flag = "excessive_edit"
    elif len(polished_text) > len(naive_text) + 20:
        flag = "content_added"
    else:
        # Check for edits far from junctions
        if junction_positions:
            opcodes = SequenceMatcher(None, naive_text, polished_text).get_opcodes()
            for op, i1, i2, j1, j2 in opcodes:
                if op == "equal":
                    continue
                mid = (i1 + i2) // 2
                if all(abs(mid - jp) > 30 for jp in junction_positions):
                    flag = "distant_edit"
                    break

    return dist, ratio, flag


async def polish_one(client, record_id, marked_text, n_junctions, args, sem):
    """Call GPT to polish one record."""
    user_msg = (
        f"The following passage has had {n_junctions} hallucinated span(s) removed. "
        "The removal points are marked with [JUNCTION] tokens. "
        "Please fix ONLY the grammar at each [JUNCTION] point following the rules above.\n\n"
        f"{marked_text}"
    )

    async with sem:
        for attempt in range(args.max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=args.temperature,
                    max_completion_tokens=len(marked_text) // 2 + 200,
                )
                return resp.choices[0].message.content.strip()
            except (RateLimitError, APITimeoutError, APIConnectionError):
                wait = min(2 ** attempt, 60)
                logger.warning(f"Retry {attempt+1}/{args.max_retries} for {record_id}, waiting {wait}s")
                await asyncio.sleep(wait)
            except BadRequestError as e:
                logger.error(f"BadRequest for {record_id}: {e}")
                return None
    logger.error(f"Exhausted retries for {record_id}")
    return None


async def process_record(client, rec, args, sem):
    """Process a single record: extract spans, remove, polish, validate."""
    ebi = sorted(
        [l for l in rec["labels"] if l["label_type"] == "Evident Baseless Info"],
        key=lambda x: x["start"],
    )
    merged = merge_intervals(ebi)
    for span in merged:
        span["text"] = rec["response"][span["start"]:span["end"]]

    naive_text, marked_text, junctions = naive_remove(rec["response"], merged)

    fully_hallucinated = not naive_text.strip()

    if fully_hallucinated or args.dry_run:
        cleaned = naive_text
        dist, ratio, flag = 0, 0.0, None
    else:
        polished = await polish_one(client, rec["id"], marked_text, len(merged), args, sem)
        if polished is None:
            cleaned = naive_text
            dist, ratio, flag = 0, 0.0, "api_failure"
        else:
            cleaned = polished
            dist, ratio, flag = validate(naive_text, cleaned, junctions, args.edit_threshold)

    return {
        "id": rec["id"],
        "source_id": rec["source_id"],
        "model": rec["model"],
        "split": rec["split"],
        "original_response": rec["response"],
        "ebi_spans": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in merged],
        "naive_removed": naive_text,
        "cleaned_response": cleaned,
        "edit_distance": dist,
        "edit_ratio": round(ratio, 4),
        "flag": flag,
        "fully_hallucinated": fully_hallucinated,
    }


async def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    records = load_and_filter(args.input, args.split)
    logger.info(f"Loaded {len(records)} records with EBI labels")

    splits = ["train", "test"] if args.split == "all" else [args.split]

    client = AsyncOpenAI() if not args.dry_run else None
    sem = asyncio.Semaphore(args.concurrency)

    stats = {"processed": 0, "flagged": {}, "fully_hallucinated": 0, "api_failure": 0}

    for sp in splits:
        split_records = [r for r in records if r["split"] == sp]
        if not split_records:
            continue

        output_path = os.path.join(args.output_dir, f"{sp}_cleaned.jsonl")

        done_ids = set()
        if args.resume and os.path.exists(output_path):
            with open(output_path) as f:
                for line in f:
                    done_ids.add(json.loads(line)["id"])
            logger.info(f"Resuming {sp}: {len(done_ids)} already done")

        pending = [r for r in split_records if r["id"] not in done_ids]
        logger.info(f"Processing {sp}: {len(pending)} records ({len(done_ids)} skipped)")

        mode = "a" if args.resume and done_ids else "w"
        with open(output_path, mode) as f:
            pbar = tqdm(total=len(pending), desc=sp, unit="rec")

            async def _process_and_write(rec):
                result = await process_record(client, rec, args, sem)
                f.write(json.dumps(result) + "\n")
                f.flush()
                stats["processed"] += 1
                if result["fully_hallucinated"]:
                    stats["fully_hallucinated"] += 1
                if result["flag"]:
                    stats["flagged"][result["flag"]] = stats["flagged"].get(result["flag"], 0) + 1
                    if result["flag"] == "api_failure":
                        stats["api_failure"] += 1
                pbar.update(1)
                return result

            tasks = [asyncio.create_task(_process_and_write(rec)) for rec in pending]
            await asyncio.gather(*tasks)
            pbar.close()

    logger.info(f"Done. Processed: {stats['processed']}")
    logger.info(f"Fully hallucinated: {stats['fully_hallucinated']}")
    logger.info(f"API failures: {stats['api_failure']}")
    for flag, count in stats["flagged"].items():
        logger.info(f"Flagged {flag}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
