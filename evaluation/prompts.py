"""Shared helpers and prompt templates for the evaluation pipeline."""

import json


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_source_map(source_info_path):
    """Load source_info.jsonl into a dict keyed by source_id."""
    source_map = {}
    with open(source_info_path) as f:
        for line in f:
            d = json.loads(line)
            source_map[d["source_id"]] = d
    return source_map


def load_responses(response_path, split="test", filter_model=None):
    """Load response.jsonl, filtering to split/quality/model."""
    records = []
    with open(response_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("split") and d["split"] != split:
                continue
            if d.get("quality") and d["quality"] != "good":
                continue
            if filter_model and d["model"] != filter_model:
                continue
            records.append(d)
    return records


def load_generated(path):
    """Load generated responses JSONL (output of generate_responses.py)."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def enrich_record(record, source_map):
    """Add reference, question, task_type from source_info to a record."""
    si = source_map[record["source_id"]]
    record["task_type"] = si["task_type"]
    if si["task_type"] == "QA":
        record["reference"] = si["source_info"]["passages"]
        record["question"] = si["source_info"]["question"]
    elif si["task_type"] == "Summary":
        record["reference"] = si["source_info"]
    else:  # Data2txt
        record["reference"] = f"{si['source_info']}"
    return record


# ---------------------------------------------------------------------------
# GPT hallucination judge prompts
# ---------------------------------------------------------------------------

GPT_JUDGE_SYSTEM = (
    "You are an expert hallucination evaluator. You will be given a source "
    "context and a generated response. Your task is to identify all "
    "hallucinated spans in the response -- text that is either:\n\n"
    "1. **Baseless Info**: Information not substantiated by or inferable from "
    "the source context.\n"
    "2. **Conflict**: Information that directly contradicts the source context.\n\n"
    "Output a JSON object with the following structure:\n"
    '{"hallucination list": ["span1", "span2", ...]}\n\n'
    'If there are no hallucinations, output: {"hallucination list": []}\n\n'
    "Rules:\n"
    "- Each span should be the exact text from the response that is hallucinated.\n"
    "- Prefer longer contiguous spans over many small fragments.\n"
    "- Only flag claims of fact. Do not flag stylistic choices, reasonable "
    "inferences, or standard phrasing.\n"
    "- Be conservative: when in doubt, do NOT flag."
)

GPT_JUDGE_USER_TEMPLATES = {
    "QA": (
        "Question: {question}\n\n"
        "Reference passages:\n{reference}\n\n"
        "Response to evaluate:\n{response}"
    ),
    "Summary": (
        "Original article:\n{reference}\n\n"
        "Summary to evaluate:\n{response}"
    ),
    "Data2txt": (
        "Structured data:\n{reference}\n\n"
        "Generated overview to evaluate:\n{response}"
    ),
}


def build_gpt_judge_messages(record, source_map):
    """Build OpenAI chat messages for the GPT hallucination judge."""
    si = source_map[record["source_id"]]
    task_type = si["task_type"]
    template = GPT_JUDGE_USER_TEMPLATES[task_type]

    kwargs = {"response": record["response"]}
    if task_type == "QA":
        kwargs["question"] = si["source_info"]["question"]
        kwargs["reference"] = si["source_info"]["passages"]
    elif task_type == "Summary":
        kwargs["reference"] = si["source_info"]
    else:
        kwargs["reference"] = json.dumps(si["source_info"], indent=2)

    # Ensure reference is a string
    if isinstance(kwargs["reference"], list):
        kwargs["reference"] = "\n\n".join(
            f"Passage {i+1}: {p}" for i, p in enumerate(kwargs["reference"])
        )

    return [
        {"role": "system", "content": GPT_JUDGE_SYSTEM},
        {"role": "user", "content": template.format(**kwargs)},
    ]
