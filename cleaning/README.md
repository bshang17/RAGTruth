# Cleaned RAGTruth Data

This directory contains RAGTruth responses with **Evident Baseless Info (EBI)** hallucination spans removed and the resulting text minimally polished by GPT to restore grammatical flow at removal points.

Only records that had at least one EBI-labeled span are included. Other hallucination types (Subtle Baseless Info, Evident Conflict, Subtle Conflict) are left untouched — their labels still apply to the cleaned text.

## Output Files

| File | Records | Description |
|------|---------|-------------|
| `output/train_cleaned.jsonl` | 3,838 | Training split |
| `output/test_cleaned.jsonl` | 541 | Test split |

## Field Schema

Each line is a JSON object with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Response ID (matches `response.jsonl`) |
| `source_id` | string | Source document ID (join to `source_info.jsonl`) |
| `model` | string | LLM that generated the response |
| `split` | string | `"train"` or `"test"` |
| `original_response` | string | Full original response text |
| `ebi_spans` | list | EBI spans that were removed, each with `start`, `end`, `text` |
| `naive_removed` | string | Text after mechanically deleting EBI spans (before GPT polish) |
| `cleaned_response` | string | Final text after GPT polishes grammar at removal junctions |
| `edit_distance` | int | Levenshtein distance between `naive_removed` and `cleaned_response` |
| `edit_ratio` | float | `edit_distance / len(naive_removed)` — fraction of text changed by GPT |
| `flag` | string or null | Quality flag (see below), `null` when clean |

## Flags

GPT is instructed to make only minimal grammar fixes, but sometimes it goes further. The `flag` field captures these cases:

| Flag | Count | Meaning |
|------|-------|---------|
| `null` (no flag) | 3,763 (85.7%) | Clean — GPT edits were minimal and localized |
| `distant_edit` | 391 (8.9%) | GPT edited text more than 30 characters away from a removal junction |
| `content_added` | 185 (4.2%) | GPT added >20 characters beyond the naive removal (may have introduced new content) |
| `excessive_edit` | 54 (1.2%) | Edit ratio exceeded the 10% threshold — GPT rewrote too much |

Records where the entire response was EBI (fully hallucinated) are excluded from the output — they would have empty `cleaned_response` and provide no usable text. This accounts for 14 records dropped from the original pipeline output.

### Recommended handling

- **No flag**: Use `cleaned_response` directly.
- **`distant_edit`**: Usually safe. GPT may have cleaned up a preamble or trailing phrase. Inspect if precision matters.
- **`content_added`**: Review manually — GPT may have inserted filler phrases that constitute new (potentially hallucinated) content.
- **`excessive_edit`**: Consider falling back to `naive_removed` or reviewing the diff.

## Examples

### Clean record (id=729)

A single EBI span ("and the fire has been extinguished") was removed mid-sentence. GPT added the word "but" to fix the transition. `edit_ratio=0.0168`, no flag.

```json
{
  "id": "729",
  "original_response": "A nuclear submarine caught fire during repairs at a Russian shipyard, with no ammunition on board. Insulation caught fire while welding work was being done, and the fire has been extinguished without any injuries.",
  "ebi_spans": [{"start": 157, "end": 191, "text": "and the fire has been extinguished"}],
  "naive_removed": "...while welding work was being done,  without any injuries.",
  "cleaned_response": "...while welding work was being done, but without any injuries.",
  "edit_distance": 3,
  "edit_ratio": 0.0168,
  "flag": null
}
```

### Flagged record (id=3814, `excessive_edit`)

One EBI sentence was removed, but GPT also stripped the "Here's the summary within 106 words:" preamble, pushing the edit ratio to 11.4%.

```json
{
  "id": "3814",
  "original_response": "Here's the summary within 106 words:\n\nOn the latest episode of Big Brother 25, Mecole Hayes was evicted...",
  "ebi_spans": [{"start": 216, "end": 280, "text": "Cirie Fields set a record for most times nominated for eviction."}],
  "naive_removed": "Here's the summary within 106 words:\n\nOn the latest episode of Big Brother 25...",
  "cleaned_response": "On the latest episode of Big Brother 25...",
  "edit_distance": 39,
  "edit_ratio": 0.1137,
  "flag": "excessive_edit"
}
```

## Re-running the Pipeline

Requirements:

```bash
pip install -r requirements.txt  # openai, tqdm, rapidfuzz
```

Set your OpenAI API key, then run from the repo root:

```bash
export OPENAI_API_KEY=sk-...
python cleaning/remove_ebi.py
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `dataset/response.jsonl` | Input file |
| `--output-dir` | `cleaning/output` | Output directory |
| `--model` | `gpt-5.2-2025-12-11` | OpenAI model for polishing |
| `--concurrency` | 20 | Max parallel API calls |
| `--edit-threshold` | 0.10 | Edit ratio above which `excessive_edit` is flagged |
| `--split` | `all` | Process `train`, `test`, or `all` |
| `--resume` | off | Skip already-processed IDs (append mode) |
| `--dry-run` | off | Skip GPT calls, output naive removal only |
