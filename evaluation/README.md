# Evaluation Pipeline for Unlearning-based Hallucination Mitigation

Evaluate whether an unlearned model produces fewer hallucinations by re-generating responses to RAGTruth test prompts and judging them for hallucinations.

## Prerequisites

```bash
pip install transformers torch openai scipy numpy tqdm
```

An OpenAI API key is required for the GPT judge (set `OPENAI_API_KEY`).

## Pipeline Stages

### Stage 1: Generate Responses

Re-generate responses using a local HuggingFace model.

```bash
python generate_responses.py \
  --local-model /path/to/unlearned-model \
  --model-name my-unlearned-model \
  --filter-model llama-2-7b-chat \
  --temperatures 0.7 \
  --output output/generated_responses.jsonl
```

Key arguments:
- `--local-model` (required) — path to a local HF model directory (loads with `device_map="auto"`)
- `--filter-model` — only use source_ids that were originally assigned to this model
- `--max-source-ids N` — limit to first N source_ids (for quick testing)
- `--resume` — skip already-generated source_ids

### Stage 2: Judge Responses

Use GPT-4o to identify hallucinated spans in both the original and unlearned model responses.

```bash
# Judge generated (unlearned) responses
python judge_gpt.py \
  --input output/generated_responses.jsonl \
  --source-info ../dataset/source_info.jsonl \
  --is-generated \
  --output output/unlearned_gpt_judged.jsonl

# Judge original responses (same source_ids for comparison)
python judge_gpt.py \
  --input ../dataset/response.jsonl \
  --source-info ../dataset/source_info.jsonl \
  --filter-model llama-2-7b-chat \
  --output output/original_gpt_judged.jsonl
```

### Stage 3: Compare

Compare hallucination rates between original and unlearned models.

```bash
python compare.py \
  --original-gpt output/original_gpt_judged.jsonl \
  --unlearned-gpt output/unlearned_gpt_judged.jsonl
```

## Quick-Start Test Workflow

Runs a small test with 5 source_ids using Llama-2-7b-chat-hf:

```bash
cd /localscratch/bshang/workspace/RAGTruth/evaluation

MODEL=/egr/research-optml/bshang/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/

# Step 1: Generate responses (5 source_ids)
python generate_responses.py \
  --source-info ../dataset/source_info.jsonl \
  --responses ../dataset/response.jsonl \
  --local-model $MODEL \
  --model-name llama-2-7b-chat-test \
  --filter-model llama-2-7b-chat \
  --temperatures 0.7 \
  --max-source-ids 5 \
  --output output/test_generated.jsonl

# Step 2: GPT judge on generated responses
python judge_gpt.py \
  --input output/test_generated.jsonl \
  --source-info ../dataset/source_info.jsonl \
  --is-generated \
  --output output/test_gpt_judged.jsonl

# Step 3: GPT judge on original responses (same source_ids)
python judge_gpt.py \
  --input ../dataset/response.jsonl \
  --source-info ../dataset/source_info.jsonl \
  --filter-model llama-2-7b-chat \
  --output output/orig_gpt_judged.jsonl

# Step 4: Compare
python compare.py \
  --original-gpt output/orig_gpt_judged.jsonl \
  --unlearned-gpt output/test_gpt_judged.jsonl
```

## Output Format

### generate_responses.py output

```json
{"source_id": "...", "task_type": "QA", "model": "my-model", "temperature": 0.7, "response": "...", "prompt": "..."}
```

### judge_gpt.py output

```json
{"_key": "...", "source_id": "...", "model": "...", "task_type": "QA", "judge": "gpt-4o", "pred": {"hallucination list": ["span1", "span2"]}, "pred_halu": true}
```

For original responses, includes `"is_halu": true/false` from ground-truth annotations.

### compare.py output

Prints a comparison table with hallucination rates (overall and per task type), McNemar's test for statistical significance, and inter-judge agreement when both GPT and baseline judges are used. Optionally saves results as JSON with `--output-json`.

## Metrics

- **Hallucination rate**: fraction of responses judged as containing hallucinations
- **Mean hallucination count**: average number of hallucinated spans per response
- **McNemar's test**: paired significance test on matching source_ids (p < 0.05 = significant)
- **Judge validation**: precision/recall/F1 of the judge against human annotations (when available)
