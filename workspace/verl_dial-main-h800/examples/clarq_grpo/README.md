# ClarQ online multi-turn GRPO

This is a minimal agentic GRPO recipe built on verl's native `ToolAgentLoop`.
It adds no changes to verl's trainer or rollout implementation.

## Components

- `prepare_data.py` converts `ClarQ/profile_split` JSONL files to verl Parquet.
- `clarify_user` asks a fixed user-simulator model one clarification question.
- `search_case` contains a self-contained BM25 + embedding weighted-RRF retriever.
- `ClarQAgentLoop` judges only the final retrieved Top5 and writes the summed
  satisfaction and retrieval reward to `AgentLoopOutput.reward_score`.

The policy sees the initial `context`, clarification replies, and compact case
results. It never sees `core_intent`, `known_info`, the reference answer, or the
ground-truth `case_id`. The simulator profile keeps a separate copy of the
initial question so it can reject clarification questions that repeat it.

## 1. Prepare data

Run this yourself from the repository root:

```bash
conda run -n dl_vllm python verl/examples/clarq_grpo/prepare_data.py
```

This creates:

```text
verl/examples/clarq_grpo/data/
├── train.parquet       # 4,800 rows
├── validation.parquet  #   800 rows
└── test.parquet        #   800 rows
```

The converter accepts alternate locations:

```bash
python verl/examples/clarq_grpo/prepare_data.py \
  --input-root /path/to/ClarQ/profile_split \
  --output-dir /path/to/clarq_parquet
```

## 2. Serve the fixed models

The user simulator must be a separate, fixed OpenAI-compatible service. For
example, adapt the model path and run the following in the environment that
already contains vLLM:

```bash
CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve /data/pretrained_models/Qwen3-32B \
  --served-model-name clarq-user-simulator \
  --host 127.0.0.1 \
  --port 8005 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.78 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --api-key EMPTY
```

Run the embedding service on another GPU with a separate vLLM process:

```bash
CUDA_VISIBLE_DEVICES=2 vllm serve /data/pretrained_models/BAAI-bge-large-en-v1.5 \
  --served-model-name clarq-embedding \
  --host 127.0.0.1 \
  --port 8001 \
  --runner pooling \
  --gpu-memory-utilization 0.10 \
  --max-model-len 512 \
  --max-num-seqs 32 \
  --api-key EMPTY
```

The training process does not load, update, or reserve GPU memory for this
model. Simulator thinking is disabled in requests so terminal judgments remain
exact tokens.

## 3. Configure retrieval

The search tool is fully contained in this directory, so a standalone Verl copy
does not need `dialogue_agent` or its `compenents` package. Configure it with:

```bash
export ELASTICSEARCH_URL=http://127.0.0.1:9200
export ELASTICSEARCH_INDEX=clarq_cases
export ELASTICSEARCH_USER=elastic
export ELASTICSEARCH_PASSWORD='your-password'
export EMBEDDING_URL=http://127.0.0.1:8001/v1/embeddings
export EMBEDDING_MODEL=clarq-embedding
```

API-key authentication can be supplied with `ELASTICSEARCH_API_KEY` instead.

## 4. Start dual-GPU training on GPUs 0–1

Set the served simulator name, then run the script yourself:

```bash
export USER_SIMULATOR_BASE_URL=http://127.0.0.1:8005/v1
export USER_SIMULATOR_MODEL=clarq-user-simulator
export USER_SIMULATOR_API_KEY=EMPTY
export EMBEDDING_MODEL=clarq-embedding

bash examples/clarq_grpo/run_qwen3_4b_grpo.sh
```

Defaults are Qwen3-4B, training GPUs 4–5, four GRPO rollouts per prompt,
batch size 2, six policy decisions, and two searches. All resource parameters
at the top of the script can be overridden with environment variables. Hydra
overrides may also be appended to the command.

Each validation run is saved to
`examples/clarq_grpo/validation_trajectories/{global_step}.jsonl`. Every row
contains the input, complete multi-turn output including tool observations,
ground-truth case ID, reward, retrieval rank, and search count. Set
`VALIDATION_DATA_DIR` to use another output directory.

## Reward

User satisfaction contributes `+0.3` for exact `<SATISFIED_DONE>` and `-1.0`
for `<FAILED_DONE>`. Retrieval contributes `+1.0` when the ground-truth case is
at rank 1, `+0.6` at ranks 2–3, `+0.3` at ranks 4–5, and `-0.5` when absent.
The two components are added. The agent loop keeps Verl's trajectory turn count
only as a saved metric. The simulator prompt receives the clarification-question
count for context, without imposing a clarification-count limit.

Validation metrics include `satisfaction_reward`, `retrieval_reward`,
`total_reward`, `target_rank`, `search_count`, `clarification_count`,
`dialogue_turns`, and `satisfied`.

## Local tests

The tests mock both external services and make no network calls:

```bash
cd verl
python -m unittest examples.clarq_grpo.test_clarq_grpo
```
