#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PRETRAINED_MODELS_ROOT="${PRETRAINED_MODELS_ROOT:-/data/pretrained_models}"

export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-4,5}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

MODEL_PATH="${MODEL_PATH:-$PRETRAINED_MODELS_ROOT/Qwen3-4B}"
TRAIN_FILE="${TRAIN_FILE:-$SCRIPT_DIR/data/train.parquet}"
VAL_FILE="${VAL_FILE:-$SCRIPT_DIR/data/validation.parquet}"
TOOL_CONFIG="$SCRIPT_DIR/config/tool_config.yaml"
AGENT_LOOP_CONFIG="$SCRIPT_DIR/config/agent_loop.yaml"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$VERL_ROOT/outputs/qwen3_4b_usersim_validation_trajectories}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
TRAIN_N_GPUS_PER_NODE="${TRAIN_N_GPUS_PER_NODE:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_N="${ROLLOUT_N:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-8192}"

ACTOR_LR="${ACTOR_LR:-1e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
SAVE_FREQ="${SAVE_FREQ:-1800}"
TEST_FREQ="${TEST_FREQ:-40}"
PROJECT_NAME="${PROJECT_NAME:-clarq_online_grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_usersim}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\",\"wandb\"]}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

export USER_SIMULATOR_MODEL="${USER_SIMULATOR_MODEL:-clarq-user-simulator}"
export USER_SIMULATOR_BASE_URL="${USER_SIMULATOR_BASE_URL:-http://127.0.0.1:8005/v1}"
: "${USER_SIMULATOR_MODEL:?Set USER_SIMULATOR_MODEL to the served Qwen3-32B model name.}"
export USER_SIMULATOR_API_KEY="${USER_SIMULATOR_API_KEY:-EMPTY}"
export USER_SIMULATOR_TEMPERATURE="${USER_SIMULATOR_TEMPERATURE:-0.0}"
export USER_SIMULATOR_TIMEOUT="${USER_SIMULATOR_TIMEOUT:-120}"
export USER_SIMULATOR_MAX_RETRIES="${USER_SIMULATOR_MAX_RETRIES:-3}"
export USER_SIMULATOR_ENABLE_THINKING="${USER_SIMULATOR_ENABLE_THINKING:-false}"
export USER_SIMULATOR_TOKENIZER_PATH="${USER_SIMULATOR_TOKENIZER_PATH:-$PRETRAINED_MODELS_ROOT/Qwen3-32B}"
export USER_SIMULATOR_MAX_INPUT_TOKENS="${USER_SIMULATOR_MAX_INPUT_TOKENS:-7680}"

export SEARCH_SERVER_HOST="${SEARCH_SERVER_HOST:-${ELASTICSEARCH_URL:-http://127.0.0.1:9200}}"
export SEARCH_SERVER_STRATEGY="${SEARCH_SERVER_STRATEGY:-hybrid}"
export ELASTICSEARCH_INDEX="${ELASTICSEARCH_INDEX:-clarq_cases}"
export ELASTICSEARCH_API_KEY="${ELASTICSEARCH_API_KEY:-${ELASTIC_API_KEY:-}}"
export ELASTICSEARCH_USER="${ELASTICSEARCH_USER:-elastic}"
export ELASTICSEARCH_PASSWORD="${ELASTICSEARCH_PASSWORD:-123456}"
export ELASTICSEARCH_CA_CERT="${ELASTICSEARCH_CA_CERT:-}"
export ELASTICSEARCH_VERIFY_CERTS="${ELASTICSEARCH_VERIFY_CERTS:-true}"

export EMBEDDING_URL="${EMBEDDING_URL:-http://127.0.0.1:8001/v1/embeddings}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-clarq-embedding}"
export EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-EMPTY}"
export SEARCH_RANK_WINDOW="${SEARCH_RANK_WINDOW:-50}"
export VECTOR_NUM_CANDIDATES="${VECTOR_NUM_CANDIDATES:-200}"
export BM25_TITLE_BOOST="${BM25_TITLE_BOOST:-3.0}"
export RRF_RANK_CONSTANT="${RRF_RANK_CONSTANT:-60}"
export RRF_BM25_WEIGHT="${RRF_BM25_WEIGHT:-1.0}"
export RRF_TITLE_VECTOR_WEIGHT="${RRF_TITLE_VECTOR_WEIGHT:-1.0}"
export RRF_ANSWER_VECTOR_WEIGHT="${RRF_ANSWER_VECTOR_WEIGHT:-1.0}"
export SEARCH_TIMEOUT="${SEARCH_TIMEOUT:-30}"
export EMBEDDING_TIMEOUT="${EMBEDDING_TIMEOUT:-60}"

for required_path in "$MODEL_PATH/config.json" "$TRAIN_FILE" "$VAL_FILE"; do
    if [[ ! -f "$required_path" ]]; then
        echo "Required file not found: $required_path" >&2
        exit 1
    fi
done

cd "$VERL_ROOT"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.tool_config_path="$TOOL_CONFIG" \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=7000 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.agent.default_agent_loop=clarq_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG" \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
    trainer.n_gpus_per_node="$TRAIN_N_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    "$@"
