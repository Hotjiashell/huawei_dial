#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/data/xujunjie/anaconda3/envs/verl/bin}"
RUNTIME_DIR="${CLARQ_RUNTIME_DIR:-$VERL_ROOT/outputs/clarq_pipeline_4gpu}"
LOG_DIR="$RUNTIME_DIR/logs"
PID_DIR="$RUNTIME_DIR/pids"
ES_ENV="$VERL_ROOT/clarq_search/app.env"
TRAIN_SCRIPT="$VERL_ROOT/examples/clarq_grpo/run_qwen3_4b_grpo.sh"

SIMULATOR_GPU="${SIMULATOR_GPU:-0}"
EMBEDDING_GPU="${EMBEDDING_GPU:-1}"
TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:1235}"

mkdir -p "$LOG_DIR" "$PID_DIR"

export PATH="$CONDA_BIN:$PATH"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

set -a
source "$ES_ENV"
set +a

pid_file() {
    printf '%s/%s.pid\n' "$PID_DIR" "$1"
}

log_file() {
    printf '%s/%s.log\n' "$LOG_DIR" "$1"
}

exit_file() {
    printf '%s/%s.exit\n' "$RUNTIME_DIR" "$1"
}

is_running() {
    local file
    file="$(pid_file "$1")"
    [[ -s "$file" ]] && kill -0 "$(<"$file")" 2>/dev/null
}

assert_gpu_free() {
    local gpu="$1"
    local pids
    pids="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')"
    if [[ -n "$pids" ]]; then
        echo "GPU $gpu already has compute process(es): $pids" >&2
        exit 1
    fi
}

wait_for_url() {
    local name="$1"
    local url="$2"
    local timeout="${3:-900}"
    local deadline=$((SECONDS + timeout))

    while ((SECONDS < deadline)); do
        if curl --noproxy '*' -fsS "$url" -H 'Authorization: Bearer EMPTY' >/dev/null 2>&1; then
            echo "$name is ready: $url"
            return 0
        fi
        if ! is_running "$name"; then
            echo "$name exited before becoming ready. See $(log_file "$name")" >&2
            tail -n 80 "$(log_file "$name")" >&2 || true
            return 1
        fi
        sleep 5
    done

    echo "Timed out waiting for $name. See $(log_file "$name")" >&2
    return 1
}

start_elasticsearch() {
    sg docker -c "docker compose --project-directory '$VERL_ROOT/clarq_search' --env-file '$ES_ENV' up -d"
    for _ in {1..60}; do
        if curl --noproxy '*' -fsS -u "elastic:${ELASTIC_PASSWORD}" \
            "${ELASTICSEARCH_URL}/_cluster/health" >/dev/null 2>&1; then
            echo "elasticsearch is ready: $ELASTICSEARCH_URL"
            return 0
        fi
        sleep 2
    done
    echo "Elasticsearch did not become ready" >&2
    return 1
}

start_simulator() {
    if is_running simulator; then
        echo "simulator already running with PID $(<"$(pid_file simulator)")"
        return
    fi
    assert_gpu_free "$SIMULATOR_GPU"

    nohup setsid env \
        CUDA_VISIBLE_DEVICES="$SIMULATOR_GPU" \
        VLLM_USE_FLASHINFER_SAMPLER=0 \
        "$CONDA_BIN/vllm" serve /data/pretrained_models/Qwen3-32B \
        --served-model-name clarq-user-simulator \
        --host 127.0.0.1 \
        --port 8005 \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --max-num-seqs 4 \
        --gpu-memory-utilization 0.94 \
        --enforce-eager \
        --api-key EMPTY \
        >"$(log_file simulator)" 2>&1 < /dev/null &
    echo "$!" >"$(pid_file simulator)"
    echo "started simulator on GPU $SIMULATOR_GPU with PID $!"
}

start_embedding() {
    if is_running embedding; then
        echo "embedding already running with PID $(<"$(pid_file embedding)")"
        return
    fi
    assert_gpu_free "$EMBEDDING_GPU"

    nohup setsid env \
        CUDA_VISIBLE_DEVICES="$EMBEDDING_GPU" \
        VLLM_USE_FLASHINFER_SAMPLER=0 \
        "$CONDA_BIN/vllm" serve /data/pretrained_models/BAAI-bge-large-en-v1.5 \
        --served-model-name clarq-embedding \
        --runner pooling \
        --host 127.0.0.1 \
        --port 8001 \
        --dtype bfloat16 \
        --max-model-len 512 \
        --max-num-seqs 64 \
        --gpu-memory-utilization 0.20 \
        --api-key EMPTY \
        >"$(log_file embedding)" 2>&1 < /dev/null &
    echo "$!" >"$(pid_file embedding)"
    echo "started embedding on GPU $EMBEDDING_GPU with PID $!"
}

check_dependencies() {
    wait_for_url simulator http://127.0.0.1:8005/v1/models
    wait_for_url embedding http://127.0.0.1:8001/v1/models
    curl --noproxy '*' -fsS -u "elastic:${ELASTIC_PASSWORD}" \
        "${ELASTICSEARCH_URL}/clarq_cases/_count" >/dev/null
}

start_training() {
    local mode="$1"
    local experiment_name="$2"
    local name="training"
    local save_freq
    local test_freq
    local total_epochs
    local -a extra_args=(data.dataloader_num_workers=0)

    if is_running "$name"; then
        echo "training already running with PID $(<"$(pid_file "$name")")" >&2
        exit 1
    fi
    check_dependencies

    IFS=',' read -r -a train_gpu_list <<<"$TRAIN_GPUS"
    for gpu in "${train_gpu_list[@]}"; do
        assert_gpu_free "$gpu"
    done

    if [[ "$mode" == smoke ]]; then
        extra_args+=(trainer.total_training_steps=1 trainer.val_before_train=False)
        save_freq=-1
        test_freq=-1
        total_epochs=1
    else
        save_freq="${SAVE_FREQ:-1800}"
        test_freq="${TEST_FREQ:-40}"
        total_epochs="${TOTAL_EPOCHS:-3}"
    fi

    if [[ -s "$(log_file "$name")" ]]; then
        if [[ -s "$RUNTIME_DIR/experiment_name" ]]; then
            previous_experiment="$(<"$RUNTIME_DIR/experiment_name")"
        else
            previous_experiment=unknown
        fi
        mv "$(log_file "$name")" "$LOG_DIR/${previous_experiment}.log"
    fi
    printf '%s\n' "$experiment_name" >"$RUNTIME_DIR/experiment_name"
    : >"$(log_file "$name")"
    rm -f "$(exit_file "$name")"
    nohup setsid env \
        TRAIN_CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
        TRAIN_N_GPUS_PER_NODE="${#train_gpu_list[@]}" \
        USER_SIMULATOR_BASE_URL=http://127.0.0.1:8005/v1 \
        USER_SIMULATOR_MODEL=clarq-user-simulator \
        USER_SIMULATOR_API_KEY=EMPTY \
        SEARCH_SERVER_STRATEGY=hybrid \
        TRAINER_LOGGER='["console","wandb"]' \
        WANDB_MODE=online \
        PROJECT_NAME=clarq_online_grpo \
        EXPERIMENT_NAME="$experiment_name" \
        SAVE_FREQ="$save_freq" \
        TEST_FREQ="$test_freq" \
        TOTAL_EPOCHS="$total_epochs" \
        bash -c '
            exit_path="$1"
            shift
            "$@"
            code=$?
            printf "%s\n" "$code" >"$exit_path"
            exit "$code"
        ' bash "$(exit_file "$name")" bash "$TRAIN_SCRIPT" "${extra_args[@]}" \
        >"$(log_file "$name")" 2>&1 < /dev/null &
    echo "$!" >"$(pid_file "$name")"
    echo "started $mode training on GPU $TRAIN_GPUS with PID $!"
    echo "experiment: $experiment_name"
    echo "log: $(log_file "$name")"
}

show_status() {
    local name
    for name in simulator embedding training; do
        if is_running "$name"; then
            echo "$name: RUNNING pid=$(<"$(pid_file "$name")") log=$(log_file "$name")"
        elif [[ -s "$(exit_file "$name")" ]]; then
            echo "$name: STOPPED exit=$(<"$(exit_file "$name")") log=$(log_file "$name")"
        else
            echo "$name: STOPPED log=$(log_file "$name")"
        fi
    done
    curl --noproxy '*' -fsS -u "elastic:${ELASTIC_PASSWORD}" \
        "${ELASTICSEARCH_URL}/_cluster/health?filter_path=status" || true
    echo
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits
}

stop_process() {
    local name="$1"
    local file
    local pid
    file="$(pid_file "$name")"
    if ! is_running "$name"; then
        rm -f "$file"
        echo "$name already stopped"
        return
    fi

    pid="$(<"$file")"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..30}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$file"
            echo "$name stopped"
            return
        fi
        sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    rm -f "$file"
    echo "$name killed after timeout"
}

usage() {
    cat <<'EOF'
Usage: scripts/clarq_pipeline_4gpu.sh COMMAND

Commands:
  start       Start Elasticsearch, Qwen3-32B on GPU 0, and embedding on GPU 1
  smoke       Start one-step online-W&B GRPO on GPU 2,3
  train       Start the default three-epoch online-W&B GRPO on GPU 2,3
  status      Show PID, Elasticsearch, and GPU status
  logs        Follow the training log
  restart-simulator
              Restart only the Qwen3-32B service and wait until it is ready
  stop        Stop training, simulator, and embedding; keep Elasticsearch running
  stop-all    Stop model processes and Elasticsearch

GPU and proxy overrides: SIMULATOR_GPU, EMBEDDING_GPU, TRAIN_GPUS, PROXY_URL
EOF
}

command="${1:-}"
case "$command" in
    start)
        start_elasticsearch
        start_simulator
        start_embedding
        wait_for_url simulator http://127.0.0.1:8005/v1/models
        wait_for_url embedding http://127.0.0.1:8001/v1/models
        ;;
    smoke | train)
        timestamp="$(date -u +%Y%m%d_%H%M%S)"
        start_training "$command" "qwen3_4b_full_pipeline_${command}_${timestamp}"
        ;;
    status)
        show_status
        ;;
    logs)
        tail -n 100 -f "$(log_file training)"
        ;;
    restart-simulator)
        stop_process simulator
        start_simulator
        wait_for_url simulator http://127.0.0.1:8005/v1/models
        ;;
    stop)
        stop_process training
        stop_process embedding
        stop_process simulator
        ;;
    stop-all)
        stop_process training
        stop_process embedding
        stop_process simulator
        sg docker -c "docker compose --project-directory '$VERL_ROOT/clarq_search' --env-file '$ES_ENV' stop"
        ;;
    *)
        usage
        exit 1
        ;;
esac
