# ClarQ Qwen3 在线 GRPO：A100 服务器运行指南

本文档针对当前实验服务器编写，使用宿主机上的 Conda `verl` 环境运行
VeRL、vLLM 和数据脚本，只用 Docker 启动 Elasticsearch。

策略模型和用户模拟器固定为：

| 角色 | 模型路径 | 启动方式 |
| --- | --- | --- |
| 待训练策略模型 | `/data/pretrained_models/Qwen3-4B` | VeRL FSDP + vLLM |
| 固定用户模拟器 | `/data/pretrained_models/Qwen3-32B` | 独立 vLLM 服务 |
| 检索 embedding | `/data/pretrained_models/BAAI-bge-large-en-v1.5` | 独立 vLLM pooling 服务 |

Qwen3-4B 和 Qwen3-32B 都是 dense `Qwen3ForCausalLM`。当前环境中的
`transformers==4.57.1`、`vllm==0.11.0` 和本仓库 VeRL 均支持该架构。
Qwen3 tokenizer 自带 Hermes 风格的 `<tool_call>` 模板，与训练配置中的
`multi_turn.format=hermes` 一致。

本次服务器实测结果：

| 检查项 | 结果 |
| --- | --- |
| Qwen3-4B 全量权重 + FlashAttention 2 前向 | 通过 |
| Qwen3-32B vLLM 加载与用户模拟器真实请求 | 通过 |
| embedding vLLM 服务与 1024 维索引请求 | 通过 |
| ClarQ 单元测试 | 10/10 通过 |
| VeRL Hydra 训练配置解析 | 通过 |
| Elasticsearch 9.4.2 Docker 服务 | 通过，集群 `green` |
| ClarQ 索引导入 | 通过，alias `clarq_cases` 共 6400 条 |
| BM25/向量/混合检索 | 通过 |
| 四卡完整并发 pipeline | 通过，32B 模拟器、embedding、hybrid 检索与 4B 双卡训练同时运行 |
| Qwen3-4B 一步 GRPO | 通过，`global_step=1`，完成 actor 参数更新并同步 W&B |

2026-08-09 已在 GPU `0,1,2,3` 上完成一次真实并发端到端测试：Qwen3-32B
生成用户回复，检索工具调用 embedding 和 Elasticsearch，Qwen3-4B 完成 rollout、
reference log-prob、GRPO actor 更新和 W&B 在线同步。验证 run 为
`qwen3_4b_full_pipeline_wandb_verified_20260809_0413`，状态为 `finished`。

## 一、固定路径与服务端口

```bash
export WORKSPACE_ROOT=/data/xujunjie/huawei_dial/workspace
export VERL_ROOT=${WORKSPACE_ROOT}/verl_dial-main-h800
export CLARQ_ROOT=${WORKSPACE_ROOT}/ClarQ
export PRETRAINED_MODELS_ROOT=/data/pretrained_models

export POLICY_MODEL_PATH=${PRETRAINED_MODELS_ROOT}/Qwen3-4B
export USER_SIMULATOR_MODEL_PATH=${PRETRAINED_MODELS_ROOT}/Qwen3-32B
export EMBEDDING_MODEL_PATH=${PRETRAINED_MODELS_ROOT}/BAAI-bge-large-en-v1.5
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
```

Qwen3-32B、embedding 和训练分别在不同终端启动时，每个终端都要设置上述
`NO_PROXY`/`no_proxy`。否则服务器继承的 HTTP 代理可能把本机请求转发出去并
返回 502。也可以在本机健康检查的 `curl` 命令中加 `--noproxy '*'`。

本 pipeline 使用以下本机端口：

| 端口 | 服务 |
| --- | --- |
| `8005` | Qwen3-32B 用户模拟器 |
| `8001` | embedding 服务 |
| `9200` | Elasticsearch |

端口只监听 `127.0.0.1`。启动前检查是否被占用：

```bash
ss -ltn | rg ':(8001|8005|9200)\b' || true
```

## 二、激活已安装的 VeRL 环境

环境已经创建在 `/data/xujunjie/anaconda3/envs/verl`：

```bash
source /data/xujunjie/anaconda3/etc/profile.d/conda.sh
conda activate verl
cd /data/xujunjie/huawei_dial/workspace/verl_dial-main-h800
```

检查关键包：

```bash
python -m pip check
python - <<'PY'
import datasets
import flash_attn
import flashinfer
import ray
import tensordict
import torch
import transformers
import vllm
import verl

print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
print("verl:", verl.__version__)
print("ray:", ray.__version__)
print("datasets:", datasets.__version__)
print("tensordict:", tensordict.__version__)
print("flash_attn:", flash_attn.__version__)
print("flashinfer:", flashinfer.__version__)
PY
```

不需要创建 VeRL Docker 容器，也不要在官方 VeRL 镜像中重新安装依赖。
本指南中的 Python、vLLM 和训练命令都在这个 Conda 环境中执行。

## 三、GPU 分配

每次启动模型前先执行：

```bash
nvidia-smi
```

不要占用已有进程使用的 GPU。当前已验证的四卡分配为：

| 进程 | 示例物理 GPU | 数量 |
| --- | --- | --- |
| Qwen3-32B 用户模拟器 | `0` | 1 |
| embedding | `1` | 1 |
| Qwen3-4B 训练 | `2,3` | 2 |

Qwen3-32B 使用 BF16、TP=1、8192 上下文和 0.94 显存利用率，实测占用约
77.4 GiB。embedding 约占 1.4 GiB；训练时 GPU 2、3 每卡峰值分配约
38.2 GiB。编号可用环境变量覆盖，但每次启动仍必须先检查 `nvidia-smi`。

### 3.1 推荐的一键后台运行方式

`scripts/clarq_pipeline_4gpu.sh` 已封装 Conda 路径、四卡分配、代理、服务健康
检查、PID 和日志。vLLM 与训练进程均使用 `nohup` 在后台运行：

```bash
cd /data/xujunjie/huawei_dial/workspace/verl_dial-main-h800

# 启动 Elasticsearch、32B 用户模拟器和 embedding；已启动时可重复执行
scripts/clarq_pipeline_4gpu.sh start

# 检查服务、PID、Elasticsearch 和 GPU
scripts/clarq_pipeline_4gpu.sh status

# 先跑一次单步完整验证，命令返回后训练仍在后台
scripts/clarq_pipeline_4gpu.sh smoke
scripts/clarq_pipeline_4gpu.sh logs

# 单步验证通过后，启动默认 3 epoch 正式训练
scripts/clarq_pipeline_4gpu.sh train
scripts/clarq_pipeline_4gpu.sh logs
```

`logs` 使用 `tail -f`，按 `Ctrl-C` 只退出日志查看，不会停止后台训练。默认日志在
`outputs/clarq_pipeline_4gpu/logs/`，PID 在
`outputs/clarq_pipeline_4gpu/pids/`。

`smoke` 只执行 1 个完整训练 step。`train` 默认使用 4800 条训练数据、batch 2、
3 epoch，约 7200 step；按本次实测 35.7 秒/step 粗略估算约 71 小时，另加初始化
和验证时间。正式训练默认每 1800 step 保存 checkpoint、每 40 step 验证，输出在
`checkpoints/clarq_online_grpo/<experiment_name>/`。需要先跑较短实验时可执行：

```bash
TOTAL_EPOCHS=1 SAVE_FREQ=200 TEST_FREQ=40 \
scripts/clarq_pipeline_4gpu.sh train
```

覆盖资源分配或代理的示例：

```bash
SIMULATOR_GPU=0 EMBEDDING_GPU=1 TRAIN_GPUS=2,3 \
PROXY_URL=http://127.0.0.1:1235 \
scripts/clarq_pipeline_4gpu.sh start
```

## 四、检查两个 Qwen3 模型

先做不加载权重的配置和 tokenizer 检查：

```bash
python - <<'PY'
from transformers import AutoConfig, AutoTokenizer

paths = [
    "/data/pretrained_models/Qwen3-4B",
    "/data/pretrained_models/Qwen3-32B",
]
tools = [{
    "type": "function",
    "function": {
        "name": "search_case",
        "description": "Search cases.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]

for path in paths:
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Find a relevant case."}],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert config.model_type == "qwen3"
    assert config.architectures == ["Qwen3ForCausalLM"]
    assert "<tools>" in rendered and "<tool_call>" in rendered
    print(path, "OK", config.architectures[0], config.max_position_embeddings)
PY
```

## 五、准备 ClarQ Parquet

当前服务器已存在：

```text
/data/xujunjie/huawei_dial/workspace/ClarQ/profile_split/
/data/xujunjie/huawei_dial/workspace/ClarQ/case_answers_with_title.json
```

生成 VeRL 使用的 Parquet：

```bash
cd /data/xujunjie/huawei_dial/workspace/verl_dial-main-h800

python examples/clarq_grpo/prepare_data.py \
  --input-root /data/xujunjie/huawei_dial/workspace/ClarQ/profile_split \
  --output-dir /data/xujunjie/huawei_dial/workspace/verl_dial-main-h800/examples/clarq_grpo/data
```

检查行数，预期为 train 4800、validation 800、test 800：

```bash
python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

root = Path("examples/clarq_grpo/data")
for split in ("train", "validation", "test"):
    path = root / f"{split}.parquet"
    print(split, pq.read_metadata(path).num_rows, path)
PY
```

如果修改了用户画像字段，必须重新运行本节，旧 Parquet 不会自动更新。

## 六、启动 Qwen3-32B 用户模拟器

不使用一键脚本时，可在独立终端选择一张 80 GiB A100。当前四卡方案使用 GPU 0：

```bash
source /data/xujunjie/anaconda3/etc/profile.d/conda.sh
conda activate verl

export SIMULATOR_CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES="${SIMULATOR_CUDA_VISIBLE_DEVICES}" \
VLLM_STAT_LOG_INTERVAL=60 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve /data/pretrained_models/Qwen3-32B \
  --served-model-name clarq-user-simulator \
  --host 127.0.0.1 \
  --port 8005 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.94 \
  --api-key EMPTY
```

验证模型列表和一次禁用 thinking 的对话：

```bash
curl -fsS http://127.0.0.1:8005/v1/models \
  -H 'Authorization: Bearer EMPTY'

curl -fsS http://127.0.0.1:8005/v1/chat/completions \
  -H 'Authorization: Bearer EMPTY' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "clarq-user-simulator",
    "messages": [{"role": "user", "content": "Reply with exactly OK"}],
    "temperature": 0,
    "max_tokens": 16,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

训练进程只通过 HTTP 调用该模型，不会更新或重复加载 Qwen3-32B。

如果需要重新生成用户画像或案例标题，也可以复用这个服务：

```bash
export VLLM_CHAT_URL=http://127.0.0.1:8005/v1/chat/completions
export VLLM_CHAT_MODEL=clarq-user-simulator
export VLLM_CHAT_API_KEY=EMPTY
```

## 七、启动 embedding 服务

当前服务器没有 `/data/pretrained_models/bge-m3`，但已有
`/data/pretrained_models/BAAI-bge-large-en-v1.5`。ClarQ 数据为英文，且该模型
同样输出 1024 维向量，因此本配置直接使用现有模型。其最大输入长度为 512，
`clarq_search/app.env` 已设置 `MAX_INPUT_TOKENS=512`，较长案例会截断到该长度。

在独立终端中选择一张空闲 GPU，当前四卡方案使用 GPU 1：

```bash
source /data/xujunjie/anaconda3/etc/profile.d/conda.sh
conda activate verl

export EMBEDDING_CUDA_VISIBLE_DEVICES=1
CUDA_VISIBLE_DEVICES="${EMBEDDING_CUDA_VISIBLE_DEVICES}" \
VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve /data/pretrained_models/BAAI-bge-large-en-v1.5 \
  --served-model-name clarq-embedding \
  --runner pooling \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 512 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.20 \
  --api-key EMPTY
```

验证服务，并检查向量维度必须为 1024：

```bash
curl -fsS http://127.0.0.1:8001/v1/embeddings \
  -H 'Authorization: Bearer EMPTY' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "clarq-embedding",
    "input": ["How do I configure a wireless router?"],
    "encoding_format": "float"
  }' > /tmp/clarq_embedding_check.json

python - <<'PY'
import json

with open("/tmp/clarq_embedding_check.json", encoding="utf-8") as file:
    payload = json.load(file)
print("embedding dims:", len(payload["data"][0]["embedding"]))
assert len(payload["data"][0]["embedding"]) == 1024
PY
```

## 八、用 Docker 启动 Elasticsearch

### 8.1 一次性处理 Docker 权限

当前 `xujunjie` 账号已经加入 `docker` 组。新登录 shell 可以直接运行：

```bash
id
docker ps
```

如果 `id` 暂时还看不到 `docker` 组，说明当前 shell 创建于加组之前。重新登录
服务器即可；临时也可用 `sg docker -c 'docker ps'` 验证权限。

不要通过 `chmod 666 /var/run/docker.sock` 绕过权限。VeRL 继续使用 Conda，
Docker 只承载 Elasticsearch。

### 8.2 检查 compose 配置

`clarq_search/app.env` 已配置为当前路径：

```text
CASE_ANSWERS_PATH=/data/xujunjie/huawei_dial/workspace/ClarQ/case_answers_with_title.json
```

检查 compose 展开结果：

```bash
cd /data/xujunjie/huawei_dial/workspace/verl_dial-main-h800/clarq_search
docker compose --env-file app.env config
```

Elasticsearch 镜像已经在本机时，直接启动：

```bash
docker compose --env-file app.env up -d
docker compose --env-file app.env ps
docker compose --env-file app.env logs --tail=100 elasticsearch
```

只有在镜像确实不存在时才需要：

```bash
docker compose --env-file app.env pull
```

### 8.3 验证 Elasticsearch

```bash
cd /data/xujunjie/huawei_dial/workspace/verl_dial-main-h800/clarq_search
set -a
source app.env
set +a

curl -fsS -u "elastic:${ELASTIC_PASSWORD}" \
  http://127.0.0.1:9200

curl -fsS -u "elastic:${ELASTIC_PASSWORD}" \
  'http://127.0.0.1:9200/_cluster/health?pretty'
```

## 九、创建并填充 ClarQ 索引

确保 embedding 服务已在 `8001` 端口运行。

### 9.1 创建索引

该命令只在物理索引不存在时创建 `clarq_cases_v1`，并同时创建
`clarq_cases` alias：

```bash
cd /data/xujunjie/huawei_dial/workspace/verl_dial-main-h800/clarq_search
set -a
source app.env
set +a

if ! curl -fsS -u "elastic:${ELASTIC_PASSWORD}" \
  "${ELASTICSEARCH_URL}/clarq_cases_v1" >/dev/null; then
  curl -fsS -u "elastic:${ELASTIC_PASSWORD}" \
    -X PUT "${ELASTICSEARCH_URL}/clarq_cases_v1?pretty" \
    -H 'Content-Type: application/json' \
    --data-binary @index_mapping.json
fi
```

### 9.2 生成向量并导入案例

索引脚本现在同时兼容 `CASE_DATA_PATH` 和
`CASE_ANSWERS_PATH`，也兼容 `EMBEDDING_DIM` 和
`EMBEDDING_DIMS`：

```bash
cd /data/xujunjie/huawei_dial/workspace/verl_dial-main-h800
set -a
source clarq_search/app.env
set +a

export BATCH_SIZE=16
export SKIP_EXISTING=1

python scripts/build_dataset/step6_index_clarq_cases.py
```

导入可断点续跑，已存在的 `case_id` 会跳过。检查数量：

```bash
curl -fsS -u "elastic:${ELASTIC_PASSWORD}" \
  "${ELASTICSEARCH_URL}/clarq_cases/_count?pretty"
```

### 9.3 检查混合检索

```bash
python - <<'PY'
import os
from examples.clarq_grpo.retriever import CaseRetriever

retriever = CaseRetriever({
    "search_server_host": os.environ["ELASTICSEARCH_URL"],
    "search_strategy": "hybrid",
    "index": os.environ["ELASTICSEARCH_INDEX"],
    "elasticsearch_user": os.environ["ELASTICSEARCH_USER"],
    "elasticsearch_password": os.environ["ELASTICSEARCH_PASSWORD"],
    "embedding_url": os.environ["EMBEDDING_URL"],
    "embedding_model": os.environ["EMBEDDING_MODEL"],
    "top_k": 5,
})
for item in retriever.search("wireless router configuration", 3):
    print(item.url, item.title, item.score)
PY
```

## 十、配置并启动 Qwen3-4B GRPO

在训练终端中激活环境并导出服务配置：

```bash
source /data/xujunjie/anaconda3/etc/profile.d/conda.sh
conda activate verl
cd /data/xujunjie/huawei_dial/workspace/verl_dial-main-h800

set -a
source clarq_search/app.env
set +a

export USER_SIMULATOR_BASE_URL=http://127.0.0.1:8005/v1
export USER_SIMULATOR_MODEL=clarq-user-simulator
export USER_SIMULATOR_API_KEY=EMPTY
export USER_SIMULATOR_ENABLE_THINKING=false

export SEARCH_SERVER_HOST=http://127.0.0.1:9200
export SEARCH_SERVER_STRATEGY=hybrid
export EMBEDDING_URL=http://127.0.0.1:8001/v1/embeddings
export EMBEDDING_MODEL=clarq-embedding
export EMBEDDING_API_KEY=EMPTY

export MODEL_PATH=/data/pretrained_models/Qwen3-4B
export TRAIN_CUDA_VISIBLE_DEVICES=2,3
export TRAIN_N_GPUS_PER_NODE=2
export TRAINER_LOGGER='["console","wandb"]'
export VLLM_USE_FLASHINFER_SAMPLER=0
```

先检查 Hydra 最终配置，不加载模型：

```bash
bash examples/clarq_grpo/run_qwen3_4b_grpo.sh --cfg job \
  > /tmp/clarq_qwen3_4b_config.yaml

rg -n 'Qwen3-4B|clarq_agent|tool_config|tensor_model_parallel_size' \
  /tmp/clarq_qwen3_4b_config.yaml
```

做一次最小端到端训练冒烟测试：

```bash
TRAIN_BATCH_SIZE=2 \
PPO_MINI_BATCH_SIZE=2 \
PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
ROLLOUT_N=4 \
SAVE_FREQ=-1 \
TEST_FREQ=-1 \
TOTAL_EPOCHS=1 \
bash examples/clarq_grpo/run_qwen3_4b_grpo.sh \
  trainer.total_training_steps=1 \
  trainer.val_before_train=False
```

正式训练：

```bash
bash examples/clarq_grpo/run_qwen3_4b_grpo.sh
```

默认训练参数为：

```text
训练模型：Qwen3-4B
训练 GPU：2,3，可用 TRAIN_CUDA_VISIBLE_DEVICES 覆盖
train batch：2 prompts
GRPO group size：4
PPO mini batch：2 prompts
每卡 PPO micro batch：1
最大 prompt：4096 tokens
最大多轮 response：4096 tokens
vLLM 最大上下文：8192 tokens
最大策略生成次数：6
最大检索次数：2
日志：console + W&B
```

训练脚本已按当前使用者要求配置 W&B key，并默认启用 `console` 和 `wandb`。
冒烟测试或离线运行时可以显式关闭 W&B：

```bash
export TRAINER_LOGGER='["console"]'
bash examples/clarq_grpo/run_qwen3_4b_grpo.sh
```

也可在 shell 中设置新的 `WANDB_API_KEY` 覆盖脚本默认值。

当前服务器配置的外网代理端口为 `127.0.0.1:1235`，已验证可访问 W&B 与
Hugging Face。正式启用 W&B 前先确认端口正在监听，然后设置：

```bash
ss -ltn | rg ':1235\b'
export HTTP_PROXY=http://127.0.0.1:1235
export HTTPS_PROXY=http://127.0.0.1:1235
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
```

如果端口没有监听，应先恢复当前 SSH/代理连接；不要修改服务器的全局网络设置。
一键脚本默认使用该端口，也可通过 `PROXY_URL` 覆盖。

### 10.1 在 W&B 查看结果

项目地址为 <https://wandb.ai/xjj200298-/clarq_online_grpo>。进入 `Runs`，按
`Created` 排序并点击本次 experiment。重点查看：

| 指标 | 含义 |
| --- | --- |
| `training/global_step` | 已完成训练步数 |
| `critic/rewards/mean` | batch 平均奖励 |
| `actor/loss`、`actor/grad_norm` | 策略更新及梯度状态 |
| `response_length/mean`、`num_turns/mean` | 多轮轨迹长度 |
| `perf/throughput`、`timing_s/step` | 吞吐与单步耗时 |

本次已验证 run：
<https://wandb.ai/xjj200298-/clarq_online_grpo/runs/au61p10o>。

## 十一、基础测试与 pipeline 判定

无需外部服务的单元测试：

```bash
python -m unittest -v examples/clarq_grpo/test_clarq_grpo.py
```

完整 pipeline 可运行应同时满足：

```bash
python -m pip check
curl -fsS http://127.0.0.1:8005/v1/models \
  -H 'Authorization: Bearer EMPTY' >/dev/null
curl -fsS http://127.0.0.1:8001/v1/models \
  -H 'Authorization: Bearer EMPTY' >/dev/null
curl -fsS -u "elastic:${ELASTIC_PASSWORD}" \
  "${ELASTICSEARCH_URL}/_cluster/health" >/dev/null
curl -fsS -u "elastic:${ELASTIC_PASSWORD}" \
  "${ELASTICSEARCH_URL}/clarq_cases/_count" >/dev/null
```

然后依次通过：

1. Qwen3 模型配置/tokenizer 检查。
2. embedding 维度检查。
3. 混合检索检查。
4. Hydra 配置检查。
5. 一步 GRPO 冒烟训练。

## 十二、停止服务

停止 Elasticsearch 但保留索引数据：

```bash
cd /data/xujunjie/huawei_dial/workspace/verl_dial-main-h800/clarq_search
docker compose --env-file app.env stop
```

重新启动：

```bash
docker compose --env-file app.env start
```

只有确定不再需要容器时才删除容器：

```bash
docker compose --env-file app.env down
```

不要添加 `-v`，否则会删除 Elasticsearch volume 和已导入索引。
使用一键脚本时，停止模型与训练但保留 Elasticsearch：

```bash
scripts/clarq_pipeline_4gpu.sh stop
```

连同 Elasticsearch 一起停止但保留索引 volume：

```bash
scripts/clarq_pipeline_4gpu.sh stop-all
```

手动前台启动 Qwen3-32B 与 embedding 时，可在各自终端中按 `Ctrl-C` 停止。

## 十三、常见问题

### Docker socket permission denied

账号已经加入 `docker` 组；旧 shell 的附加组列表不会自动刷新，请重新登录。
需要在旧 shell 临时执行命令时使用 `sg docker -c '<docker command>'`。

### Elasticsearch 返回 401

确保训练 shell 与 compose 使用同一个 `clarq_search/app.env`。使用 Basic Auth
时不要设置错误的 `ELASTICSEARCH_API_KEY`，API Key 的优先级高于用户名密码。

### 本机服务请求返回 502

这是 HTTP 代理错误转发本机请求。确认 `NO_PROXY` 和 `no_proxy` 均包含
`127.0.0.1,localhost`；Qwen3-4B 训练脚本已自动补上这两个地址。

### vLLM 启动 Qwen3-32B 时 OOM

当前 TP=1 配置要求一张完整空闲的 80 GiB A100。先确认没有残留进程，再依次
降低 `--gpu-memory-utilization`、`--max-num-seqs`；不要把
`--max-model-len` 降到 8192 以下，否则真实多轮用户模拟 prompt 可能超过上下文。
也可改用两张空闲卡和 `--tensor-parallel-size 2`。

### FlashInfer sampler JIT 编译失败

宿主机系统 `nvcc` 是 11.5，而 Conda 环境中的 PyTorch/vLLM 使用 CUDA 12.8。
保持 `VLLM_USE_FLASHINFER_SAMPLER=0`，让 vLLM 使用内置 sampler。模型注意力
仍使用 FlashAttention，不需要卸载 FlashInfer。

### Qwen3-4B 训练 OOM

先降低 `ROLLOUT_GPU_MEMORY_UTILIZATION`、`MAX_MODEL_LEN` 和
`actor_rollout_ref.rollout.max_num_seqs`。仍有问题时再启用 optimizer/parameter
offload。

### PPO batch 整除错误

保持：

```text
PPO_MINI_BATCH_SIZE <= TRAIN_BATCH_SIZE
TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE == 0
MAX_MODEL_LEN >= MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH
ROLLOUT_N >= 4
```

### 旧 Parquet 缺少 initial_question

重新执行第五节的 `prepare_data.py`，不要继续使用旧 Parquet。

## 奖励定义

```text
用户模拟器输出 <SATISFIED_DONE>：+0.3
用户模拟器输出 <FAILED_DONE>：   -1.0
标答案例 Top 1：                  +1.0
标答案例 Top 2-3：                +0.6
标答案例 Top 4-5：                +0.3
未命中或未检索：                 -0.5
```

总奖励为用户满意度奖励与检索奖励之和。
