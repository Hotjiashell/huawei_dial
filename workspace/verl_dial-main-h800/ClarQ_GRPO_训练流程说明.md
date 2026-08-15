# ClarQ 对话式案例检索 Agent —— GRPO 训练流程说明

## 一句话概括

这个代码库基于开源 RL 训练框架 **VeRL**，用 **GRPO**（Group Relative Policy
Optimization）算法在线训练一个**对话式案例检索智能客服 Agent**（策略模型
`Qwen3-4B`）。这个 Agent 面对用户提出的技术问题时，要学会：

1. 判断是否需要先向用户澄清一个关键信息（调用 `clarify_user` 工具）；
2. 基于用户请求和已澄清的信息，检索出最相关的历史问答案例（调用
   `search_case` 工具，底层是 Elasticsearch 的 BM25 + 向量混合检索）；
3. 在检索结果足够好时输出 `Complete` 结束对话。

训练信号来自两部分：一个固定不训练的 **Qwen3-32B 用户模拟器**（扮演真实提问
用户，回答澄清问题、判断检索结果是否满足需求），以及一个基于「目标案例排名」
的规则奖励。策略模型全程看不到用户的真实意图（`core_intent`）、已知信息
（`known_info`）、参考答案或标准案例 ID，只能通过多轮工具调用去"猜"。

## 数据来源：ClarQ 数据集

`ClarQ`（位于 `workspace/ClarQ/`）来源于 Stack Exchange 的四个站点：
`superuser`（电脑/IT）、`electronics`（电子工程）、`travel`（旅行）、
`money`（金融），每条原始数据是"用户提问 + 澄清问题/回复 + 最终答案"的真实
问答记录。

数据构建管线在 `scripts/build_dataset/` 下，按 6 步顺序执行：

| 步骤 | 脚本 | 作用 |
| --- | --- | --- |
| step1 | `step1_extract_clarq_train_sites.py` | 从原始 ClarQ 训练集中按站点分层抽样（每站 1600 条） |
| step2 | `step2_build_clarq_case_files.py` | 给每条抽样记录分配唯一 `case_id`，把答案单独整理成 `case_answers.json` |
| step3 | `step3_generate_case_titles.py` | 调用一个 LLM（vLLM 服务）为每个案例的答案生成一个简洁标题，产出 `case_answers_with_title.json`（用于检索的标题+正文） |
| step4 | `step4_generate_user_profiles.py` | 调用 LLM，从"提问+答案+澄清对话"反推出模拟用户画像：`core_intent`（用户真实意图）与 `known_info`（用户已知但未主动说出的事实列表），只允许从提问本身和对话中推断，禁止从答案里"偷看"泄露信息 |
| step5 | `step5_split_clarq_dataset.py` | 按 6:1:1 划分 train / validation / test |
| step6 | `step6_index_clarq_cases.py` | 把 `case_answers_with_title.json` 中每条案例的标题、正文向量化后写入 Elasticsearch 索引 `clarq_cases`，供检索工具查询 |

最终产物：
- `workspace/ClarQ/case_answers_with_title.json`：全部案例库（标题+答案），是检索系统的"知识库"。
- `workspace/ClarQ/profile_split/{train,validation,test}/*.json`：训练/验证/测试用的模拟用户画像（按站点分文件），每条包含 `context`（用户初始提问）、`core_intent`、`known_info`、`case_id`（标准答案案例）。

## 从画像到训练样本

`examples/clarq_grpo/prepare_data.py` 把 `profile_split` 的 JSON 转成 VeRL
训练用的 Parquet（`train.parquet` 4800 条、`validation.parquet`/`test.parquet`
各 800 条）。每条样本包含：

- `prompt`：系统提示词（定义 Agent 的行为规则）+ 用户初始问题（`context`）；
- `reward_model.ground_truth`：标准答案案例 `case_id`（策略模型不可见，只用于算奖励）；
- `extra_info.profile`：模拟用户的完整画像（`initial_question`、`core_intent`、`known_info`），会传给用户模拟器，但**不会**出现在策略模型看到的 prompt 里。

## Agent 的行为定义（系统提示词）

策略模型被要求每次只做一个动作：
- 认为缺一个关键信息会显著影响检索结果时，调用 `clarify_user` 问一个具体问题；
- 认为条件已足够时，调用 `search_case` 检索（最多 4 次）；
- 认为最近一次检索结果已经够好时，直接输出 `Complete` 结束。

## 两个工具的实现

代码位于 `examples/clarq_grpo/agent.py`：

### 1. `ClarifyUserTool`（对应 `clarify_user`）
把策略模型提出的澄清问题转发给 `UserSimulatorClient`（背后是 Qwen3-32B）。
模拟器会在"用户已知信息列表"中挑一条**精确匹配**该问题的事实作为回复；如果
没有匹配项，回复"我不知道"；如果问题本身不合理（比如一次问多个事实、或者
重复问初始问题里已经说过的内容），回复"这不是一个合理的澄清问题，我拒绝回
答"。这三种结果都会被记录用于奖励计算。

### 2. `SearchCaseTool`（对应 `search_case`）
调用 `retriever.py` 中的 `CaseRetriever`，对 Elasticsearch 做 **BM25 + 标题向量
+ 答案向量** 三路混合检索，用 **加权 RRF（Reciprocal Rank Fusion）** 融合排序，
返回 Top 5 案例（标题+正文片段）。查询向量由独立部署的 embedding 服务
（`BAAI-bge-large-en-v1.5`，1024 维）生成。单次对话最多检索 4 次，超过后只返回
上一次的结果并提示"已达检索上限"。

## 用户模拟器（Qwen3-32B，固定不训练）

`UserSimulatorClient`（同样在 `agent.py`）承担两个职责：
1. **回答澄清问题**（如上）；
2. **判断最终检索结果是否满足用户需求**：把 Top5 案例的标题和正文交给模拟器，
   结合真实的 `core_intent` 判断是否有一条案例真正解决了用户问题，输出
   `<SATISFIED_DONE>` 或 `<FAILED_DONE>`。

模拟器请求使用 `structured_outputs` 强制模型只能从给定选项中选一个，关闭
"思考"模式，保证判断是精确的固定 token，不做自由文本生成。它作为独立 vLLM
服务运行在 `8005` 端口，训练进程只通过 HTTP 调用它，不会加载或更新它的参数。

## 奖励设计（`compute_result_reward` + `ClarQAgentLoop.run`）

奖励由两大部分相加，再加两个惩罚/加分项：

| 组成 | 规则 |
| --- | --- |
| 用户满意度 | 模拟器输出 `<SATISFIED_DONE>`：`+0.3`；`<FAILED_DONE>`：`-1.0` |
| 检索质量 | 标准案例排名第 1：`+1.0`；第 2-3：`+0.6`；第 4-5：`+0.3`；未命中：`-0.5` |
| 重复检索惩罚 | 每多检索一次（超过第 1 次）：`-0.05` |
| 澄清质量 | 问到真正命中已知信息的问题：`+0.1`；问到未知信息：`-0.02`；问了无效问题：`-0.02` |

总奖励 = 满意度奖励 + 检索奖励 + 重复检索惩罚 + 澄清奖励，写入
`AgentLoopOutput.reward_score`，作为 GRPO 的组内相对奖励信号。

## 训练算法与框架

- **框架**：VeRL（字节跳动开源的 RLHF/Agent RL 训练框架），复用其原生
  `ToolAgentLoop` 多轮工具调用机制，`ClarQAgentLoop` 只是在一轮完整对话结束后
  追加"调用用户模拟器打分 + 计算奖励"的逻辑，不修改 VeRL 的 trainer 或 rollout
  实现。
- **算法**：`algorithm.adv_estimator=grpo`，即 GRPO —— 对同一个 prompt 采样多条
  轨迹（`rollout.n=4`，即每个问题生成 4 条不同对话轨迹），用组内奖励的相对
  大小计算优势函数，不需要单独训练 Critic/Value 模型。
- **策略模型**：`Qwen3-4B`，用 FSDP 做参数分片训练，用 **vLLM 异步 rollout**
  （`rollout.mode=async`）生成多轮对话，多轮工具调用格式为 Hermes 风格的
  `<tool_call>`。
- **训练脚本**：`examples/clarq_grpo/run_qwen3_4b_grpo.sh` 拼装 Hydra 参数调用
  `verl.trainer.main_ppo`；默认 batch=2、GRPO 组大小=4、3 个 epoch（约 4800
  条数据、约 7200 step）。

## 依赖服务与端口（单机四卡实测配置）

| GPU | 进程 | 端口 | 作用 |
| --- | --- | --- | --- |
| GPU0 | Qwen3-32B 用户模拟器（vLLM） | 8005 | 判分/答疑，不参与训练 |
| GPU1 | BAAI-bge-large-en-v1.5（vLLM pooling） | 8001 | 检索用 embedding，1024 维 |
| — | Elasticsearch（Docker） | 9200 | 案例库存储与混合检索 |
| GPU2,3 | Qwen3-4B（VeRL FSDP + vLLM） | — | 被训练的策略模型 |

一键管理脚本：`scripts/clarq_pipeline_4gpu.sh`，支持 `start`（起模拟器+
embedding+ES）、`smoke`（单 step 冒烟测试）、`train`（正式训练）、`status`、
`logs`、`stop`/`stop-all`。

## 整体训练循环（一次对话轨迹）

```
用户初始问题(context)
      │
      ▼
Qwen3-4B(策略模型) 决策 ──► clarify_user? ──► Qwen3-32B(模拟器)按 known_info 回答/拒绝/不知道
      │                                              │
      │◄─────────────────────────────────────────────┘
      ▼
Qwen3-4B 决策 ──► search_case? ──► Elasticsearch 混合检索(BM25+向量, RRF融合) 返回 Top5 案例
      │                                              │
      │◄─────────────────────────────────────────────┘
      ▼
   (可反复澄清/检索，最多4次检索)
      │
      ▼
Qwen3-4B 输出 Complete
      │
      ▼
Qwen3-32B(模拟器) 用 core_intent 判断 Top5 是否满足需求 → <SATISFIED_DONE>/<FAILED_DONE>
      │
      ▼
按目标案例排名 + 满意度 + 澄清质量 + 重复检索 计算 total_reward
      │
      ▼
GRPO：同一问题的多条轨迹按组内相对奖励计算优势 → 更新 Qwen3-4B 参数
```

## 关键代码/配置索引

| 内容 | 路径 |
| --- | --- |
| Agent 与两个工具、用户模拟器客户端、奖励计算 | `examples/clarq_grpo/agent.py` |
| 混合检索实现（BM25+向量+RRF） | `examples/clarq_grpo/retriever.py` |
| 数据画像 → Parquet 转换 | `examples/clarq_grpo/prepare_data.py` |
| 工具/Agent Loop 的 Hydra 配置 | `examples/clarq_grpo/config/{tool_config,agent_loop}.yaml` |
| 训练启动脚本 | `examples/clarq_grpo/run_qwen3_4b_grpo.sh`（也有 `run_qwen2_5_3b_grpo.sh`） |
| 单元测试（mock 外部服务） | `examples/clarq_grpo/test_clarq_grpo.py` |
| 数据构建 6 步脚本 | `scripts/build_dataset/step1_*.py` ~ `step6_*.py` |
| Elasticsearch 索引结构 | `clarq_search/index_mapping.json` |
| Elasticsearch/Docker 配置 | `clarq_search/docker-compose.yml`、`clarq_search/app.env` |
| 单机四卡一键运行/管理脚本 | `scripts/clarq_pipeline_4gpu.sh` |
| 详细服务器操作手册（中文） | `README_zh.md` |
| 验证轨迹样例（每条含输入/输出/各项奖励指标） | `outputs/qwen3_4b_usersim_validation_trajectories/0.jsonl` |

## 备注

- 已在服务器上完成过一次真实端到端验证（2026-08-09，四卡并发：模拟器+
  embedding+hybrid 检索+训练同时跑通，`global_step=1` GRPO 更新成功并同步
  W&B），详情见 `README_zh.md` 顶部的实测结果表。
- W&B 项目地址：`https://wandb.ai/xjj200298-/clarq_online_grpo`。
