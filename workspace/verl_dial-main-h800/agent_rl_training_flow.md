# 多轮 Agent RL 训练机制详解（ClarQ GRPO 案例）

## 一、核心问题：为什么多轮 Agent RL 比单轮 RLHF 复杂

传统单轮 RLHF：
```
用户问题 → 模型生成一段回答 → 奖励模型打分 → 更新策略
```

多轮 Agent RL：
```
用户问题 
  → 模型决策1（澄清？检索？结束？）
    → 工具执行/用户回复
      → 模型决策2
        → 工具执行/用户回复
          → ...
            → 模型最终决策
              → 整条轨迹结束后才能判断成败 → 奖励 → 更新策略
```

关键差异：
1. **可变长度轨迹**：不同样本的对话轮数不同（1-6 轮）
2. **稀疏奖励**：只在轨迹结束时才知道是否成功（找到正确案例/满足用户需求）
3. **部分可观察**：模型看不到用户真实意图，只能通过工具调用"探索"
4. **组合动作空间**：每步可以选不同工具 + 不同参数

---

## 二、VeRL 的解决方案：ToolAgentLoop 状态机

### 2.1 核心设计：三状态循环

`verl/experimental/agent_loop/tool_agent_loop.py` 实现了一个状态机：

```python
class AgentState(Enum):
    PENDING          # 等待生成（准备 prompt）
    GENERATING       # 模型正在生成
    PROCESSING_TOOLS # 执行工具调用
    TERMINATED       # 轨迹结束
```

**主循环逻辑**（`ToolAgentLoop.run` 方法）：

```python
async def run(self, sampling_params, **kwargs):
    # 1. 初始化 agent_data（存储整条轨迹的状态）
    agent_data = AgentData(
        messages=[],          # 对话历史
        response_ids=[],      # 累积的 token ids
        response_mask=[],     # 区分模型生成 vs 工具返回
        response_logprobs=[], # log 概率（用于 PPO）
        tool_rewards=[],      # 每个工具的即时奖励
        ...
    )
    
    # 2. 状态机循环
    state = AgentState.PENDING
    while state != AgentState.TERMINATED:
        if state == AgentState.PENDING:
            state = await self._handle_pending_state(agent_data)
        elif state == AgentState.GENERATING:
            state = await self._handle_generating_state(agent_data)
        elif state == AgentState.PROCESSING_TOOLS:
            state = await self._handle_processing_tools_state(agent_data)
    
    # 3. 返回完整轨迹
    return AgentLoopOutput(
        prompt_ids=...,
        response_ids=agent_data.response_ids,  # 包含所有轮次
        response_mask=agent_data.response_mask,
        response_logprobs=agent_data.response_logprobs,
        num_turns=agent_data.user_turns + agent_data.assistant_turns,
        reward_score=None,  # 此时还没奖励，后面 ClarQAgentLoop 会算
        ...
    )
```

### 2.2 三个状态的处理细节

#### PENDING 状态：准备 prompt
```python
async def _handle_pending_state(agent_data):
    # 把当前对话历史 + 工具定义 → Hermes 格式的 prompt
    prompt_ids = await self.apply_chat_template(
        agent_data.messages,
        tools=agent_data._active_tool_schemas,  # 工具的 JSON schema
        ...
    )
    agent_data.prompt_ids = prompt_ids
    return AgentState.GENERATING
```

**Hermes 格式示例**（Qwen3 tokenizer 内置支持）：
```
<|im_start|>system
You are a conversational case-retrieval agent.
<tools>
[{"type": "function", "function": {"name": "clarify_user", ...}},
 {"type": "function", "function": {"name": "search_case", ...}}]
</tools>
<|im_end|>
<|im_start|>user
My router keeps disconnecting randomly.
<|im_end|>
<|im_start|>assistant
```

#### GENERATING 状态：模型生成
```python
async def _handle_generating_state(agent_data):
    # 调用 vLLM/SGLang 异步生成
    output: TokenOutput = await self.server_manager.generate(
        request_id=agent_data.request_id,
        prompt_ids=agent_data.prompt_ids,
        sampling_params=sampling_params,
        ...
    )
    
    # 累积到轨迹
    agent_data.response_ids += output.token_ids
    agent_data.response_mask += [1] * len(output.token_ids)  # 1=模型生成
    agent_data.response_logprobs += output.log_probs
    agent_data.assistant_turns += 1
    
    # 检查终止条件
    if len(agent_data.response_mask) >= self.response_length:
        return AgentState.TERMINATED
    if agent_data.assistant_turns >= self.max_assistant_turns:  # 如 6 轮
        return AgentState.TERMINATED
    
    # 解析生成的 token 是否包含工具调用
    _, tool_calls = await self.tool_parser.extract_tool_calls(
        agent_data.response_ids, 
        tools
    )
    
    if tool_calls:
        agent_data.tool_calls = tool_calls
        return AgentState.PROCESSING_TOOLS
    else:
        return AgentState.TERMINATED  # 模型输出 "Complete" 或纯文本
```

**工具调用格式**（模型生成的 token）：
```
<tool_call>
{"name": "search_case", "arguments": {"query": "router disconnecting WiFi"}}
</tool_call>
```

#### PROCESSING_TOOLS 状态：执行工具并返回结果
```python
async def _handle_processing_tools_state(agent_data):
    tasks = []
    for tool_call in agent_data.tool_calls[:self.max_parallel_calls]:
        tasks.append(self._call_tool(tool_call, agent_data))
    
    # 并发执行多个工具（如果模型一次调了多个）
    responses = await asyncio.gather(*tasks)
    
    # 把工具返回结果转成 token，追加到轨迹
    for tool_response, tool_reward, _ in responses:
        message = {"role": "tool", "content": tool_response.text}
        agent_data.messages.append(message)
        
        # 工具返回的文本也要 tokenize
        tool_response_ids = tokenizer.encode(tool_response.text)
        agent_data.prompt_ids += tool_response_ids
        agent_data.response_mask += [0] * len(tool_response_ids)  # 0=工具返回
        
        # 可选：工具的即时奖励（如 ClarQ 里澄清是否有用）
        if tool_reward is not None:
            agent_data.tool_rewards.append(tool_reward)
    
    # 回到 PENDING，准备下一轮生成
    return AgentState.PENDING
```

**关键点**：`response_mask` 区分模型生成(1)和工具返回(0)的 token，PPO 更新时**只对 mask=1 的 token 计算梯度**，工具返回的 token 不参与训练。

---

## 三、GRPO 如何处理多轮轨迹

### 3.1 GRPO 原理回顾

**Group Relative Policy Optimization**：同一个 prompt 采样 N 条轨迹（这里 N=4），用**组内相对奖励**计算优势函数，不需要单独训练 Value 网络。

公式：
```
优势 A_i = (R_i - mean(R_group)) / (std(R_group) + ε)
```

其中 `R_group = [R_1, R_2, R_3, R_4]` 是同一个问题的 4 条不同轨迹的奖励。

### 3.2 VeRL 的 GRPO 实现细节

`verl/trainer/ppo/core_algos.py:compute_grpo_outcome_advantage`：

```python
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,  # (bs, response_length)
    response_mask: torch.Tensor,        # (bs, response_length)
    index: np.ndarray,                  # 每个样本属于哪个 prompt group
    ...
):
    # 1. 把 token-level 奖励求和成轨迹级奖励
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    
    # 2. 按 index 分组，计算每组的均值和标准差
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    
    for idx in id2score:
        scores_tensor = torch.stack(id2score[idx])
        id2mean[idx] = scores_tensor.mean()
        id2std[idx] = scores_tensor.std()
    
    # 3. 归一化：(R_i - μ_group) / (σ_group + ε)
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + 1e-6)
    
    # 4. 广播到 token 维度（每个 token 分到相同的优势值）
    scores = scores.unsqueeze(-1) * response_mask  # (bs, response_length)
    
    return scores, scores
```

**关键机制**：
- `index` 数组标记哪些样本属于同一个 prompt（如 `[0,0,0,0, 1,1,1,1, ...]` 表示前 4 个是第 1 个问题的 4 条轨迹）
- 即使轨迹长度不同（有的 3 轮，有的 6 轮），奖励都是**轨迹级标量**，按组归一化后广播到每个 token
- `response_mask` 确保只有模型生成的 token 参与梯度计算

### 3.3 ClarQ 的奖励计算时机

`examples/clarq_grpo/agent.py:ClarQAgentLoop.run`：

```python
class ClarQAgentLoop(ToolAgentLoop):
    async def run(self, sampling_params, **kwargs):
        # 1. 调用父类的 run，跑完整条多轮对话
        output = await super().run(sampling_params, **kwargs)
        
        # 2. 从 output.extra_fields 里取出最后一次检索的 Top5 案例
        last_cases = output.extra_fields.pop(LAST_CASES_KEY, [])
        last_case_ids = output.extra_fields.pop(LAST_CASE_IDS_KEY, [])
        
        # 3. 让用户模拟器判断 Top5 是否满足需求
        simulator_feedback = await asyncio.to_thread(
            self.simulator.respond,
            profile,           # 包含 core_intent 和 known_info
            last_cases,        # Top5 的标题+正文
        )
        # simulator_feedback 是 "<SATISFIED_DONE>" 或 "<FAILED_DONE>"
        
        # 4. 计算奖励（满意度 + 检索排名 + 澄清质量 + 重复检索惩罚）
        reward_info = compute_result_reward(
            simulator_feedback=simulator_feedback,
            retrieved_case_ids=last_case_ids,
            target_case_id=kwargs["reward_model"]["ground_truth"],
            reward_config=self.reward_config,
        )
        # reward_info = {
        #   "satisfaction_reward": 0.3 或 -1.0,
        #   "retrieval_reward": 1.0/0.6/0.3/-0.5,
        #   "clarification_reward": ...,
        #   "search_penalty": ...,
        #   "total_reward": ...
        # }
        
        # 5. 把总奖励写入 output，返回给 VeRL trainer
        output.reward_score = reward_info["total_reward"]
        output.extra_fields["reward_extra_info"] = reward_info
        return output
```

**核心**：`output.reward_score` 是一个**标量**，对应整条多轮轨迹的最终奖励，不管这条轨迹有几轮对话。

---

## 四、完整训练流程示意图

```
┌─────────────────────────────────────────────────────────────┐
│ VeRL Trainer (main_ppo.py)                                  │
│                                                              │
│  for epoch in epochs:                                       │
│    for batch in dataloader:                                 │
│      ┌────────────────────────────────────────────┐         │
│      │ Rollout Phase（GPU 0-1 vLLM async）       │         │
│      │                                            │         │
│      │  同一个 prompt → rollout N=4 次           │         │
│      │                                            │         │
│      │  轨迹1（3轮）：                            │         │
│      │    用户问 → clarify → 模拟器回 → search →  │         │
│      │    Top5 → Complete                         │         │
│      │    → reward_score = 0.8                    │         │
│      │                                            │         │
│      │  轨迹2（5轮）：                            │         │
│      │    用户问 → search → clarify → 模拟器回 →  │         │
│      │    search → Top5 → Complete                │         │
│      │    → reward_score = 1.2                    │         │
│      │                                            │         │
│      │  轨迹3（2轮）：                            │         │
│      │    用户问 → search → Top5 → Complete      │         │
│      │    → reward_score = -0.3                   │         │
│      │                                            │         │
│      │  轨迹4（4轮）：                            │         │
│      │    用户问 → clarify → 模拟器回 → clarify → │         │
│      │    模拟器回 → search → Top5 → Complete    │         │
│      │    → reward_score = 0.5                    │         │
│      └────────────────────────────────────────────┘         │
│                                                              │
│      ↓                                                       │
│                                                              │
│      ┌────────────────────────────────────────────┐         │
│      │ Advantage 计算（GRPO）                     │         │
│      │                                            │         │
│      │  rewards = [0.8, 1.2, -0.3, 0.5]          │         │
│      │  μ = mean(rewards) = 0.55                 │         │
│      │  σ = std(rewards) = 0.56                  │         │
│      │                                            │         │
│      │  adv_1 = (0.8 - 0.55) / 0.56 = 0.45       │         │
│      │  adv_2 = (1.2 - 0.55) / 0.56 = 1.16       │         │
│      │  adv_3 = (-0.3 - 0.55) / 0.56 = -1.52     │         │
│      │  adv_4 = (0.5 - 0.55) / 0.56 = -0.09      │         │
│      └────────────────────────────────────────────┘         │
│                                                              │
│      ↓                                                       │
│                                                              │
│      ┌────────────────────────────────────────────┐         │
│      │ Actor Update（FSDP，GPU 2-3）             │         │
│      │                                            │         │
│      │  PPO loss = -Σ[adv * ratio * mask]        │         │
│      │                                            │         │
│      │  其中：                                    │         │
│      │  - ratio = π_new / π_old（重要性采样）    │         │
│      │  - mask = response_mask（只更新模型生成部分）│         │
│      │  - 不同轨迹长度 → padding 到同一长度      │         │
│      │                                            │         │
│      │  梯度更新 → 同步权重到 vLLM               │         │
│      └────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## 五、关键技术点总结

### 5.1 可变长度轨迹的处理

**问题**：不同样本的对话轮数不同，token 数量不同，怎么做 batch 训练？

**解决**：
1. **Padding**：`verl/trainer/ppo/padding_utils.py` 把短轨迹 pad 到 batch 最大长度
2. **Mask**：`response_mask` 标记哪些是真实 token，哪些是 padding，loss 计算时用 `mask` 过滤
3. **动态 batch size**：`use_dynamic_bsz=True` 让 VeRL 根据当前 GPU 显存自动调整 micro batch

### 5.2 工具返回 token 的处理

**问题**：工具返回的文本（如检索结果）也会 tokenize 加入轨迹，但这不是模型生成的，不应该算梯度？

**解决**：
- `response_mask`：模型生成=1，工具返回=0
- PPO loss 计算时只对 `mask=1` 的 token 算梯度：
  ```python
  loss = -(advantages * ratio * response_mask).sum() / response_mask.sum()
  ```

### 5.3 稀疏奖励的传播

**问题**：只有轨迹结束时才知道奖励，中间步骤怎么办？

**解决**：
- GRPO 的优势值是**轨迹级**的，直接广播到每个 token：
  ```python
  adv = (R_trajectory - R_group_mean) / R_group_std
  token_advantages = adv.unsqueeze(-1).expand(-1, seq_len) * mask
  ```
- 这等价于"整条轨迹的每个决策都同等负责最终结果"
- 如果想要更细粒度的 credit assignment，可以用 step-level reward（ClarQ 里有 `tool_rewards`，但最终没用上，只用了 outcome reward）

### 5.4 异步 rollout 与同步训练的协调

**流程**：
1. Trainer 把当前策略权重同步给 vLLM worker（通过 checkpoint engine）
2. vLLM worker 异步 rollout（多个 prompt 并发生成，每个生成 N 条轨迹）
3. Rollout 完成后，Trainer 收集所有轨迹 + 奖励
4. Trainer 在 FSDP 模式下计算梯度、更新参数
5. 重复步骤 1

**关键**：`actor_rollout_ref.rollout.free_cache_engine=True` 让 vLLM 在每次 rollout 后释放 KV cache，节省显存给训练用。

---

## 六、与传统单轮 RLHF 的对比

| 维度 | 单轮 RLHF | 多轮 Agent RL (ClarQ GRPO) |
|------|-----------|----------------------------|
| **轨迹结构** | 固定长度回答 | 可变长度多轮对话 |
| **奖励来源** | Reward Model 打分 | 规则奖励 + 用户模拟器判断 |
| **奖励时机** | 生成结束立即可算 | 轨迹结束后才能判断成败 |
| **动作空间** | 纯文本生成 | 工具调用 + 文本生成 |
| **状态表示** | 单次 prompt | 对话历史 + 工具执行结果 |
| **训练目标** | 优化回答质量 | 优化决策序列（什么时候澄清/检索/结束）|
| **Value 网络** | PPO 需要 | GRPO 不需要（用组内相对奖励）|
| **Batch 处理** | 简单 padding | 需要 mask 区分模型生成 vs 工具返回 |

---

## 七、ClarQ 训练配置一览

```yaml
# 关键超参数（run_qwen3_4b_grpo.sh）
ROLLOUT_N: 4                    # 每个 prompt 生成 4 条轨迹（GRPO 组大小）
MAX_ASSISTANT_TURNS: 6          # 最多 6 轮模型决策
MAX_TOOL_RESPONSE_LENGTH: 7000  # 单次工具返回最多 7000 字符（检索结果）
TRAIN_BATCH_SIZE: 2             # 每次训练 2 个不同的 prompt
PPO_MINI_BATCH_SIZE: 2          # Mini batch = 2 prompts × 4 轨迹 = 8 条轨迹
TOTAL_EPOCHS: 3                 # 总共 3 个 epoch
KL_LOSS_COEF: 0.001             # KL 散度权重（防止偏离初始策略太远）

# 奖励权重（agent_loop.yaml）
satisfied: 0.3
not_satisfied: -1.0
top1: 1.0
top3: 0.6
top5: 0.3
miss: -0.5
repeat_search_penalty: -0.05
known_info_clarification_reward: 0.1
unknown_clarification_penalty: -0.02
```

---

## 八、实际训练案例

假设一个 batch：
- Prompt 1（初始问题："路由器一直掉线"）生成 4 条轨迹
- Prompt 2（初始问题："如何配置防火墙规则"）生成 4 条轨迹

**Prompt 1 的 4 条轨迹**：

| 轨迹 | 决策序列 | 轮数 | 最终奖励 | 优势值 |
|------|----------|------|----------|--------|
| 1 | search → Complete | 1 | -0.5（未命中）| -1.2 |
| 2 | clarify → search → Complete | 2 | 1.3（Top1 + 满意）| +1.5 |
| 3 | clarify → clarify → search → Complete | 3 | 0.9（Top3 + 满意）| +0.8 |
| 4 | search → search → Complete | 2 | -0.1（重复检索惩罚）| -1.1 |

GRPO 会让轨迹 2 的每个 token 得到正优势（增强概率），轨迹 1 和 4 得到负优势（降低概率）。

**训练效果**：模型逐渐学会"先澄清关键信息（路由器型号、错误日志）再检索"比"直接盲目检索"效果更好。

---

## 九、代码阅读路径建议

如果想深入理解，建议按这个顺序阅读：

1. **ToolAgentLoop 核心**：`verl/experimental/agent_loop/tool_agent_loop.py`（状态机逻辑）
2. **ClarQ 具体实现**：`examples/clarq_grpo/agent.py`（工具定义 + 奖励计算）
3. **GRPO 优势计算**：`verl/trainer/ppo/core_algos.py:compute_grpo_outcome_advantage`
4. **Trainer 主循环**：`verl/trainer/ppo/ray_trainer.py`（rollout + 训练的编排）
5. **配置文件**：`examples/clarq_grpo/config/*.yaml`（理解超参数含义）

---

希望这个文档能帮你彻底理解多轮 Agent RL 的训练机制！如果还有哪部分不清楚，随时问我。
