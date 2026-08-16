# 对话式检索评估指标

本文定义 `workspace/eval` 中测试流程采用的正式统计口径。默认值可通过
`evaluate.py` 的参数调整，但同一份对比实验应固定参数。

## 1. 符号与样本范围

- 测试样本 $i$ 的唯一正确案例为 $g_i$，来自数据字段 `case_id`。
- 一次检索返回按相关性排序的案例列表 $R_i=[r_{i,1},...,r_{i,K}]$。
- `baseline` 是直接用原始问题 `context` 检索，不经过 Agent。
- `first`、`final` 分别是 Agent 第一次、最后一次真实执行的检索。
- `any` 表示 Agent 任意一次真实检索中的最佳目标排名。
- 一轮（turn）定义为一次策略模型决策，包括追问、检索、`Complete` 或异常文本。
- 默认最多 6 轮、4 次真实检索，每次保留 Top 5。
- 服务、网络、Elasticsearch 或 embedding 故障属于基础设施失败，不进入质量指标分母；
  失败数量和错误明细单独报告。LLM Judge 失败不影响确定性检索指标。

## 2. 结果指标

### Recall@K

对阶段 $s\in\{baseline,first,final,any\}$：

\[
Recall_s@K = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[rank_s(g_i)\le K]
\]

若 Agent 没有执行过检索，该样本的 `first` 和 `final` 记为未命中。默认报告
`K = 1, 3, 5`，并报告 baseline、首次检索、最终检索和任意检索最佳命中四组结果。

### Success Rate

默认成功定义为 GT 案例出现在 Agent 最终一次检索的 Top 1：

\[
SuccessRate = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[rank_{final}(g_i)\le success\_k]
\]

默认 `success_k=1`。可离线改为 3 或 5 后重新聚合，不需要再次调用模型服务。
这里不把“曾经搜到但最终又搜丢”算作最终成功；该现象可由 `any Recall@K` 与
`final Recall@K` 的差异观察。

### MRR

\[
MRR_s = \frac{1}{N}\sum_{i=1}^{N}
\begin{cases}
1/rank_s(g_i), & rank_s(g_i)>0 \\
0, & otherwise
\end{cases}
\]

报告 baseline、final 和 any MRR。Success Rate 与 final Recall@K 同时给出 Wilson
95% 置信区间，便于不同 checkpoint 之间比较。

## 3. 过程指标

### 追问增益

一个“追问块”是两次真实检索之间的一次或连续多次追问。对每个之后确实发生了
检索的追问块进行配对：

- 追问前：最近一次 Agent 检索；如果此前没有 Agent 检索，使用原始问题 baseline。
- 追问后：该追问块结束后的第一次真实检索。

对每个配对 $j$：

\[
Gain_j@K = Hit_{post,j}@K - Hit_{pre,j}@K
\]

总体追问增益是所有可配对追问块的 `Gain@K` 均值，同时报告追问前和追问后
Recall@K。连续追问只产生一个区间，避免把同一次检索收益重复计算。

另报告“有追问样本的 final 相对 baseline 增益”，用于观察整段对话的净效果。
没有后续检索的追问不进入配对增益，但仍进入追问次数和质量统计。

### 追问质量

- 有效追问率：模拟用户返回某条 `known_info` 的次数 / 总追问次数。
- 未知信息追问率：模拟用户返回 `I don't know.` 的次数 / 总追问次数。
- 无效追问率：问题不满足单一、简洁、未重复已知信息等约束的次数 / 总追问次数。
- 重复追问率：规范化后与之前问题相同的次数 / 总追问次数。
- 协议违规率：出现多工具调用、工具调用夹带文本等协议问题的样本比例。

## 4. 效率指标

- 策略轮数：每个样本的策略决策次数，报告均值、中位数和 P95。
- 追问次数：每个样本的 `clarify_user` 调用次数，报告均值。
- 检索次数：真实执行的 `search_case` 次数，报告均值。
- 检索尝试次数：包含超过 4 次上限而未执行的调用，报告均值。
- 端到端耗时：单样本轨迹耗时，报告均值；并发数会影响该指标，比较实验需固定并发配置。
- 正常完成率：以严格文本 `Complete` 结束的样本比例；停止原因另以计数形式报告。

### Success@Turn N

令 $t_i$ 为样本首次在 Agent 的某次真实检索中命中 `success_k` 的策略轮次：

\[
Success@Turn(N) = \frac{1}{N_{samples}}\sum_i\mathbb{1}[t_i\le N]
\]

这是累计指标。baseline 不计入此指标；没有命中的样本在所有轮次均为 0。默认报告
第 1 到第 6 轮。

## 5. 用户满意度（LLM Judge）

Judge 可使用独立服务，默认复用用户模拟器服务。它读取完整轨迹、隐藏的
`core_intent` 和最终案例，结构化判断：

- 是否存在无关问题、重复问题、表达不清晰、非规范用语；
- 是否过早结束；
- 最终案例是否实质满足用户意图；
- 总体是否满意；
- 1 至 5 分总体体验及简短理由。

报告 Judge 覆盖率、平均分、总体满意率和各问题发生率。所有比率只以成功获得
Judge 结果的样本为分母，必须连同覆盖率一起解读。Judge 指标是主观补充，不替代
基于 GT `case_id` 的 Success Rate 和 Recall@K。
