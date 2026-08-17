# 对话式检索评估指标

本文定义 `workspace/eval` 中测试流程采用的正式统计口径。默认值可通过
`evaluate.py` 的参数调整，但同一份对比实验应固定参数。

## 1. 符号与样本范围

- 测试样本 $i$ 的标准案例标题为 $g_i$。测试数据仍提供 `case_id`，评估启动时从可配置的
  案例文档（默认 `workspace/ClarQ/case_answers_with_title.json`）解析出对应 `title`；
  `case_id` 只用于这一步映射和轨迹审计。
- 检索结果中的案例仍保留服务返回的 `case_id`，但所有召回、MRR 和追问增益的命中判断
  都使用 `title` 的精确匹配（区分大小写，去除首尾空白）。
- title 不要求全局唯一；如果案例文档中多个案例具有完全相同的 title，检索到其中任意一个
  都按该 title 命中。这是按 title 评估的既定语义。
- 一次检索返回按相关性排序的案例列表 $R_i=[r_{i,1},...,r_{i,K}]$。
- `baseline` 是直接用原始问题 `context` 检索，不经过 Agent。
- `first`、`final` 分别是 Agent 第一次、最后一次真实执行的检索。
- `any` 表示 Agent 任意一次真实检索中的最佳目标排名。
- 一轮（turn）定义为一次策略模型决策，包括追问、检索、`Complete` 或异常文本。
- 默认最多 6 轮、4 次真实检索，每次保留 Top 5。
- 与训练状态机一致，到达最后一个允许的策略轮次时先终止；该轮若生成工具调用，记录该
  决策但不执行工具，终局判定使用此前最后一次真实检索结果。
- 服务、网络、Elasticsearch、embedding 或终局满意度模拟器故障属于基础设施失败，
  不进入质量指标分母；失败数量和错误明细单独报告。可选轨迹质量 Judge 失败不影响
  Success、Recall 等指标。

## 2. 结果指标

### Recall@K

对阶段 $s\in\{baseline,first,final,any\}$：

\[
Recall_s@K = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[rank_s(g_i)\le K]
\]

若 Agent 没有执行过检索，该样本的 `first` 和 `final` 记为未命中。默认报告
`K = 1, 3, 5`，并报告 baseline、首次检索、最终检索和任意检索最佳命中四组结果。

### Success Rate

Success 完全复用训练时的满意度奖励口径。Agent 轨迹结束后：

- 如果 Agent 没有执行过 `search_case`，或者最后一次真实检索结果为空，直接记为
  `<FAILED_DONE>`，不调用模型；
- 否则把 `core_intent`、原始问题和最后一次真实检索结果交给训练同款 case judge；
- case judge 仅在至少一个最终案例真正解决核心意图时返回 `<SATISFIED_DONE>`，否则返回
  `<FAILED_DONE>`；
- 每条轨迹只进行这一次终局判定，不在每轮或每次检索后调用；
- 不要求策略以严格文本 `Complete` 结束。`Complete` 率作为独立的协议/效率指标报告。

令终局反馈为 $f_i$：

\[
SuccessRate = \frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}[f_i=\texttt{<SATISFIED\_DONE>}]
\]

模拟器返回空字符串、非法选项或 HTTP 400 时，与训练一致回退为 `<FAILED_DONE>`；其他服务
错误按基础设施失败处理。标准 title 是否命中不参与 Success 判定，只用于下方的
Recall@K、MRR 和追问增益；测试数据中的 `case_id` 只用于解析标准 title 和轨迹审计。
因此，可能出现“GT title 未命中但 judge 满意”或“GT title 命中但 judge 不满意”，报告会
如实保留这两组信号。标准 title 解析失败时样本不能进入在线评估；缺少
`target_case_title` 的旧轨迹不能按新口径离线聚合。

### MRR

\[
MRR_s = \frac{1}{N}\sum_{i=1}^{N}
\begin{cases}
1/rank_s(g_i), & rank_s(g_i)>0 \\
0, & otherwise
\end{cases}
\]

报告 baseline、final 和 any MRR。Success Rate 与 final Recall@K 分别给出 Wilson
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

令 $t_i$ 为一条成功轨迹结束时的策略轮数。这里的“结束”采用训练状态机口径：模型输出
任意不包含工具调用的文本（包括 `Complete`），或达到配置的最大策略轮数，都会结束；
最终仍只调用一次 case judge。失败轨迹的 $t_i$ 不定义。

\[
Success@Turn(N) = \frac{1}{N_{samples}}\sum_i
\mathbb{1}[f_i=\texttt{<SATISFIED\_DONE>} \land t_i\le N]
\]

这是累计指标：在第 $N$ 轮内结束且终局满意的样本，会计入第 $N$ 轮及其后的所有轮次。
它不会为了计算不同的 $N$ 重复调用 case judge。默认报告第 1 到第 6 轮；最后一轮的
`Success@Turn` 应等于总体 Success Rate。

## 5. 可选轨迹质量 Judge

这是独立于正式 Success 的诊断项。Judge 可使用独立服务，默认复用用户模拟器服务。
它读取完整轨迹、隐藏的
`core_intent` 和用户已知事实，结构化判断：

- 是否存在无关问题、重复问题、表达不清晰、非规范用语；
- 1 至 5 分总体体验及简短理由。

报告 Judge 覆盖率、平均分和各问题发生率。所有比率只以成功获得 Judge
结果的样本为分母，必须连同覆盖率一起解读。使用 `--skip-judge` 只关闭这个可选诊断，
不会关闭生成 `<SATISFIED_DONE>/<FAILED_DONE>` 的训练同款终局判定。
