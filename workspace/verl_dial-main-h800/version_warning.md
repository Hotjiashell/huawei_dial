**结论**

不是 BGE-M3、GPU 0 或 `--gpu-memory-utilization 0.10` 导致的。实际复现发现：

- 连 `vllm --help` 都会以退出码 `139` 崩溃，说明发生在模型加载之前。
- PyTorch CUDA 张量计算正常，H800 和 CUDA 基础通信没有问题。
- `vllm._C` 和 FlashInfer 可以正常导入。
- 当前 `flash-attn 2.8.1` 要求 `GLIBC_2.32`，但 Ubuntu 20.04 只有 `GLIBC_2.31`，因此 FlashAttention 和 xFormers 无法正常加载。这也是该预编译 wheel 的已知问题。[相关问题](https://github.com/Dao-AILab/flash-attention/issues/1708)
- `vLLM 0.11.0` 搭配了过新的 `transformers 5.14.1`。绕过段错误后实测报错：
  ```text
  XLMRobertaTokenizer has no attribute all_special_tokens_extended
  ```
- `numpy 2.5.1` 也不满足 `numba 0.61.2` 和 `mistral-common` 的版本要求。
- 顶层 vLLM CLI 导入 benchmark/datasets/PyArrow 时还存在原生库导入顺序崩溃。

这些冲突来自 [install_vllm_sglang_mcore.sh](</home/deeplearn/myhome/verl_dial/scripts/install_vllm_sglang_mcore.sh:16>) 中未设置上限的依赖安装，以及不兼容 Ubuntu 20.04 的 FlashAttention wheel。

**推荐修复**

不要修改训练用的 `verl` 环境，单独建立一个 BGE-M3 服务环境：

```bash
conda create -n bge_vllm python=3.12 -y
conda activate bge_vllm

python -m pip install --upgrade pip

python -m pip install \
  "numpy==1.26.4" \
  "opencv-python-headless==4.11.0.86" \
  "transformers==4.57.1" \
  "vllm==0.11.0"

python -m pip check
```

不要在这个新环境安装脚本中的 `flash_attn-2.8.1...whl`。外部 `flash-attn` 不是 vLLM 的必需依赖；需要使用时，应按照官方说明针对本机 glibc 编译源码。[FlashAttention 安装说明](https://github.com/dao-ailab/flash-attention)

然后启动：

```bash
conda activate bge_vllm

CUDA_VISIBLE_DEVICES=0 vllm serve \
  /home/deeplearn/myhome/pretrained_models/bge-m3 \
  --runner pooling \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.10
```

如果后面出现明确的显存不足错误，再把比例提高到 `0.15` 或 `0.20`。目前的段错误发生在显存分配之前，所以调整这个参数无法解决当前问题。

若必须继续使用现有 `verl` 环境，至少要降级 `transformers` 和 NumPy，并移除不兼容的 FlashAttention；但这可能影响现有 verl 训练依赖，因此独立服务环境更稳妥。