#!/usr/bin/env bash
set -e

# 获取调用脚本时的当前工作目录
CURRENT_DIR="$(pwd)"

# 拼接模型路径（如果传入了第1个参数则优先使用参数，否则使用当前目录拼接）
MODEL_NAME="vlm/Qwen3-VL-8B-Instruct-FP8"
MODEL_PATH="${1:-${CURRENT_DIR}/${MODEL_NAME}}"

echo "=========================================="
echo "当前工作目录: ${CURRENT_DIR}"
echo "加载模型路径: ${MODEL_PATH}"
echo "=========================================="

# 检查模型目录是否存在
if [ ! -d "${MODEL_PATH}" ]; then
    echo "警告: 模型路径不存在: ${MODEL_PATH}"
    echo "请确认是否在包含 ${MODEL_NAME} 的目录下执行此脚本，或传入正确的模型路径参数。"
fi

# 启动 vLLM 服务
exec vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 8222 \
    --limit-mm-per-prompt.video 0 \
    --async-scheduling \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.6 \
    --trust-remote-code \
    --max-num-seqs 1 \
    --kv-cache-dtype fp8
