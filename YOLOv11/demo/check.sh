#!/bin/bash

echo "===== 硬件检查 ====="
nvidia-smi || echo "错误: nvidia-smi 不可用"

echo "\n===== CUDA 检查 ====="
nvcc --version || echo "错误: nvcc 不可用"

echo "\n===== PyTorch 检查 ====="
python3 -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); \
           print(f'CUDA 可用: {torch.cuda.is_available()}'); \
           if torch.cuda.is_available(): \
             print(f'设备: {torch.cuda.get_device_name(0)}')"

echo "\n===== 计算测试 ====="
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q True; then
    python3 -c "import torch; x=torch.rand(1000,1000).cuda(); print('GPU 计算成功:', (x@x).mean())"
else
    echo "无法执行 GPU 计算"
fi