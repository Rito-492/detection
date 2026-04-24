#!/bin/bash

echo "=========================================="
echo "DroneVehicle 基线模型 - 快速启动脚本"
echo "=========================================="
echo ""

# 检查 Python
echo "✓ 检查 Python 环境..."
python3 --version

# 安装依赖
echo ""
echo "✓ 安装依赖..."
pip install -q -r requirements.txt

# 启动 Jupyter
echo ""
echo "✓ 启动 Jupyter Lab..."
echo ""
echo "请在浏览器中打开以下链接:"
echo "  http://localhost:8888"
echo ""
echo "然后打开 Notebook:"
echo "  55230316_基线数据探索与模型构建.ipynb"
echo ""

jupyter lab 55230316_基线数据探索与模型构建.ipynb
