#!/bin/bash

# Gemini API 服务启动脚本

echo "🚀 启动 Gemini API 服务..."

# 检测并使用正确的 Python
if [ -f "$HOME/miniconda3/bin/python3" ]; then
    PYTHON_CMD="$HOME/miniconda3/bin/python3"
    PIP_CMD="$HOME/miniconda3/bin/pip"
    echo "✅ 使用 Miniconda3 Python"
elif [ -f "$HOME/anaconda3/bin/python3" ]; then
    PYTHON_CMD="$HOME/anaconda3/bin/python3"
    PIP_CMD="$HOME/anaconda3/bin/pip"
    echo "✅ 使用 Anaconda3 Python"
else
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
    echo "⚠️  使用系统 Python"
fi

# 检查 Python 版本
python_version=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "📌 Python 版本: $python_version"

# 检查依赖是否安装
echo "📦 检查依赖..."
if ! $PYTHON_CMD -c "import fastapi" 2>/dev/null; then
    echo "⚠️  依赖未安装，正在安装..."
    $PIP_CMD install -r requirements.txt
fi

# 检查环境变量
if [ -z "$GEMINI_1PSID" ] || [ -z "$GEMINI_1PSIDTS" ]; then
    echo "⚠️  未设置环境变量 GEMINI_1PSID 或 GEMINI_1PSIDTS"
    echo "💡 提示: 可以在 api.py 中直接配置，或设置环境变量"
fi

# 启动服务
echo "✅ 启动服务..."
$PYTHON_CMD api.py

