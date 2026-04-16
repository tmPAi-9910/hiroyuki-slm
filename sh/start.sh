#!/usr/bin/env bash
set -e

echo "Starting Hiroyuki-SLM..."

if command -v git-lfs &> /dev/null; then
    echo "Pulling model files via Git LFS..."
    git lfs pull
else
    echo "git-lfs not found, assuming files are present."
fi

source venv/bin/activate

export MODEL_PATH="./models/qwen2.5-0.5b-hiroyuki-4bit"

echo "🧠 Loading model from: $MODEL_PATH"
python main.py
