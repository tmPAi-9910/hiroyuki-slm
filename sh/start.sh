#!/usr/bin/env bash
echo "Setting up Hiroyuki-SLM..."

# 一時的な措置
bash sh/build.sh

# 起動
echo "Starting Hiroyuki-SLM..."
python main.py
