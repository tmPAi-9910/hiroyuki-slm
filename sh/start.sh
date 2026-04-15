#!/usr/bin/env bash
set -e

echo "Setting up Hiroyuki-SLM..."
source venv/bin/activate

echo "Starting Hiroyuki-SLM..."
python main.py
