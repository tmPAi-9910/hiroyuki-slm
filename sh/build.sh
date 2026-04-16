#!/usr/bin/env bash
set -e

echo "Building Hiroyuki-SLM..."
echo "Setting up environment..."
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Build completed."
