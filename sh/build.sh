#!/usr/bin/env bash
set -e

python -m pip install --upgrade pip
python -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python -m pip install --no-deps -r requirements.txt
