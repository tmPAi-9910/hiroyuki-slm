#!/usr/bin/env bash
set -e

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
