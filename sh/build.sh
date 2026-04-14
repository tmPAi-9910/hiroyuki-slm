#!/usr/bin/env bash
set -e

python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if python -c \"import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)\"; then
  echo \"CUDA available, installing GPU optimizations...\"
  pip install \"bitsandbytes>=0.43.2\"
  pip install xformers triton
  pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"
else
  echo \"No CUDA, using CPU mode\"
fi