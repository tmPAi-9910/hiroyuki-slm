#!/usr/bin/env bash
set -e

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if python -c &quot;import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)&quot; ; then
  echo &quot;CUDA available, installing GPU optimizations...&quot;
  pip install &quot;bitsandbytes&gt;=0.43.2&quot;
  pip install xformers triton
  pip install &quot;unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git&quot;
else
  echo &quot;No CUDA, using CPU mode&quot;
fi