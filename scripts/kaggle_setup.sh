#!/usr/bin/env bash
# Kaggle environment setup. Lives in the repo so fixes arrive via git pull (no notebook re-import).
set -euo pipefail

pip install -q -r requirements.txt

# Kaggle ships torchao 0.10; peft>=0.21 refuses torchao<0.16 when it is installed. We don't use torchao.
pip uninstall -y -q torchao || true

python - <<'EOF'
import peft, torch, transformers
from peft import LoraConfig, get_peft_model
m = torch.nn.Sequential()
m.add_module("q_proj", torch.nn.Linear(4, 4))
get_peft_model(m, LoraConfig(r=2, target_modules=["q_proj"]))
print("setup ok: torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
EOF
