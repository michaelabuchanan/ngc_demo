#!/bin/bash
pip install -q --root-user-action=ignore transformers bitsandbytes datasets einops pandas accelerate peft==0.4.0

python ./falcon_finetune.py