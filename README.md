# MedEvalSeg 
Unified Medical Reasoning Localization (UMRL) Evaluation based on [MedEvalKit](https://github.com/alibaba-damo-academy/MedEvalKit)

## Installation
```bash
conda create -n med_eval python=3.10 -y
conda activate med_eval
pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0
pip install -r requirements.txt

pip install transformers==4.52.4
pip install vllm==0.8.5.post1
pip install xformers
pip install PyYAML safetensors filelock regex tokenizers psutil
pip install 'open_clip_torch[training]'
pip install flash-attn --no-build-isolation
pip install hydra-core hydra-core[omegaconf]
```
