#!/bin/bash

SUPPORTED_DATASETS=(
    "MedFrameQA"
    "MedXpertQA-MM"
    "MMMU-Medical-test"
    "MMMU-Medical-val"
    "OmniMedVQA"
    "PATH_VQA"
    "PMC_VQA"
    "SLAKE"
    "VQA_RAD"
)

DATASETS_PATH="/home/dataset-assist-0/diaomuxi/dataset_zoo/MedEvalKit"
OUTPUT_PATH="/home/dataset-assist-0/diaomuxi/yzh/experiments/MedEvalKit/medgemma-4b-it"
MODEL_NAME="MedGemma"
MODEL_PATH="/home/dataset-assist-0/diaomuxi/model_zoo/medgemma-4b-it"

# vllm setting
CUDA_VISIBLE_DEVICES="3"
TENSOR_PARALLEL_SIZE="1"
USE_VLLM="False"

# Eval setting
SEED=42
REASONING="False"
TEST_TIMES=1

# Eval LLM setting
MAX_NEW_TOKENS=8192
MAX_IMAGE_NUM=6
TEMPERATURE=0
TOP_P=0.0001
REPETITION_PENALTY=1

# LLM judge setting
USE_LLM_JUDGE="False"
GPT_MODEL="gpt-4o"
OPENAI_API_KEY="sk-vP6xTOHCJslZlMD686Bc4eA748E44eAe96B182Dd6a11A3Dd"


for EVAL_DATASET in "${SUPPORTED_DATASETS[@]}"; do
    echo "==== Evaluating: $EVAL_DATASET ===="

    python eval.py \
        --eval_datasets "$EVAL_DATASET" \
        --datasets_path "$DATASETS_PATH" \
        --output_path "$OUTPUT_PATH" \
        --model_name "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --seed $SEED \
        --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --use_vllm "$USE_VLLM" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --max_image_num "$MAX_IMAGE_NUM" \
        --temperature "$TEMPERATURE"  \
        --top_p "$TOP_P" \
        --repetition_penalty "$REPETITION_PENALTY" \
        --reasoning "$REASONING" \
        --use_llm_judge "$USE_LLM_JUDGE" \
        --judge_gpt_model "$GPT_MODEL" \
        --openai_api_key "$OPENAI_API_KEY" \
        --test_times "$TEST_TIMES"

    echo "==== Done: $EVAL_DATASET ===="
    echo
done
