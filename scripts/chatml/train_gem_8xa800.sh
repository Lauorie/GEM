#!/bin/bash

set -euo pipefail
set -x

export FLASH_ATTENTION_DETERMINISTIC="${FLASH_ATTENTION_DETERMINISTIC:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

PYTHON_BIN=${PYTHON_BIN:-python3}

TRAIN_TOKENIZED_FILE=${TRAIN_TOKENIZED_FILE:-"./data/chatml/train_tokenized.jsonl"}
TEST_TOKENIZED_FILE=${TEST_TOKENIZED_FILE:-""}

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"meta-llama/Llama-3.1-8B"}
TOKENIZER_NAME_OR_PATH=${TOKENIZER_NAME_OR_PATH:-"meta-llama/Llama-3.1-8B-Instruct"}
OUTPUT_DIR=${OUTPUT_DIR:-"./log/chatml-gem-8xa800-$(date "+%Y-%m-%d-%H-%M-%S")"}

DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"scripts/zero3.json"}
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29500}
SEED=${SEED:-1234}

PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-2}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-"cosine"}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_STRATEGY=${SAVE_STRATEGY:-"epoch"}
EVALUATION_STRATEGY=${EVALUATION_STRATEGY:-"no"}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}

LOSS=${LOSS:-"gem"}
GEM_BETA=${GEM_BETA:-0.7}
GEM_H=${GEM_H:-"logsigmoid"}

REPORT_TO=${REPORT_TO:-"tensorboard"}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
BF16=${BF16:-True}
USE_FLASH_ATTN=${USE_FLASH_ATTN:-True}

mkdir -p "$OUTPUT_DIR"

TRAIN_ARGS=(
    --deepspeed "$DEEPSPEED_CONFIG"
    --seed "$SEED"
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --train_tokenized_file "$TRAIN_TOKENIZED_FILE"
    --output_dir "$OUTPUT_DIR"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --evaluation_strategy "$EVALUATION_STRATEGY"
    --save_strategy "$SAVE_STRATEGY"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --loss "$LOSS"
    --gem_beta "$GEM_BETA"
    --gem_h "$GEM_H"
    --learning_rate "$LEARNING_RATE"
    --lr_scheduler_type "$LR_SCHEDULER_TYPE"
    --warmup_ratio "$WARMUP_RATIO"
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --logging_steps "$LOGGING_STEPS"
    --report_to "$REPORT_TO"
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
    --overwrite_output_dir
    --bf16 "$BF16"
    --use_flash_attn "$USE_FLASH_ATTN"
)

if [[ -n "$TEST_TOKENIZED_FILE" ]]; then
    TRAIN_ARGS+=(--test_tokenized_file "$TEST_TOKENIZED_FILE")
fi

TRAIN_ARGS+=(--tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH")

deepspeed --num_gpus "$NUM_GPUS" --master_port "$MASTER_PORT" "$PYTHON_BIN" train.py "${TRAIN_ARGS[@]}" \
    2>&1 | tee "$OUTPUT_DIR/training.log"
