#!/bin/bash

set -euo pipefail
set -x

export RAW_TRAIN_FILE="${RAW_TRAIN_FILE:-/root/app/merged_weak.json}"
export RAW_EVAL_FILE="${RAW_EVAL_FILE:-""}"

export TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-Qwen/Qwen3-8B}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
export PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-32}"

export TRAIN_OUTPUT_FILE="${TRAIN_OUTPUT_FILE:-./data/qwen3_merged_weak/train_tokenized.jsonl}"
export EVAL_OUTPUT_FILE="${EVAL_OUTPUT_FILE:-./data/qwen3_merged_weak/eval_tokenized.jsonl}"

export MESSAGES_KEY="${MESSAGES_KEY:-messages}"
export ROLE_KEY="${ROLE_KEY:-role}"
export CONTENT_KEY="${CONTENT_KEY:-content}"
export JSON_FIELD="${JSON_FIELD:-""}"

export SHUFFLE_BEFORE_SPLIT="${SHUFFLE_BEFORE_SPLIT:-0}"
export SEED="${SEED:-42}"

export TRAIN_START="${TRAIN_START:-0}"
export TRAIN_END="${TRAIN_END:-""}"
export EVAL_START="${EVAL_START:-0}"
export EVAL_END="${EVAL_END:-""}"

bash scripts/chatml/preprocess_chatml.sh
