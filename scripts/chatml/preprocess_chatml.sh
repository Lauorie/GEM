#!/bin/bash

set -euo pipefail
set -x

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python
fi

RAW_TRAIN_FILE=${RAW_TRAIN_FILE:-"./data/chatml/train.json"}
RAW_EVAL_FILE=${RAW_EVAL_FILE:-""}

TOKENIZER_NAME_OR_PATH=${TOKENIZER_NAME_OR_PATH:-"meta-llama/Llama-3.1-8B-Instruct"}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
PREPROCESSING_NUM_WORKERS=${PREPROCESSING_NUM_WORKERS:-32}

TRAIN_OUTPUT_FILE=${TRAIN_OUTPUT_FILE:-"./data/chatml/train_tokenized.jsonl"}
EVAL_OUTPUT_FILE=${EVAL_OUTPUT_FILE:-"./data/chatml/eval_tokenized.jsonl"}

MESSAGES_KEY=${MESSAGES_KEY:-"messages"}
ROLE_KEY=${ROLE_KEY:-"role"}
CONTENT_KEY=${CONTENT_KEY:-"content"}
JSON_FIELD=${JSON_FIELD:-""}
DISABLE_THINKING=${DISABLE_THINKING:-1}

SHUFFLE_BEFORE_SPLIT=${SHUFFLE_BEFORE_SPLIT:-0}
SEED=${SEED:-42}

TRAIN_START=${TRAIN_START:-0}
TRAIN_END=${TRAIN_END:-""}

EVAL_START=${EVAL_START:-0}
EVAL_END=${EVAL_END:-""}

COMMON_ARGS=(
    --dataset_format json
    --tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH"
    --max_seq_length "$MAX_SEQ_LENGTH"
    --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS"
    --messages_key "$MESSAGES_KEY"
    --role_key "$ROLE_KEY"
    --content_key "$CONTENT_KEY"
)

if [[ -n "$JSON_FIELD" ]]; then
    COMMON_ARGS+=(--json_field "$JSON_FIELD")
fi

if [[ "$DISABLE_THINKING" == "1" ]]; then
    COMMON_ARGS+=(--disable_thinking)
fi

if [[ "$SHUFFLE_BEFORE_SPLIT" == "1" ]]; then
    COMMON_ARGS+=(--shuffle --seed "$SEED")
fi

mkdir -p "$(dirname "$TRAIN_OUTPUT_FILE")"

TRAIN_ARGS=(
    --dataset_name_or_path "$RAW_TRAIN_FILE"
    --output_file "$TRAIN_OUTPUT_FILE"
    --start "$TRAIN_START"
)

if [[ -n "$TRAIN_END" ]]; then
    TRAIN_ARGS+=(--end "$TRAIN_END")
fi

"$PYTHON_BIN" preprocess_data.py "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}"

if [[ -n "$RAW_EVAL_FILE" || -n "$EVAL_END" ]]; then
    EVAL_SOURCE=${RAW_EVAL_FILE:-"$RAW_TRAIN_FILE"}
    mkdir -p "$(dirname "$EVAL_OUTPUT_FILE")"

    EVAL_ARGS=(
        --dataset_name_or_path "$EVAL_SOURCE"
        --output_file "$EVAL_OUTPUT_FILE"
        --start "$EVAL_START"
    )

    if [[ -n "$EVAL_END" ]]; then
        EVAL_ARGS+=(--end "$EVAL_END")
    fi

    "$PYTHON_BIN" preprocess_data.py "${COMMON_ARGS[@]}" "${EVAL_ARGS[@]}"
else
    echo "RAW_EVAL_FILE and EVAL_END are both empty, skip eval preprocessing."
fi
