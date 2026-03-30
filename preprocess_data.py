import json
import os
import random
from glob import glob
from multiprocessing import Pool

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch

from argparse import ArgumentParser
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

parser = ArgumentParser()
parser.add_argument(
    "--dataset_name_or_path",
    type=str,
    default="HuggingFaceH4/ultrafeedback_binarized",
    help="HuggingFace dataset name, or a local JSON/JSONL file or directory.",
)
parser.add_argument(
    "--dataset_format",
    type=str,
    default="auto",
    choices=["auto", "hf", "json"],
    help="How to interpret dataset_name_or_path. Use auto to infer local JSON/JSONL paths.",
)
parser.add_argument(
    "--json_field",
    type=str,
    default=None,
    help="Optional top-level field name when loading a local JSON object file.",
)
parser.add_argument(
    "--split",
    type=str,
    default="train",
    help="Split name for HuggingFace datasets. Ignored for local JSON/JSONL files.",
)
parser.add_argument(
    "--start",
    type=int,
    default=0,
)
parser.add_argument(
    "--end",
    type=int,
    default=None,
)
parser.add_argument(
    "--messages_key",
    type=str,
    default="messages",
    help="Column name that stores the ChatML message list.",
)
parser.add_argument(
    "--role_key",
    type=str,
    default="role",
    help="Key name for each message role.",
)
parser.add_argument(
    "--content_key",
    type=str,
    default="content",
    help="Key name for each message content.",
)
parser.add_argument(
    "--shuffle",
    action="store_true",
    help="Shuffle the dataset before applying start/end slicing.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed used when --shuffle is enabled.",
)
parser.add_argument(
    "--output_file",
    type=str,
    required=True,
)
parser.add_argument(
    "--tokenizer_name_or_path",
    type=str,
    required=True,
)
parser.add_argument("--max_seq_length", type=int, default=4096)
parser.add_argument("--preprocessing_num_workers", type=int, default=64)
parser.add_argument(
    "--disable_thinking",
    action="store_true",
    help="Disable tokenizer-specific thinking mode (for example Qwen3 enable_thinking=False).",
)
parser.add_argument(
    "--strip_think_tags",
    action="store_true",
    help="Remove literal <think> and </think> tags from message content before tokenization.",
)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
if getattr(tokenizer, "chat_template", None) is None:
    raise ValueError(
        f"Tokenizer {args.tokenizer_name_or_path} does not provide a chat template. "
        "Please use a chat/instruct tokenizer for ChatML data."
    )
print(f"load tokenizer from {args.tokenizer_name_or_path} done.")
max_seq_length = args.max_seq_length


def resolve_local_data_files(dataset_name_or_path):
    if os.path.isfile(dataset_name_or_path):
        return [dataset_name_or_path]
    if os.path.isdir(dataset_name_or_path):
        data_files = []
        for pattern in ("*.json", "*.jsonl"):
            data_files.extend(sorted(glob(os.path.join(dataset_name_or_path, pattern))))
        return data_files
    return None


def normalize_loaded_records(records, data_file):
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list):
        raise TypeError(
            f"Expected parsed data from {data_file} to be a list or dict, got {type(records)}."
        )
    normalized_records = []
    for record_idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(
                f"Record {record_idx} from {data_file} must be a dict, got {type(record)}."
            )
        normalized_records.append(record)
    return normalized_records


def extract_json_field(record_or_records, data_file):
    if args.json_field is None:
        return record_or_records
    if not isinstance(record_or_records, dict):
        raise TypeError(
            f"--json_field={args.json_field} requires the parsed object from {data_file} "
            f"to be a dict, got {type(record_or_records)}."
        )
    if args.json_field not in record_or_records:
        raise KeyError(
            f"Cannot find json field '{args.json_field}' in {data_file}. "
            f"Available keys: {list(record_or_records.keys())}"
        )
    return record_or_records[args.json_field]


def load_local_json_records(data_files):
    records = []
    for data_file in data_files:
        if data_file.endswith(".jsonl"):
            with open(data_file, "r", encoding="utf-8") as jsonl_reader:
                for line_idx, line in enumerate(jsonl_reader, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed_line = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Failed to parse JSONL line {line_idx} from {data_file}: {exc}"
                        ) from exc
                    extracted = extract_json_field(parsed_line, f"{data_file}:{line_idx}")
                    records.extend(
                        normalize_loaded_records(extracted, f"{data_file}:{line_idx}")
                    )
        else:
            with open(data_file, "r", encoding="utf-8") as json_reader:
                try:
                    parsed_json = json.load(json_reader)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Failed to parse JSON file {data_file}: {exc}"
                    ) from exc
            extracted = extract_json_field(parsed_json, data_file)
            records.extend(normalize_loaded_records(extracted, data_file))
    return records


def load_input_data():
    local_data_files = resolve_local_data_files(args.dataset_name_or_path)
    use_local_json = args.dataset_format == "json" or (
        args.dataset_format == "auto" and local_data_files
    )

    if use_local_json:
        if not local_data_files:
            raise ValueError(
                f"Cannot find local JSON/JSONL data under {args.dataset_name_or_path}."
            )
        if args.split not in ("", "train"):
            print(
                f"warning: --split={args.split} is ignored for local JSON/JSONL files."
            )
        input_data = load_local_json_records(local_data_files)
        data_source = f"local JSON/JSONL files: {', '.join(local_data_files)}"
    else:
        input_data = load_dataset(args.dataset_name_or_path)
        if args.split:
            input_data = input_data[args.split]
        data_source = f"HuggingFace dataset {args.dataset_name_or_path}, split={args.split}"

    if use_local_json:
        if args.shuffle:
            random.Random(args.seed).shuffle(input_data)
        end = len(input_data) if args.end is None else min(args.end, len(input_data))
        if args.start < 0 or args.start > end:
            raise ValueError(
                f"Invalid slice range: start={args.start}, end={end}, len={len(input_data)}."
            )
        input_data = input_data[args.start:end]
    else:
        if args.shuffle:
            input_data = input_data.shuffle(seed=args.seed)
        end = len(input_data) if args.end is None else min(args.end, len(input_data))
        if args.start < 0 or args.start > end:
            raise ValueError(
                f"Invalid slice range: start={args.start}, end={end}, len={len(input_data)}."
            )
        input_data = input_data.select(range(args.start, end))
    print(f"load input data from {data_source} done. len(input_data): {len(input_data)}")
    return input_data


def normalize_messages(example):
    if args.messages_key not in example:
        raise KeyError(
            f"Cannot find messages key '{args.messages_key}' in example. "
            f"Available keys: {list(example.keys())}"
        )
    messages = example[args.messages_key]
    if not isinstance(messages, list):
        raise TypeError(
            f"Expected '{args.messages_key}' to be a list, got {type(messages)}."
        )
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    normalized_messages = []
    for message_idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(
                f"Message at index {message_idx} must be a dict, got {type(message)}."
            )
        if args.role_key not in message or args.content_key not in message:
            raise KeyError(
                f"Message at index {message_idx} must contain keys "
                f"'{args.role_key}' and '{args.content_key}'."
            )
        role = message[args.role_key]
        content = message[args.content_key]
        if not isinstance(role, str):
            raise TypeError(
                f"Role at message index {message_idx} must be a string, got {type(role)}."
            )
        if not isinstance(content, str):
            raise TypeError(
                f"Content at message index {message_idx} must be a string, got {type(content)}."
            )
        if args.strip_think_tags:
            content = content.replace("<think>", "").replace("</think>", "")
        normalized_messages.append({"role": role, "content": content})
    return normalized_messages


def extract_input_ids(chat_template_output):
    if isinstance(chat_template_output, torch.Tensor):
        input_ids = chat_template_output
    elif hasattr(chat_template_output, "input_ids"):
        input_ids = chat_template_output["input_ids"]
    elif isinstance(chat_template_output, dict) and "input_ids" in chat_template_output:
        input_ids = chat_template_output["input_ids"]
    else:
        input_ids = chat_template_output

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids.to(dtype=torch.long)


def apply_chat_template_tensor(messages, add_generation_prompt=False):
    apply_kwargs = dict(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=add_generation_prompt,
    )
    if args.disable_thinking:
        apply_kwargs["enable_thinking"] = False
    chat_template_output = tokenizer.apply_chat_template(**apply_kwargs)
    return extract_input_ids(chat_template_output)


input_data = load_input_data()
if len(input_data) == 0:
    raise ValueError("No input examples selected after applying start/end.")


def encode_sft_example(example):
    """
    Encode a single ChatML example for SFT.
    The input example is expected to contain a messages list, where each message
    includes role/content fields.
    """
    messages = normalize_messages(example)
    input_ids = apply_chat_template_tensor(messages, add_generation_prompt=False)
    labels = input_ids.clone()
    # Mask all non-assistant segments so GEM/CE only trains on assistant responses.
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = apply_chat_template_tensor(
                    messages[:message_idx],
                    add_generation_prompt=False,
                ).shape[1]
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                message_end_idx = apply_chat_template_tensor(
                    messages[: message_idx + 1],
                    add_generation_prompt=True,
                ).shape[1]
            else:
                message_end_idx = apply_chat_template_tensor(
                    messages[: message_idx + 1],
                    add_generation_prompt=False,
                ).shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            if max_seq_length and message_end_idx >= max_seq_length:
                break
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten().tolist(),
        "labels": labels.flatten().tolist(),
        "attention_mask": attention_mask.flatten().tolist(),
    }


progress_bar = tqdm(total=len(input_data), desc="tokenizing")
os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
num_examples = 0
first_example = None
with open(args.output_file, "w", encoding="utf-8") as output_writer:
    if args.preprocessing_num_workers <= 1:
        iterator = (encode_sft_example(example) for example in input_data)
    else:
        process_pool = Pool(args.preprocessing_num_workers)
        iterator = process_pool.imap(encode_sft_example, input_data)
    try:
        for tokenized_example in iterator:
            if first_example is None:
                first_example = tokenized_example
            output_writer.write(json.dumps(tokenized_example) + "\n")
            num_examples += 1
            progress_bar.update(1)
    finally:
        if args.preprocessing_num_workers > 1:
            process_pool.close()
            process_pool.join()
        progress_bar.close()

assistant_tokens = sum(token != -100 for token in first_example["labels"])
print(
    f"save tokenized data to {args.output_file} done. "
    f"num_examples={num_examples}, "
    f"first_example_total_tokens={len(first_example['input_ids'])}, "
    f"first_example_assistant_tokens={assistant_tokens}"
)
