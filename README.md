# GEM 训练指南（中文）

本仓库提供了 GEM（Preserving Diversity in Supervised Fine-tuning of Large Language Models）的 PyTorch 实现。GEM 可以直接替换标准交叉熵（CE）损失，用于监督微调（SFT），以减轻过拟合并提升生成多样性。

你的使用场景是：
- 数据是私有数据；
- 原始格式是 ChatML JSON；
- 希望在单机 8 张 A800（80GB）上训练；
- 希望尽量少改代码，直接跑通 GEM。

这次已经把仓库补成支持下面这条流程：

1. 本地 ChatML JSON/JSONL -> 运行 `preprocess_data.py`
2. 得到训练所需的 `input_ids` / `labels` / `attention_mask` JSONL
3. 使用 `train.py` + DeepSpeed 在 8 卡上训练 GEM

---

## 1. 仓库当前训练流程

当前仓库的核心训练入口是：

- `preprocess_data.py`：把原始对话数据转换为 tokenized JSONL
- `train.py`：读取 tokenized JSONL，使用 CE 或 GEM 进行训练
- `sft_trainer.py` / `sft_trainer_v2.py`：具体实现训练逻辑
- `utils/gem_triton_ops.py` / `utils/gem_triton_loss.py`：Triton 版 GEM 实现

也就是说，**GEM 本身不要求你的原始数据必须是某种固定文本格式**；它真正需要的是训练阶段读到的 `input_ids` 和 `labels`。因此最关键的事情是把你的 ChatML 数据正确预处理好。

---

## 2. 支持的数据格式

### 2.1 推荐的原始数据格式

你给出的数据格式是支持的，推荐如下：

```json
[
  {
    "messages": [
      {"role": "system", "content": "系统提示词"},
      {"role": "user", "content": "用户输入"},
      {"role": "assistant", "content": "助手回复"}
    ]
  },
  {
    "messages": [
      {"role": "user", "content": "请介绍一下 GEM"},
      {"role": "assistant", "content": "GEM 是一种可替换 CE 的训练目标。"}
    ]
  }
]
```

也支持 JSONL，每行一条样本，例如：

```json
{"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好，有什么可以帮你？"}]}
{"messages": [{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": "2"}]}
```

### 2.2 字段要求

每条样本默认需要：

- 顶层字段：`messages`
- 每条消息字段：
  - `role`
  - `content`

默认支持的 role 包括：

- `system`
- `user`
- `assistant`

训练时只会对 `assistant` 对应的 token 计算 loss，其他角色会被自动 mask 成 `-100`。

### 2.3 自定义字段名

如果你的数据不是 `messages/role/content` 这组键名，也可以通过以下参数覆盖：

- `--messages_key`
- `--role_key`
- `--content_key`

---

## 3. 环境准备

建议使用 Python 3.10。

```bash
conda create -n gem python=3.10
conda activate gem
pip install -r requirements.txt
```

`requirements.txt` 中的版本与论文实验接近。如果你后续希望使用更新版 Transformers，也是可行的，但需要自行验证兼容性。

### 3.1 Triton 版 GEM

仓库也提供 Triton 版 GEM 实现，速度通常更好：

```bash
python3 train.py --loss gem_triton
```

Triton 相关说明见：

- `utils/README.md`

---

## 4. ChatML 数据预处理

这次已经补充了一个适合本地私有数据的脚本：

- `scripts/chatml/preprocess_chatml.sh`

它本质上调用的是 `preprocess_data.py`，但把常用参数整理好了，方便直接使用。

### 4.1 最简单用法：训练集和验证集分别是两个文件

假设你的目录如下：

```text
./data/chatml/train.json
./data/chatml/eval.json
```

你可以这样运行：

```bash
RAW_TRAIN_FILE=./data/chatml/train.json \
RAW_EVAL_FILE=./data/chatml/eval.json \
TOKENIZER_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct \
MAX_SEQ_LENGTH=4096 \
PREPROCESSING_NUM_WORKERS=32 \
bash scripts/chatml/preprocess_chatml.sh
```

预处理后默认会生成：

```text
./data/chatml/train_tokenized.jsonl
./data/chatml/eval_tokenized.jsonl
```

### 4.2 单个 JSON 文件切分训练集 / 验证集

如果你的数据全部在一个文件里，例如：

```text
./data/chatml/all_data.json
```

可以先打乱再切分：

```bash
RAW_TRAIN_FILE=./data/chatml/all_data.json \
SHUFFLE_BEFORE_SPLIT=1 \
SEED=42 \
TRAIN_START=0 \
TRAIN_END=9800 \
EVAL_START=9800 \
EVAL_END=10000 \
bash scripts/chatml/preprocess_chatml.sh
```

说明：
- `SHUFFLE_BEFORE_SPLIT=1`：先打乱样本
- `TRAIN_START/TRAIN_END`：训练集切片范围
- `EVAL_START/EVAL_END`：验证集切片范围
- 这里没有设置 `RAW_EVAL_FILE`，脚本会自动从同一个源文件切验证集

### 4.3 如果你的本地文件是 JSONL

直接把 `RAW_TRAIN_FILE` / `RAW_EVAL_FILE` 指向 `.jsonl` 文件即可，无需额外改动。

### 4.4 如果你的 JSON 文件是带顶层字段的对象

例如：

```json
{
  "data": [
    {"messages": [...]},
    {"messages": [...]} 
  ]
}
```

可以额外指定：

```bash
JSON_FIELD=data \
RAW_TRAIN_FILE=./data/chatml/train.json \
bash scripts/chatml/preprocess_chatml.sh
```

### 4.5 如果字段名不是 `messages/role/content`

例如你的样本是：

```json
[
  {
    "dialog": [
      {"speaker": "user", "text": "你好"},
      {"speaker": "assistant", "text": "你好"}
    ]
  }
]
```

可这样指定：

```bash
RAW_TRAIN_FILE=./data/chatml/train.json \
MESSAGES_KEY=dialog \
ROLE_KEY=speaker \
CONTENT_KEY=text \
bash scripts/chatml/preprocess_chatml.sh
```

---

## 5. 直接使用 `preprocess_data.py`

如果你不想用 shell 脚本，也可以直接调用 Python：

```bash
python3 preprocess_data.py \
  --dataset_format json \
  --dataset_name_or_path ./data/chatml/train.json \
  --tokenizer_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --max_seq_length 4096 \
  --preprocessing_num_workers 32 \
  --output_file ./data/chatml/train_tokenized.jsonl
```

如果要处理验证集：

```bash
python3 preprocess_data.py \
  --dataset_format json \
  --dataset_name_or_path ./data/chatml/eval.json \
  --tokenizer_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --max_seq_length 4096 \
  --preprocessing_num_workers 32 \
  --output_file ./data/chatml/eval_tokenized.jsonl
```

### 5.1 `preprocess_data.py` 新增/重要参数

- `--dataset_format {auto,hf,json}`
  - `auto`：自动判断本地 JSON/JSONL
  - `hf`：强制按 HuggingFace 数据集名称处理
  - `json`：强制按本地 JSON/JSONL 处理
- `--json_field`
  - 用于读取顶层 JSON 对象中的某个字段
- `--messages_key`
- `--role_key`
- `--content_key`
- `--shuffle`
- `--seed`

### 5.2 预处理脚本现在的行为

当前会：

- 校验每条样本是否存在消息列表；
- 校验每条消息是否有 role/content；
- 使用 tokenizer 的 `apply_chat_template` 完成拼接与分词；
- 自动把非 assistant 部分的 labels 置为 `-100`；
- 边处理边写 JSONL，避免大数据集时占用过多内存；
- 默认不再把你的原始样本内容完整打印到日志中，减少私有数据泄露风险。

---

## 6. 8 × A800（80GB）训练 GEM

这次新增了一个专门面向单机 8 卡训练的脚本：

- `scripts/chatml/train_gem_8xa800.sh`

### 6.1 最简单训练命令

在你已经得到 tokenized 文件后，直接运行：

```bash
TRAIN_TOKENIZED_FILE=./data/chatml/train_tokenized.jsonl \
TEST_TOKENIZED_FILE=./data/chatml/eval_tokenized.jsonl \
MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B \
TOKENIZER_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct \
bash scripts/chatml/train_gem_8xa800.sh
```

### 6.2 脚本默认配置

默认脚本等价于：

- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- `NUM_GPUS=8`
- `DEEPSPEED_CONFIG=scripts/zero3.json`
- `LOSS=gem`
- `GEM_BETA=0.7`
- `GEM_H=logsigmoid`
- `PER_DEVICE_TRAIN_BATCH_SIZE=2`
- `GRADIENT_ACCUMULATION_STEPS=8`
- `BF16=True`
- `GRADIENT_CHECKPOINTING=True`

对于 8B 量级模型，这是一套相对稳妥的起点；你可以再按吞吐和显存情况微调。

### 6.3 推荐起始配置（8 × A800, 80GB）

如果你测试的是 7B/8B 级别模型，建议先从下面的量级开始：

```bash
TRAIN_TOKENIZED_FILE=./data/chatml/train_tokenized.jsonl TEST_TOKENIZED_FILE=./data/chatml/eval_tokenized.jsonl MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B TOKENIZER_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct PER_DEVICE_TRAIN_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=8 NUM_TRAIN_EPOCHS=3 LOSS=gem bash scripts/chatml/train_gem_8xa800.sh
```

说明：
- 实际训练序列长度不是在训练脚本里控制，而是在预处理阶段由 `MAX_SEQ_LENGTH` 决定；
- 所以你只需要确保预处理时的 `MAX_SEQ_LENGTH` 与你的训练计划一致。

### 6.4 如果你想减小显存压力

优先尝试：

1. 减小 `MAX_SEQ_LENGTH`
2. 减小 `PER_DEVICE_TRAIN_BATCH_SIZE`
3. 增大 `GRADIENT_ACCUMULATION_STEPS`
4. 保持 `zero3 + bf16 + gradient_checkpointing`

例如：

```bash
TRAIN_TOKENIZED_FILE=./data/chatml/train_tokenized.jsonl MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B TOKENIZER_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct PER_DEVICE_TRAIN_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=16 bash scripts/chatml/train_gem_8xa800.sh
```

### 6.5 如果你想用 Triton GEM

直接覆盖 loss：

```bash
TRAIN_TOKENIZED_FILE=./data/chatml/train_tokenized.jsonl MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B TOKENIZER_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct LOSS=gem_triton bash scripts/chatml/train_gem_8xa800.sh
```

---

## 7. 训练入口 `train.py` 的重要变化

为了更好支持 ChatML 数据，这次还补了一个能力：

- `train.py` 现在支持 `--tokenizer_name_or_path`

这意味着你可以使用：

- 模型权重：`MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B`
- 聊天模板 tokenizer：`TOKENIZER_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct`

这对于 ChatML / instruct 数据通常更合理，因为 `apply_chat_template` 依赖 tokenizer 中的聊天模板定义。

如果不传 `--tokenizer_name_or_path`，训练仍会回退到原来的行为：

- 默认使用 `model_name_or_path` 作为 tokenizer 路径。

---

## 8. 你最可能直接复制的完整流程

下面是一套最适合你当前需求的完整示例。

### 第一步：准备原始数据

例如：

```text
./data/chatml/train.json
./data/chatml/eval.json
```

内容格式：

```json
[
  {
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手。"},
      {"role": "user", "content": "介绍一下 GEM。"},
      {"role": "assistant", "content": "GEM 是一种用于监督微调的训练目标。"}
    ]
  }
]
```

### 第二步：预处理

```bash
RAW_TRAIN_FILE=./data/chatml/train.json RAW_EVAL_FILE=./data/chatml/eval.json TOKENIZER_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct MAX_SEQ_LENGTH=4096 PREPROCESSING_NUM_WORKERS=32 bash scripts/chatml/preprocess_chatml.sh
```

### 第三步：8 卡 GEM 训练

```bash
TRAIN_TOKENIZED_FILE=./data/chatml/train_tokenized.jsonl TEST_TOKENIZED_FILE=./data/chatml/eval_tokenized.jsonl MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B TOKENIZER_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct PER_DEVICE_TRAIN_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=8 NUM_TRAIN_EPOCHS=3 LOSS=gem bash scripts/chatml/train_gem_8xa800.sh
```

---

## 9. 常见问题

### 9.1 报错：tokenizer 没有 chat template

说明你传入的 tokenizer 不是聊天模型对应的 tokenizer，或者模型仓库里没有定义 chat template。

解决方式：
- 换成 instruct/chat 版本的 tokenizer；
- 显式传 `TOKENIZER_NAME_OR_PATH`；
- 确保该 tokenizer 支持 `apply_chat_template`。

### 9.2 报错：找不到 `messages`

说明你的字段名不是默认的 `messages`。

解决方式：
- 使用 `MESSAGES_KEY`、`ROLE_KEY`、`CONTENT_KEY` 覆盖默认字段名。

### 9.3 显存不够

优先调整：
- `MAX_SEQ_LENGTH`
- `PER_DEVICE_TRAIN_BATCH_SIZE`
- `GRADIENT_ACCUMULATION_STEPS`
- `DEEPSPEED_CONFIG=scripts/zero3.json`

### 9.4 日志里的 `ce_loss` 和总 `loss` 不完全一致

这是仓库原本就有的现象。`ce_loss` 更像一个诊断指标，而真正优化的 `loss` 取决于你使用的是 CE 还是 GEM。

---

## 10. 原有示例脚本

仓库中仍保留了原始示例：

- `scripts/llama3.1/tokenize_data.sh`
- `scripts/llama3.1/train_gem_ultrafeedback.sh`
- `scripts/qwen2.5/tokenize_data.sh`
- `scripts/qwen2.5/train_gem_numina.sh`

它们主要用于公开数据集示例；如果你要跑私有 ChatML 数据，优先使用：

- `scripts/chatml/preprocess_chatml.sh`
- `scripts/chatml/train_gem_8xa800.sh`

---

## 11. 引用

如果这个仓库对你的研究或项目有帮助，可以引用 GEM 论文：

```bibtex
@inproceedings{li2025preserving,
  title={Preserving Diversity in Supervised Fine-Tuning of Large Language Models},
  author={Ziniu Li and Congliang Chen and Tian Xu and Zeyu Qin and Jiancong Xiao and Zhi-Quan Luo and Ruoyu Sun},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=NQEe7B7bSw}
}
```

此前版本也可参考：

```bibtex
@article{li2024entropic,
  title={Entropic Distribution Matching in Supervised Fine-tuning of LLMs: Less Overfitting and Better Diversity},
  author={Li, Ziniu and Chen, Congliang and Xu, Tian and Qin, Zeyu and Xiao, Jiancong and Sun, Ruoyu and Luo, Zhi-Quan},
  journal={arXiv preprint arXiv:2408.16673},
  year={2024}
}
```
