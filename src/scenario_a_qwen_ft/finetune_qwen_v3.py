from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


@dataclass
class QwenFTConfig:
    project_root: str
    model_name: str
    train_file: str
    val_file: str
    output_dir: str
    max_length: int = 4608
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    logging_steps: int = 10
    save_total_limit: int = 2
    seed: int = 20260413
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_steps: int = -1
    target_modules: list[str] | None = None
    auto_resume: bool = True


class SupervisedDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _load_config(config_path: Path) -> QwenFTConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return QwenFTConfig(**payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_record(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    max_length: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    answer_ids = full_ids[len(prompt_ids) :]
    if not answer_ids:
        answer_text = messages[-1]["content"] + tokenizer.eos_token
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    raw_total_tokens = len(prompt_ids) + len(answer_ids)
    truncated = raw_total_tokens > max_length
    kept_prompt_tokens = min(len(prompt_ids), max_length - len(answer_ids))
    if kept_prompt_tokens < 0:
        kept_prompt_tokens = 0
        answer_ids = answer_ids[:max_length]
    prompt_ids = prompt_ids[:kept_prompt_tokens]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)
    stats = {
        "raw_total_tokens": raw_total_tokens,
        "prompt_tokens": len(prompt_ids),
        "answer_tokens": len(answer_ids),
        "final_tokens": len(input_ids),
        "truncated": truncated,
    }
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }, stats


def _build_dataset(
    *,
    rows: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int,
    sample_limit: int | None,
) -> tuple[Dataset, dict[str, Any], list[dict[str, Any]]]:
    if sample_limit is not None:
        rows = rows[:sample_limit]
    encoded_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    for row in rows:
        encoded, stats = _format_record(tokenizer, row["messages"], max_length)
        encoded_rows.append(encoded)
        stats_rows.append(
            {
                "question_preview": row["messages"][1]["content"][:120],
                **stats,
            }
        )
    dataset = Dataset.from_list(encoded_rows)
    token_lengths = [item["final_tokens"] for item in stats_rows]
    trunc_count = sum(1 for item in stats_rows if item["truncated"])
    summary = {
        "row_count": len(stats_rows),
        "truncated_count": trunc_count,
        "truncated_ratio": round((trunc_count / len(stats_rows)) if stats_rows else 0.0, 4),
        "max_final_tokens": max(token_lengths) if token_lengths else 0,
        "mean_final_tokens": round(sum(token_lengths) / len(token_lengths), 2) if token_lengths else 0.0,
    }
    return dataset, summary, stats_rows


def _load_model_and_tokenizer(config: QwenFTConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules
        or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    return model, tokenizer


def run_training(config_path: Path) -> dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True

    config = _load_config(config_path)
    project_root = Path(config.project_root).resolve()
    train_file = project_root / config.train_file
    val_file = project_root / config.val_file
    output_dir = project_root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(config.seed)

    model, tokenizer = _load_model_and_tokenizer(config)
    train_rows = _read_jsonl(train_file)
    val_rows = _read_jsonl(val_file)
    train_dataset, train_summary, train_stats = _build_dataset(
        rows=train_rows,
        tokenizer=tokenizer,
        max_length=config.max_length,
        sample_limit=config.max_train_samples,
    )
    val_dataset, val_summary, val_stats = _build_dataset(
        rows=val_rows,
        tokenizer=tokenizer,
        max_length=config.max_length,
        sample_limit=config.max_eval_samples,
    )

    collator = SupervisedDataCollator(tokenizer.pad_token_id)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        do_train=True,
        do_eval=True,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=config.save_total_limit,
        fp16=True,
        bf16=False,
        report_to=[],
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        seed=config.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        tf32=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    resume_checkpoint = None
    checkpoints_dir = output_dir / "checkpoints"
    if config.auto_resume and checkpoints_dir.exists():
        resume_checkpoint = get_last_checkpoint(str(checkpoints_dir))
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))

    train_loss = float(train_result.training_loss) if train_result.training_loss is not None else math.nan
    summary = {
        "config": asdict(config),
        "resume_checkpoint": resume_checkpoint,
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "train_summary": train_summary,
        "val_summary": val_summary,
        "train_loss": train_loss,
        "eval_metrics": eval_metrics,
    }
    _write_json(output_dir / "train_summary.json", summary)
    _write_json(output_dir / "train_truncation_stats.json", train_stats[:200])
    _write_json(output_dir / "val_truncation_stats.json", val_stats[:200])
    _write_markdown(
        output_dir / "train_report.md",
        [
            "# Qwen FT Main Experiment",
            "",
            f"- model: `{config.model_name}`",
            f"- train_rows: `{len(train_dataset)}`",
            f"- val_rows: `{len(val_dataset)}`",
            f"- max_length: `{config.max_length}`",
            f"- epochs: `{config.num_train_epochs}`",
            f"- train_loss: `{train_loss:.4f}`",
            f"- eval_loss: `{eval_metrics.get('eval_loss', math.nan):.4f}`",
            f"- truncated_train_ratio: `{train_summary['truncated_ratio']}`",
            f"- truncated_val_ratio: `{val_summary['truncated_ratio']}`",
        ],
    )
    return summary
