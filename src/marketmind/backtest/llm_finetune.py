"""LoRA / QLoRA fine-tuning of a local Qwen or Llama on the scanner SFT data.

Consumes the chat JSONL from dataset.save_dataset (sft.jsonl) and trains a small
LoRA adapter on top of a base instruct model. QLoRA (4-bit base + LoRA) is the
default so it fits on a single consumer GPU.

Heavy deps (transformers, peft, trl, bitsandbytes, datasets, accelerate, torch)
live in requirements-train.txt and are imported lazily — importing this module
costs nothing until you actually train. GPU + those deps required to run; this
will not train on CPU-only in any reasonable time.

Run:  python scripts/train_llm.py --jsonl data/training/sft.jsonl
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FinetuneConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"  # or e.g. meta-llama/Llama-3.2-3B-Instruct
    output_dir: str = "data/models/llm_adapter"
    use_qlora: bool = True          # 4-bit base (QLoRA); False = fp16 LoRA
    epochs: float = 3.0
    batch_size: int = 2
    grad_accum: int = 8
    learning_rate: float = 2e-4
    max_seq_len: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


# LoRA target modules that cover both Qwen2 and Llama attention/MLP projections.
_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def train_lora(jsonl_path: str | Path, cfg: FinetuneConfig | None = None) -> dict[str, Any]:
    """Fine-tune a LoRA/QLoRA adapter on the chat JSONL and save it.

    Args:
        jsonl_path: path to sft.jsonl ({"messages": [...]} per line).
        cfg: FinetuneConfig (base model, QLoRA toggle, hyperparameters).

    Returns:
        {"adapter_dir": str, "base_model": str, "n_examples": int}
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = cfg or FinetuneConfig()
    jsonl_path = str(jsonl_path)
    if not Path(jsonl_path).exists():
        raise FileNotFoundError(f"SFT data not found: {jsonl_path} (run build_dataset first).")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Base model (4-bit for QLoRA, else fp16) ------------------------
    model_kwargs: dict[str, Any] = {"device_map": "auto"}
    if cfg.use_qlora:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
    if cfg.use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=_LORA_TARGETS,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # --- Data: render chat messages to text via the model's chat template
    raw = load_dataset("json", data_files=jsonl_path, split="train")

    def _format(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = raw.map(_format, remove_columns=raw.column_names)

    # --- Trainer (trl SFTTrainer) ---------------------------------------
    from trl import SFTConfig, SFTTrainer

    sft_cfg = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        max_seq_length=cfg.max_seq_len,
        logging_steps=10,
        save_strategy="epoch",
        bf16=cfg.use_qlora,
        dataset_text_field="text",
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()

    adapter_dir = str(Path(cfg.output_dir))
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return {"adapter_dir": adapter_dir, "base_model": cfg.base_model, "n_examples": len(dataset)}


def predict_from_tech(tech: dict, adapter_dir: str | Path = "data/models/llm_adapter") -> str | None:
    """Load base + LoRA adapter and generate a BUY/HOLD/SELL line for one tech dict.

    Returns None if the adapter hasn't been trained yet. Intended for offline
    evaluation, not the live Groq-hosted pipeline.
    """
    from pathlib import Path as _P

    if not _P(adapter_dir).exists():
        return None

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from marketmind.backtest.dataset import _SYSTEM, _user_prompt

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    base = AutoModelForCausalLM.from_pretrained(
        FinetuneConfig().base_model, device_map="auto", torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _user_prompt(tech)},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=80, do_sample=False)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
