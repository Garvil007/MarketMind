"""Fine-tune a LoRA/QLoRA adapter (Qwen/Llama) on the scanner SFT data.

Reads data/training/sft.jsonl and trains a small adapter saved under
data/models/llm_adapter. Requires a GPU and the training extras:
  pip install -r requirements-train.txt

Run:
  python scripts/build_dataset.py                          # first
  python scripts/train_llm.py
  python scripts/train_llm.py --base meta-llama/Llama-3.2-3B-Instruct --no-qlora
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketmind.backtest.llm_finetune import FinetuneConfig, train_lora  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA/QLoRA fine-tune on scanner SFT data.")
    p.add_argument("--jsonl", default="data/training/sft.jsonl")
    p.add_argument("--base", default=FinetuneConfig.base_model)
    p.add_argument("--out", default=FinetuneConfig.output_dir)
    p.add_argument("--epochs", type=float, default=FinetuneConfig.epochs)
    p.add_argument("--no-qlora", action="store_true", help="full fp16 LoRA instead of 4-bit QLoRA")
    args = p.parse_args()

    if not Path(args.jsonl).exists():
        print(f"SFT data not found: {args.jsonl}\nRun: python scripts/build_dataset.py")
        return

    cfg = FinetuneConfig(
        base_model=args.base, output_dir=args.out,
        epochs=args.epochs, use_qlora=not args.no_qlora,
    )
    print(f"Fine-tuning {cfg.base_model}  (QLoRA={cfg.use_qlora}, epochs={cfg.epochs})...")
    info = train_lora(args.jsonl, cfg)
    print(f"Done. Adapter -> {info['adapter_dir']}  ({info['n_examples']} examples)")


if __name__ == "__main__":
    main()
