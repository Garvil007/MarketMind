"""Backtesting + training stack for MarketMind.

Turns historical OHLCV into:
  - features.py : a per-bar feature table mirroring scanner.py's indicators.
  - engine.py   : a trade simulator + performance metrics for the script signal.
  - dataset.py  : labeled tabular data (for the ML model) and instruction JSONL
                  (for LoRA/QLoRA LLM fine-tuning).
  - ml_model.py : a scikit-learn classifier trained on the tabular dataset.
  - llm_finetune.py : LoRA/QLoRA fine-tuning of a local Qwen/Llama on the JSONL.

The deterministic decision rule lives in marketmind.quant_signal and is reused
here so "what the script would have done" is defined in exactly one place.
"""
