"""Local Hugging Face causal language models without a custom decoder."""

from pathlib import Path

import torch


def load_causal_lm(checkpoint, *, device="cpu", dtype=torch.float32):
    """Load a local Llama or Qwen checkpoint and tokenizer, without remote code.

    Returns ``(model, tokenizer)``. Attention, caching and generation remain
    implemented by transformers; Linear layers can then be converted normally.
    """
    root = Path(checkpoint).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"expected a local Hugging Face checkpoint directory: {root}")
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("language models require the optional transformers package") from exc
    config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if config.model_type not in ("llama", "qwen2", "qwen3"):
        raise ValueError("supported checkpoint model types: llama, qwen2, qwen3")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    model = (
        AutoModelForCausalLM.from_pretrained(
            root,
            config=config,
            torch_dtype=dtype,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    return model, tokenizer
