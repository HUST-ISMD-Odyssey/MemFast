"""Local Hugging Face causal language models without a custom decoder."""

from pathlib import Path

import torch


def language_model_factory(config):
    """Select the upstream text-generation class for a recognized architecture."""
    import transformers

    if config.model_type in ("llama", "qwen2", "qwen3", "qwen3_5_text"):
        return transformers.AutoModelForCausalLM
    if config.model_type == "qwen3_5":
        # Official Qwen3.5 checkpoints wrap the hybrid text decoder and vision
        # encoder. Text-only calls omit pixel inputs and use the text decoder.
        if not hasattr(transformers, "AutoModelForImageTextToText"):
            raise ImportError("Qwen3.5 needs a transformers release providing its model classes")
        return transformers.AutoModelForImageTextToText
    raise ValueError(
        "supported local decoder types: llama (Llama 2/3), qwen2 (Qwen2/2.5), "
        "qwen3, qwen3_5_text, qwen3_5"
    )


def load_causal_lm(checkpoint, *, device="cpu", dtype=torch.float32):
    """Load a recognized local Llama or Qwen architecture without remote code.

    Returns ``(model, tokenizer)``. Attention, caching and generation remain
    implemented by transformers; Linear layers can then be converted normally.
    Qwen3.5 uses its upstream hybrid decoder, with the multimodal wrapper when
    present in the checkpoint. Text-only callers do not supply image inputs.
    Small digital/CIM architecture checks cover these paths on transformers
    5.8.0; this does not certify every model size or checkpoint in each family.
    """
    root = Path(checkpoint).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"expected a local Hugging Face checkpoint directory: {root}")
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise ImportError("language models require the optional transformers package") from exc
    config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if getattr(config, "quantization_config", None):
        raise ValueError(
            "load a floating-point checkpoint; packed quantized checkpoints need a decoder"
        )
    factory = language_model_factory(config)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    model = (
        factory.from_pretrained(
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
