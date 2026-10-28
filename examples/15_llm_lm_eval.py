"""Load a Hugging Face causal LM, convert it and run language evaluation."""

import math
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from memfast import Config, Nonidealities, Simulator, convert_model

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-0.6B"  # Or a local Hugging Face checkpoint directory.
DTYPE = torch.float32
TASKS = ("hellaswag",)
TEXT_FILE = None  # Path to a UTF-8 corpus for perplexity.
PROMPT = None  # Optional text for greedy generation.
MAX_NEW_TOKENS = 32
BATCH_SIZE = 1
MAX_LENGTH = 256
STRIDE = 128
NUM_FEWSHOT = 0
INCLUDE = None
EXCLUDE = ("lm_head",)  # Keep the language head in software.
CONFIG = Config(
    activation_bits=8,
    weight_bits=8,
    input_slice=(1,) * 8,
    weight_slice=(1,) * 8,
    array_size=(64, 64),
    rows_per_read=64,
    input_quant_gran=(1, None),
    weight_quant_gran=(None, 1),
    adc_bits=6,
    bw_e=None,
    nonidealities=Nonidealities(write_variation=0.02),
)


def perplexity_windows(input_ids, *, max_length=256, stride=128):
    """Use overlapping context and score each next-token target exactly once."""
    if not 1 <= stride < max_length:
        raise ValueError("stride must be positive and smaller than max_length")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
        raise ValueError("perplexity requires one sequence with at least two tokens")
    previous_end = 0
    for begin in range(0, input_ids.shape[1] - 1, stride):
        end = min(begin + max_length, input_ids.shape[1])
        tokens = input_ids[:, begin:end]
        labels = tokens.clone()
        labels[:, : max(0, previous_end - begin)] = -100
        yield tokens, labels
        previous_end = end
        if end == input_ids.shape[1]:
            break


@torch.inference_mode()
def perplexity(model, input_ids, *, device, max_length=256, stride=128):
    total_loss, total_targets = 0.0, 0
    for tokens, labels in perplexity_windows(input_ids, max_length=max_length, stride=stride):
        tokens, labels = tokens.to(device), labels.to(device)
        logits = model(input_ids=tokens, use_cache=False).logits[:, :-1].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite language-model logits")
        targets = labels[:, 1:]
        total_loss += F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
            ignore_index=-100,
        ).item()
        total_targets += (targets != -100).sum().item()
    mean = total_loss / total_targets
    return dict(
        target_tokens=total_targets,
        negative_log_likelihood=mean,
        perplexity=math.exp(mean) if mean < 709 else float("inf"),
    )


def harness_evaluate(
    model,
    tokenizer,
    *,
    tasks,
    device,
    batch_size=1,
    max_length=256,
    num_fewshot=0,
):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    adapter = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    results = simple_evaluate(
        model=adapter,
        tasks=list(tasks),
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        random_seed=135,
        numpy_random_seed=135,
        torch_random_seed=135,
        fewshot_random_seed=135,
        log_samples=False,
    )
    return results["results"]


def main():
    torch.manual_seed(135)
    architecture = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    model_class = AutoModelForCausalLM
    if architecture.model_type == "qwen3_5":
        # The upstream multimodal checkpoint wraps its text decoder.
        from transformers import AutoModelForImageTextToText

        model_class = AutoModelForImageTextToText
    model = (
        model_class.from_pretrained(
            MODEL_ID,
            config=architecture,
            torch_dtype=DTYPE,
            attn_implementation="eager",
            trust_remote_code=False,
        )
        .to(DEVICE)
        .eval()
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    simulator = Simulator(CONFIG, device=DEVICE)
    include = INCLUDE
    if include is None and architecture.model_type == "qwen3_5":
        include = ("model.language_model",)
    model = convert_model(model, simulator, include=include, exclude=EXCLUDE, inplace=True)
    results = {}
    if TASKS:
        results["tasks"] = harness_evaluate(
            model,
            tokenizer,
            tasks=TASKS,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            num_fewshot=NUM_FEWSHOT,
        )
        print(results["tasks"])
    if TEXT_FILE is not None:
        text = Path(TEXT_FILE).read_text(encoding="utf-8")
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        results["text"] = perplexity(
            model,
            input_ids,
            device=DEVICE,
            max_length=MAX_LENGTH,
            stride=STRIDE,
        )
        print(results["text"])
    if PROMPT is not None:
        inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        results["generation"] = tokenizer.decode(
            generated[0, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        print(results["generation"])
    return results


if __name__ == "__main__":
    main()
