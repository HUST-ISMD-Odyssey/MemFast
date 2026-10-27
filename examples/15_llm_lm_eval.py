"""Evaluate local language checkpoints with visible CIM conversion and lm-eval."""

import hashlib
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from memfast import AnalogMatrix, convert_model
from memfast.models import load_causal_lm

if __package__:
    from .common import parser, positive, report, setup, simulator
else:
    from common import parser, positive, report, setup, simulator


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def text_conversion_include(model, include, analog_head=False):
    if include is None and model.config.model_type == "qwen3_5":
        return ["model.language_model", *(["lm_head"] if analog_head else [])]
    return include


def local_checkpoint(path):
    root = Path(path).expanduser().resolve()
    if (root / "config.json").is_file():
        return root
    reference = root / "refs" / "main"
    if reference.is_file():
        candidate = root / "snapshots" / reference.read_text(encoding="utf-8").strip()
        if (candidate / "config.json").is_file():
            return candidate
    candidates = sorted((root / "snapshots").glob("*/config.json"))
    if len(candidates) == 1:
        return candidates[0].parent
    raise ValueError(f"choose a local checkpoint snapshot containing config.json: {root}")


def smoke_model(family, device, dtype=torch.float32):
    """Build a tiny upstream decoder and tokenizer without pretrained downloads."""
    try:
        import transformers
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
    except ImportError as error:
        raise ImportError(
            "language-model smoke runs require transformers and tokenizers"
        ) from error
    config_type = {
        "llama": "LlamaConfig",
        "qwen2": "Qwen2Config",
        "qwen3": "Qwen3Config",
        "qwen3_5_text": "Qwen3_5TextConfig",
    }[family]
    hybrid = (
        dict(
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            head_dim=8,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            rope_parameters=dict(
                rope_type="default",
                rope_theta=10000.0,
                partial_rotary_factor=1.0,
                mrope_section=[1, 1, 2],
                mrope_interleaved=True,
            ),
        )
        if family == "qwen3_5_text"
        else dict(num_hidden_layers=1)
    )
    config = getattr(transformers, config_type)(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        **hybrid,
    )
    config._attn_implementation = "eager"
    model = (
        transformers.AutoModelForCausalLM.from_config(config).to(device=device, dtype=dtype).eval()
    )
    words = ["<pad>", "<bos>", "<eos>", "<unk>"] + [f"word{index}" for index in range(28)]
    backend = Tokenizer(
        WordLevel({word: index for index, word in enumerate(words)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    return model, tokenizer


def perplexity_windows(input_ids, *, max_length=256, stride=128):
    """Yield identical context windows and masks for serial or batched evaluation."""
    if not 1 <= stride < max_length:
        raise ValueError("stride must be positive and smaller than max-length")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
        raise ValueError("perplexity requires one sequence containing at least two tokens")
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
def perplexity(model, input_ids, *, max_length=256, stride=128, device="cpu"):
    """Use overlapping context while scoring every next-token target exactly once."""
    total_loss, total_targets = 0.0, 0
    started = time.perf_counter()
    for tokens, labels in perplexity_windows(input_ids, max_length=max_length, stride=stride):
        tokens, labels = tokens.to(device), labels.to(device)
        logits = model(input_ids=tokens, use_cache=False).logits[:, :-1].float()
        targets = labels[:, 1:]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite language-model logits")
        count = int((targets != -100).sum())
        total_loss += float(
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
                ignore_index=-100,
            )
        )
        total_targets += count
    if total_targets != input_ids.shape[1] - 1:
        raise AssertionError("perplexity target coverage mismatch")
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    mean = total_loss / total_targets
    return dict(
        target_tokens=total_targets,
        negative_log_likelihood=mean,
        perplexity=math.exp(mean) if mean < 709 else None,
        seconds=time.perf_counter() - started,
    )


def in_memory_task():
    from datasets import Dataset
    from lm_eval.api.task import ConfigurableTask

    class SmokeTask(ConfigurableTask):
        def download(self, *args, **kwargs):
            self.dataset = {
                "test": Dataset.from_list(
                    [
                        dict(prompt=f"word{index}", choices=[" word2", " word3"], label=index % 2)
                        for index in range(4)
                    ]
                )
            }

    return SmokeTask(
        config=dict(
            task="memfast_in_memory",
            dataset_path="memfast_in_memory",
            test_split="test",
            output_type="multiple_choice",
            doc_to_text="prompt",
            doc_to_target="label",
            doc_to_choice="choices",
            metric_list=[dict(metric="acc", aggregation="mean", higher_is_better=True)],
            num_fewshot=0,
        )
    )


def harness_evaluate(model, tokenizer, args, *, smoke=False):
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as error:
        raise ImportError("--tasks requires the optional lm-eval[hf] dependencies") from error
    adapter = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    if smoke:
        from lm_eval.evaluator import evaluate

        task = in_memory_task()
        task.set_fewshot_seed(args.seed)
        results = evaluate(
            lm=adapter,
            task_dict={"memfast_in_memory": task},
            limit=min(args.limit or 4, 4),
            bootstrap_iters=0,
            log_samples=False,
        )
    else:
        results = simple_evaluate(
            model=adapter,
            tasks=[task.strip() for task in args.tasks.split(",") if task.strip()],
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
            log_samples=False,
        )
    return dict(
        results=results["results"],
        versions=results.get("versions", {}),
        n_samples=results.get("n-samples", {}),
        num_fewshot=0 if smoke else args.num_fewshot,
    )


def main(argv=None):
    command = parser(__doc__)
    command.add_argument(
        "--family", choices=("llama", "qwen2", "qwen3", "qwen3_5_text"), default="llama"
    )
    command.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    command.add_argument("--tasks", help="comma-separated lm-evaluation-harness task names")
    command.add_argument("--num-fewshot", type=int, default=0)
    command.add_argument("--text-file", type=Path, help="UTF-8 corpus for local perplexity")
    command.add_argument("--max-length", type=positive, default=256)
    command.add_argument("--stride", type=positive, default=128)
    command.add_argument("--prompt")
    command.add_argument("--max-new-tokens", type=positive, default=32)
    command.add_argument("--analog-head", action="store_true")
    args = command.parse_args(argv)
    if args.stage != "inference":
        command.error("this example implements --stage inference")
    setup(args)
    if not 1 <= args.stride < args.max_length or args.num_fewshot < 0:
        command.error("require 0 < stride < max-length and nonnegative num-fewshot")
    if args.tasks is not None and not any(item.strip() for item in args.tasks.split(",")):
        command.error("--tasks requires at least one task name")
    dtype = DTYPES[args.dtype]
    if args.smoke:
        args.max_length, args.stride, args.max_new_tokens = 16, 8, 2
        model, tokenizer = smoke_model(args.family, args.device, dtype)
        text = " ".join(f"word{index % 28}" for index in range(33))
    else:
        if args.checkpoint is None:
            command.error("a local Hugging Face --checkpoint directory is required")
        args.checkpoint = local_checkpoint(args.checkpoint)
        model, tokenizer = load_causal_lm(args.checkpoint, device=args.device, dtype=dtype)
        text = None
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    if not args.smoke and text is None and not args.tasks and args.prompt is None:
        command.error("choose --tasks, --text-file, or --prompt")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            command.error("tokenizer needs a pad token or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    engine = None
    args.include = text_conversion_include(model, args.include, args.analog_head)
    excluded = tuple(dict.fromkeys([*args.exclude, *(() if args.analog_head else ("lm_head",))]))
    if not args.digital:
        engine = simulator(args)
        model = convert_model(model, engine, include=args.include, exclude=excluded, inplace=True)
        if not any(isinstance(module, AnalogMatrix) for module in model.modules()):
            command.error("CIM conversion selected no matrices; check --include/--exclude")
    result = dict(
        task="causal_lm",
        stage=args.stage,
        model_type=model.config.model_type,
        pretrained=not args.smoke,
        untrained=bool(args.smoke),
        dtype=args.dtype,
        conversion=dict(include=args.include, exclude=list(excluded)),
        analog_head=any(isinstance(module, AnalogMatrix) for module in model.lm_head.modules()),
    )
    if text is not None:
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        if args.limit is not None and not args.smoke:
            tokens = tokens[:, : args.limit]
        result["text_evaluation"] = perplexity(
            model, tokens, max_length=args.max_length, stride=args.stride, device=args.device
        )
        result["text_evaluation"].update(
            corpus_tokens=tokens.shape[1],
            max_length=args.max_length,
            stride=args.stride,
            corpus_token_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
            text_file=str(args.text_file) if args.text_file else None,
        )
    if args.smoke or args.tasks:
        result["harness"] = harness_evaluate(model, tokenizer, args, smoke=args.smoke)
    prompt = args.prompt if args.prompt is not None else ("word1 word2" if args.smoke else None)
    if prompt is not None:
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        suffix = generated[0, inputs.input_ids.shape[1] :]
        result["generation"] = dict(
            new_tokens=len(suffix), text=tokenizer.decode(suffix, skip_special_tokens=True)
        )
    report(args, result, engine)
    return result


if __name__ == "__main__":
    main()
