"""Local pretrained BERT question answering on SQuAD v1 JSON or HF parquet."""

import json
import re
import string
from collections import Counter
from pathlib import Path

import torch

from memfast import convert_model

if __package__:
    from .common import (
        load_checkpoint,
        parser,
        positive,
        report,
        setup,
        simulator,
    )
else:
    from common import (
        load_checkpoint,
        parser,
        positive,
        report,
        setup,
        simulator,
    )


def checkpoint_directory(path):
    """Accept an exported HF directory or a local HF cache model directory."""
    path = Path(path)
    if (path / "config.json").is_file():
        return path
    reference = path / "refs" / "main"
    if reference.is_file():
        candidate = path / "snapshots" / reference.read_text(encoding="utf-8").strip()
        if (candidate / "config.json").is_file():
            return candidate
    candidates = sorted(path.glob("snapshots/*/config.json"))
    if len(candidates) == 1:
        return candidates[0].parent
    raise ValueError("checkpoint must identify one local Hugging Face model snapshot")


def smoke_bert():
    """Offline BERT-Base dimensions and a fast tokenizer with real pair offsets."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.normalizers import BertNormalizer
    from tokenizers.pre_tokenizers import BertPreTokenizer
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertForQuestionAnswering, PreTrainedTokenizerFast

    words = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "what",
        "color",
        "is",
        "the",
        "sky",
        "blue",
        "grass",
        "green",
        ".",
        "?",
    ]
    backend = Tokenizer(
        WordPiece({word: index for index, word in enumerate(words)}, unk_token="[UNK]")
    )
    backend.normalizer = BertNormalizer(lowercase=True)
    backend.pre_tokenizer = BertPreTokenizer()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_input_names=["input_ids", "attention_mask", "token_type_ids"],
    )
    configuration = BertConfig(hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    configuration._attn_implementation = "eager"
    model = BertForQuestionAnswering(configuration)
    context = "the sky is blue . " * 20 + "the grass is green ."
    records = [
        dict(
            id="sky",
            question="what color is the sky ?",
            context=context,
            answers=dict(text=["blue"], answer_start=[context.index("blue")]),
        )
    ]
    return model, tokenizer, records


def squad_records(path, *, limit=None):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("SQuAD parquet input requires the datasets package") from exc
        source = load_dataset("parquet", data_files=str(path), split="train")
        rows = list(source.select(range(min(limit or len(source), len(source)))))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for article in payload["data"]:
            for paragraph in article["paragraphs"]:
                for question in paragraph["qas"]:
                    if question.get("is_impossible", False):
                        raise ValueError("this example evaluates answerable SQuAD v1 questions")
                    answers = question["answers"]
                    rows.append(
                        dict(
                            id=str(question["id"]),
                            question=question["question"],
                            context=paragraph["context"],
                            answers=dict(
                                text=[item["text"] for item in answers],
                                answer_start=[item["answer_start"] for item in answers],
                            ),
                        )
                    )
        rows = rows[:limit] if limit else rows
    if not rows or len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("SQuAD examples must be nonempty and have unique identifiers")
    result = []
    for row in rows:
        texts = list(row["answers"]["text"])
        starts = [int(value) for value in row["answers"]["answer_start"]]
        if not texts or len(texts) != len(starts):
            raise ValueError("each SQuAD v1 example needs answer text and character offsets")
        for text, start in zip(texts, starts):
            if not text or start < 0 or row["context"][start : start + len(text)] != text:
                raise ValueError("SQuAD answer text does not match its context offset")
        result.append(
            dict(
                id=str(row["id"]),
                question=row["question"],
                context=row["context"],
                answers=dict(text=texts, answer_start=starts),
            )
        )
    return result


def squad_file(args):
    if args.data_file:
        return Path(args.data_file)
    root = Path(args.data_root)
    if root.is_file():
        return root
    for name in ("validation-00000-of-00001.parquet", "dev-v1.1.json"):
        if (root / name).is_file():
            return root / name
    raise FileNotFoundError("provide a local SQuAD --data-file or --data-root")


def qa_features(records, tokenizer, *, max_length, stride):
    """Keep context offsets and example identity for every overflow window."""
    encoded = tokenizer(
        [record["question"].strip() for record in records],
        [record["context"] for record in records],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    result = []
    for index, example_index in enumerate(encoded["overflow_to_sample_mapping"]):
        sequence_ids = encoded.sequence_ids(index)
        offsets = [
            tuple(pair) if sequence_ids[position] == 1 and pair[1] > pair[0] else None
            for position, pair in enumerate(encoded["offset_mapping"][index])
        ]
        inputs = {
            key: torch.tensor(encoded[key][index], dtype=torch.long)
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in encoded
        }
        result.append(dict(inputs=inputs, offsets=offsets, example_index=int(example_index)))
    return result


def qa_batch(features, indexes, device):
    return {
        key: torch.stack([features[index]["inputs"][key] for index in indexes]).to(device)
        for key in features[indexes[0]]["inputs"]
    }


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(prediction, answers):
    prediction = normalize_answer(prediction)
    exact, f1 = 0.0, 0.0
    for answer in answers:
        answer = normalize_answer(answer)
        exact = max(exact, float(prediction == answer))
        predicted_tokens, answer_tokens = prediction.split(), answer.split()
        if not predicted_tokens or not answer_tokens:
            value = float(predicted_tokens == answer_tokens)
        else:
            common = sum((Counter(predicted_tokens) & Counter(answer_tokens)).values())
            value = 2 * common / (len(predicted_tokens) + len(answer_tokens))
        f1 = max(f1, value)
    return exact, f1


@torch.inference_mode()
def evaluate_qa(model, records, features, args):
    model.eval()
    predictions = [""] * len(records)
    best_scores = [float("-inf")] * len(records)
    for begin in range(0, len(features), args.batch_size):
        indexes = list(range(begin, min(begin + args.batch_size, len(features))))
        output = model(**qa_batch(features, indexes, args.device))
        starts, ends = output.start_logits.float().cpu(), output.end_logits.float().cpu()
        if not torch.isfinite(starts).all() or not torch.isfinite(ends).all():
            raise FloatingPointError("nonfinite answer-span logits")
        for batch_index, feature_index in enumerate(indexes):
            feature = features[feature_index]
            offsets = feature["offsets"]
            valid = torch.tensor([offset is not None for offset in offsets])
            start_logits = starts[batch_index].masked_fill(~valid, float("-inf"))
            end_logits = ends[batch_index].masked_fill(~valid, float("-inf"))
            count = min(args.n_best, int(valid.sum()))
            if not count:
                continue
            for start in start_logits.topk(count).indices.tolist():
                for end in end_logits.topk(count).indices.tolist():
                    if end < start or end - start + 1 > args.max_answer_length:
                        continue
                    example_index = feature["example_index"]
                    score = float(start_logits[start] + end_logits[end])
                    if score > best_scores[example_index]:
                        best_scores[example_index] = score
                        predictions[example_index] = records[example_index]["context"][
                            offsets[start][0] : offsets[end][1]
                        ]
    exact, f1 = 0.0, 0.0
    for record, prediction in zip(records, predictions):
        match, overlap = answer_scores(prediction, record["answers"]["text"])
        exact += match
        f1 += overlap
    return (
        dict(
            examples=len(records),
            features=len(features),
            exact_match=100 * exact / len(records),
            f1=100 * f1 / len(records),
        ),
        {record["id"]: prediction for record, prediction in zip(records, predictions)},
    )


def main():
    command = parser(__doc__)
    command.set_defaults(batch_size=8)
    command.add_argument("--data-file", type=Path)
    command.add_argument("--state-checkpoint", type=Path, help="optional saved FP32 master weights")
    command.add_argument("--max-length", type=positive, default=384)
    command.add_argument("--doc-stride", type=int, default=128)
    command.add_argument("--max-answer-length", type=positive, default=30)
    command.add_argument("--n-best", type=positive, default=20)
    command.add_argument("--predictions", type=Path)
    args = command.parse_args()
    setup(args)
    if args.stage != "inference":
        command.error("BERT SQuAD example supports inference only")
    if not 0 <= args.doc_stride < args.max_length:
        command.error("require 0 <= doc-stride < max-length")
    if args.smoke:
        model, tokenizer, records = smoke_bert()
        args.max_length = min(args.max_length, 64)
        args.doc_stride = min(args.doc_stride, 16, args.max_length // 4)
    else:
        if args.checkpoint is None:
            command.error("BERT QA requires a local pretrained --checkpoint directory")
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        root = checkpoint_directory(args.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(
            root, local_files_only=True, trust_remote_code=False, use_fast=True
        )
        model, loading = AutoModelForQuestionAnswering.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        if model.config.model_type != "bert":
            command.error("this example requires a BERT question-answering checkpoint")
        missing = set(loading["missing_keys"]) & set(dict(model.named_parameters()))
        if missing:
            command.error(f"checkpoint is missing pretrained parameters: {sorted(missing)}")
        records = squad_records(squad_file(args), limit=args.limit)
    if not tokenizer.is_fast:
        command.error("answer reconstruction requires a fast tokenizer with offsets")
    if args.max_length > model.config.max_position_embeddings:
        command.error("max-length exceeds the checkpoint's position embedding capacity")
    load_checkpoint(model, args.state_checkpoint)
    model.float().to(args.device)
    evaluation_features = qa_features(
        records, tokenizer, max_length=args.max_length, stride=args.doc_stride
    )
    engine = None
    if not args.digital:
        engine = simulator(args)
        model = convert_model(
            model,
            engine,
            include=args.include,
            exclude=args.exclude,
            training=args.stage == "hat",
            inplace=True,
        )
    metrics, predictions = evaluate_qa(model, records, evaluation_features, args)
    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        args.predictions.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    report(
        args,
        dict(
            task="squad_v1",
            model="bert",
            stage=args.stage,
            untrained=bool(args.smoke),
            evaluation=metrics,
        ),
        engine,
    )


if __name__ == "__main__":
    main()
