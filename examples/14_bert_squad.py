"""Pretrained BERT question answering with full SQuAD v1 EM/F1 evaluation."""

import json
import re
import string
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from memfast import Config, Nonidealities, Simulator, convert_model

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_ID = "csarron/bert-base-uncased-squad-v1"  # Or a local save_pretrained directory.
DATA_FILE = Path("data/squad/dev-v1.1.json")  # Also accepts HF validation parquet.
PREDICTIONS = None  # Optional destination JSON path.
BATCH_SIZE = 8
MAX_LENGTH = 384
DOC_STRIDE = 128
INCLUDE = None
EXCLUDE = ()
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
def evaluate_qa(
    model,
    records,
    features,
    *,
    device,
    batch_size=8,
    n_best=20,
    max_answer_length=30,
):
    model.eval()
    predictions = [""] * len(records)
    best_scores = [float("-inf")] * len(records)
    for begin in range(0, len(features), batch_size):
        indexes = list(range(begin, min(begin + batch_size, len(features))))
        output = model(**qa_batch(features, indexes, device))
        starts, ends = output.start_logits.float().cpu(), output.end_logits.float().cpu()
        if not torch.isfinite(starts).all() or not torch.isfinite(ends).all():
            raise FloatingPointError("nonfinite answer-span logits")
        for batch_index, feature_index in enumerate(indexes):
            feature = features[feature_index]
            offsets = feature["offsets"]
            valid = torch.tensor([offset is not None for offset in offsets])
            start_logits = starts[batch_index].masked_fill(~valid, float("-inf"))
            end_logits = ends[batch_index].masked_fill(~valid, float("-inf"))
            count = min(n_best, int(valid.sum()))
            if not count:
                continue
            for start in start_logits.topk(count).indices.tolist():
                for end in end_logits.topk(count).indices.tolist():
                    if end < start or end - start + 1 > max_answer_length:
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
    torch.manual_seed(135)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=False)
    model = (
        AutoModelForQuestionAnswering.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            trust_remote_code=False,
        )
        .float()
        .to(DEVICE)
    )
    records = squad_records(DATA_FILE)
    evaluation_features = qa_features(
        records,
        tokenizer,
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
    )
    simulator = Simulator(CONFIG, device=DEVICE)
    model = convert_model(model, simulator, include=INCLUDE, exclude=EXCLUDE, inplace=True)
    metrics, predictions = evaluate_qa(
        model,
        records,
        evaluation_features,
        device=DEVICE,
        batch_size=BATCH_SIZE,
    )
    if PREDICTIONS is not None:
        destination = Path(PREDICTIONS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    print(f"SQuAD v1: {metrics}")
    return metrics


if __name__ == "__main__":
    main()
