"""FP32 classification loops shared by digital training and HAT."""

import torch
from torch.nn import functional as F


def train(model, train_loader, optimizer, *, device, epochs=1):
    """Train for the requested epochs; HAT models supply their analog forward."""
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    history = []
    for epoch in range(epochs):
        model.train()
        loss_sum, correct, samples = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device).float(), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            # No autocast: master weights and surrogate backward use FP32.
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            optimizer.step()
            count = targets.numel()
            loss_sum += loss.detach().item() * count
            correct += (logits.argmax(1) == targets).sum().item()
            samples += count
        if samples == 0:
            raise ValueError("training loader is empty")
        row = dict(
            epoch=epoch + 1, samples=samples, loss=loss_sum / samples, top1=correct / samples
        )
        history.append(row)
        print(f"Epoch {epoch + 1}/{epochs}: loss={row['loss']:.4f}, top1={row['top1']:.2%}")
    return history


@torch.inference_mode()
def evaluate(model, test_loader, *, device):
    """Evaluate every sample yielded by the loader, including the final batch."""
    model.eval()
    loss_sum, correct, top5, samples = 0.0, 0, 0, 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite evaluation logits")
        loss_sum += F.cross_entropy(logits.float(), targets, reduction="sum").item()
        correct += (logits.argmax(1) == targets).sum().item()
        top5 += (
            (logits.topk(min(5, logits.shape[1]), dim=1).indices == targets[:, None])
            .any(1)
            .sum()
            .item()
        )
        samples += targets.numel()
    if samples == 0:
        raise ValueError("evaluation loader is empty")
    result = dict(
        samples=samples,
        correct=correct,
        top1=correct / samples,
        top5=top5 / samples,
        cross_entropy=loss_sum / samples,
    )
    print(
        f"Test: {correct}/{samples}, top1={result['top1']:.2%}, loss={result['cross_entropy']:.4f}"
    )
    return result
