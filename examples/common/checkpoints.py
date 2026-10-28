"""Strict PyTorch state loading and portable FP32 training checkpoints."""

from pathlib import Path

import torch


def load_checkpoint(model, path):
    """Load a raw state_dict or a training checkpoint before CIM conversion."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get(
        "state_dict", payload.get("model_state_dict", payload.get("model", payload))
    )
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return {key: value for key, value in payload.items() if key != "state_dict"}


def save_checkpoint(model, path, *, state_keys=None, **metadata):
    """Save digital keys after training or HAT, without programmed CIM buffers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    if state_keys is not None:
        state = {key: state[key] for key in state_keys}
    state = {key: value.detach().cpu() for key, value in state.items()}
    torch.save(dict(state_dict=state, **metadata), path)
    print(f"Saved checkpoint: {path}")
