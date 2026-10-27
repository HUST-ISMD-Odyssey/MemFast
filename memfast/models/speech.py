"""Speech Commands GRU and the feature frontend stored with legacy checkpoints."""

import numpy as np
import torch
from torch import nn


class AudioGRU(nn.Module):
    """The legacy 40-feature, two-layer Speech Commands classifier."""

    def __init__(self, classes=35, *, hidden_size=128, num_layers=2):
        super().__init__()
        if any(type(size) is not int or size < 1 for size in (classes, hidden_size, num_layers)):
            raise ValueError("speech model dimensions must be positive integers")
        self.gru = nn.GRU(40, hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, classes)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.classifier(hidden[-1])


class SpeechFeatures:
    """Reproduce the checkpoint's padded waveform, log-mel and normalization."""

    def __init__(self, checkpoint):
        try:
            import torchaudio
        except ImportError as exc:
            raise ImportError("speech audio preprocessing requires torchaudio") from exc
        self.config = dict(checkpoint["features"])
        config = self.config
        if config["n_mels"] != 40 or config["sample_rate"] != 16000:
            raise ValueError("speech checkpoint must use 40 mel bins and 16 kHz audio")
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config["sample_rate"],
            n_fft=config["n_fft"],
            win_length=config["win_length"],
            hop_length=config["hop_length"],
            n_mels=config["n_mels"],
            center=config["center"],
            power=config["power"],
        )
        self.mean = torch.as_tensor(checkpoint["mean"]).float()
        self.std = torch.as_tensor(checkpoint["std"]).float()
        if (
            self.mean.shape != (40,)
            or self.std.shape != (40,)
            or not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.std).all()
            or not (self.std > 0).all()
        ):
            raise ValueError("invalid checkpoint feature normalization")

    def __call__(self, path):
        try:
            import soundfile
        except ImportError as exc:
            raise ImportError("speech audio loading requires soundfile") from exc
        waveform, rate = soundfile.read(str(path), dtype="float32")
        if (
            rate != self.config["sample_rate"]
            or waveform.ndim != 1
            or len(waveform) > self.config["samples"]
            or not np.isfinite(waveform).all()
        ):
            raise ValueError("expected finite mono 16 kHz audio within the checkpoint duration")
        waveform = np.pad(waveform, (0, self.config["samples"] - len(waveform)))
        feature = self.mel(torch.from_numpy(waveform)).clamp_min(self.config["log_floor"]).log().T
        return (feature - self.mean) / self.std
