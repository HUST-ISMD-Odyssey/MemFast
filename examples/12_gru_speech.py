"""Train the custom Speech Commands GRU, restore it and evaluate CIM inference."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from memfast import Config, Nonidealities, Simulator, convert_model
from memfast.models import AudioGRU
from memfast.models.speech import SpeechFeatures

if __package__:
    from .common import evaluate, load_checkpoint, save_checkpoint, train
else:
    from common import evaluate, load_checkpoint, save_checkpoint, train

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("data/speech/SpeechCommands/speech_commands_v0.02")
CHECKPOINT = Path(__file__).resolve().parent / "output/gru_speech_commands.pt"
HAT_CHECKPOINT = CHECKPOINT.with_name("gru_speech_commands_hat.pt")
WAV = None  # Path to a mono 16 kHz WAV for single-file inference.
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
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


class SpeechCommands(Dataset):
    """Use the published split lists and the checkpoint's log-mel frontend."""

    def __init__(self, root, metadata, *, split="testing"):
        self.root = Path(root)
        self.classes = list(metadata["classes"])
        self.features = SpeechFeatures(metadata)
        testing = (self.root / "testing_list.txt").read_text().splitlines()
        validation = (self.root / "validation_list.txt").read_text().splitlines()
        if split == "training":
            held_out = set(testing + validation)
            ids = [
                path.relative_to(self.root).as_posix() for path in sorted(self.root.glob("*/*.wav"))
            ]
            ids = [name for name in ids if name not in held_out]
        else:
            ids = testing if split == "testing" else validation
        self.ids = [name for name in ids if name.split("/")[0] in self.classes]
        if not self.ids:
            raise ValueError(f"Speech Commands {split} split is empty")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        return self.features(self.root / name), self.classes.index(name.split("/")[0])


def main(stage="inference"):
    torch.manual_seed(135)
    if stage == "train":
        classes = sorted(
            path.name
            for path in DATA_ROOT.iterdir()
            if path.is_dir() and not path.name.startswith("_") and any(path.glob("*.wav"))
        )
        metadata = dict(
            classes=classes,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            features=dict(
                sample_rate=16000,
                n_fft=400,
                win_length=400,
                hop_length=160,
                n_mels=40,
                center=False,
                power=2,
                samples=16000,
                log_floor=1e-10,
            ),
            mean=torch.zeros(40),
            std=torch.ones(40),
        )
        # Fit normalization on training frames only, then save it with the weights.
        raw_data = SpeechCommands(DATA_ROOT, metadata, split="training")
        total, squares, frames = (
            torch.zeros(40, dtype=torch.float64),
            torch.zeros(
                40,
                dtype=torch.float64,
            ),
            0,
        )
        for features, _ in DataLoader(raw_data, batch_size=BATCH_SIZE):
            features = features.double().reshape(-1, 40)
            total += features.sum(0)
            squares += features.square().sum(0)
            frames += len(features)
        mean = total / frames
        metadata["mean"] = mean.float()
        metadata["std"] = (squares / frames - mean.square()).clamp_min(1e-12).sqrt().float()
    else:
        metadata = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = (
        AudioGRU(
            len(metadata["classes"]),
            hidden_size=metadata.get("hidden_size", HIDDEN_SIZE),
            num_layers=metadata.get("num_layers", NUM_LAYERS),
        )
        .float()
        .to(DEVICE)
    )
    if stage != "train":
        load_checkpoint(model, CHECKPOINT)
        metadata.setdefault("hidden_size", HIDDEN_SIZE)
        metadata.setdefault("num_layers", NUM_LAYERS)
    state_keys = tuple(model.state_dict())
    if stage in ("inference", "hat"):
        simulator = Simulator(CONFIG, device=DEVICE)
        model = convert_model(
            model,
            simulator,
            include=INCLUDE,
            exclude=EXCLUDE,
            training=stage == "hat",
            inplace=True,
        )
    if stage in ("train", "hat"):
        train_data = SpeechCommands(DATA_ROOT, metadata, split="training")
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train(model, train_loader, optimizer, device=DEVICE, epochs=EPOCHS)
        save_checkpoint(
            model,
            HAT_CHECKPOINT if stage == "hat" else CHECKPOINT,
            state_keys=state_keys,
            **{
                key: metadata[key]
                for key in (
                    "classes",
                    "hidden_size",
                    "num_layers",
                    "features",
                    "mean",
                    "std",
                )
            },
        )
    if WAV is not None:
        model.eval()
        inputs = SpeechFeatures(metadata)(WAV)[None].to(DEVICE)
        with torch.inference_mode():
            prediction = model(inputs).argmax(1).item()
        print(metadata["classes"][prediction])
        return prediction
    test_data = SpeechCommands(DATA_ROOT, metadata, split="testing")
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return evaluate(model, test_loader, device=DEVICE)


if __name__ == "__main__":
    command = argparse.ArgumentParser(__doc__)
    command.add_argument("--stage", choices=("train", "inference", "hat"), default="inference")
    main(command.parse_args().stage)
