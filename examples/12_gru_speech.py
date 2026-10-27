"""Speech Commands classification using the legacy checkpoint's audio frontend."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from memfast import convert_model
from memfast.models.speech import AudioGRU, SpeechFeatures

if __package__:
    from .common import (
        classification_epoch,
        load_checkpoint,
        loader,
        parser,
        report,
        setup,
        simulator,
        synthetic,
    )
else:
    from common import (
        classification_epoch,
        load_checkpoint,
        loader,
        parser,
        report,
        setup,
        simulator,
        synthetic,
    )


class SpeechCommands(Dataset):
    def __init__(self, root, checkpoint, *, split="testing", download=False):
        root = Path(root)
        if not (root / "testing_list.txt").is_file():
            nested = root / "SpeechCommands" / "speech_commands_v0.02"
            if not (nested / "testing_list.txt").is_file() and download:
                import torchaudio

                root.mkdir(parents=True, exist_ok=True)
                torchaudio.datasets.SPEECHCOMMANDS(str(root), download=True, subset="testing")
            root = nested
        self.root = root
        self.classes = list(checkpoint["classes"])
        self.features = SpeechFeatures(checkpoint)
        if len(self.classes) != len(set(self.classes)):
            raise ValueError("checkpoint contains duplicate speech classes")
        testing = (root / "testing_list.txt").read_text(encoding="utf-8").splitlines()
        validation = (root / "validation_list.txt").read_text(encoding="utf-8").splitlines()
        self.ids = testing if split == "testing" else validation
        self.ids = [item for item in self.ids if item.split("/")[0] in self.classes]
        if not self.ids:
            raise ValueError("selected speech split is empty")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        return self.features(self.root / name), self.classes.index(name.split("/")[0])


def main():
    command = parser(__doc__)
    command.add_argument("--wav", type=Path, help="classify one mono 16 kHz WAV file")
    command.add_argument("--split", choices=("testing", "validation"), default="testing")
    args = command.parse_args()
    setup(args)
    if args.stage != "inference":
        command.error("speech example supports inference only")
    if not args.smoke and args.checkpoint is None:
        command.error("speech requires a checkpoint with classes and feature normalization")
    if args.checkpoint:
        metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        for key in ("classes", "features", "mean", "std"):
            if key not in metadata:
                command.error(f"speech checkpoint is missing {key}")
        hidden_size = metadata.get("hidden_size", 128)
        num_layers = metadata.get("num_layers", 2)
    else:
        metadata = dict(
            classes=[f"class{index}" for index in range(35)],
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
        hidden_size, num_layers = 128, 2
    model = AudioGRU(
        len(metadata["classes"]), hidden_size=hidden_size, num_layers=num_layers
    ).float()
    load_checkpoint(model, args.checkpoint)
    model.to(args.device)
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
    if args.wav:
        inputs = SpeechFeatures(metadata)(args.wav)[None].to(args.device)
        model.eval()
        with torch.inference_mode():
            logits = model(inputs).float().cpu()[0]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite speech output")
        report(
            args,
            dict(
                task="speech",
                model="gru",
                stage=args.stage,
                prediction=metadata["classes"][int(logits.argmax())],
                logits=logits.tolist(),
            ),
            engine,
        )
        return
    if args.smoke:
        evaluation_data = synthetic((8, 40), len(metadata["classes"]), args.limit, args.seed)
    else:
        evaluation_data = SpeechCommands(
            args.data_root, metadata, split=args.split, download=args.download
        )
    metrics = classification_epoch(model, loader(evaluation_data, args), args.device)
    report(
        args,
        dict(
            task="speech",
            model="gru",
            stage=args.stage,
            classes=metadata["classes"],
            evaluation=metrics,
        ),
        engine,
    )


if __name__ == "__main__":
    main()
