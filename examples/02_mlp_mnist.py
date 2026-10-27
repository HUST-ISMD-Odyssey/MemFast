"""Build a custom three-layer MLP, train digitally, then infer or run HAT."""

import torch
from torch import nn

from memfast import convert_model

try:
    from .common import (
        classification_epoch,
        load_checkpoint,
        loader,
        optimizer_for,
        parser,
        positive,
        report,
        save_checkpoint,
        setup,
        simulator,
        training_options,
    )
    from .common.mnist import mnist_dataset
except ImportError:
    from common import (
        classification_epoch,
        load_checkpoint,
        loader,
        optimizer_for,
        parser,
        positive,
        report,
        save_checkpoint,
        setup,
        simulator,
        training_options,
    )
    from common.mnist import mnist_dataset


class ThreeLayerMLP(nn.Module):
    """Change dimensions here, or pass --hidden-sizes, to use your own network."""

    def __init__(self, hidden_sizes=(512, 128), input_features=784, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_features, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(self.flatten(x)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main():
    command = parser(__doc__)
    training_options(command)
    command.add_argument("--hidden-sizes", type=positive, nargs=2, default=[512, 128])
    args = command.parse_args()
    setup(args)
    model = ThreeLayerMLP(args.hidden_sizes)
    load_checkpoint(model, args.checkpoint)
    if args.stage != "train" and not args.smoke and args.checkpoint is None:
        command.error("inference/HAT requires --checkpoint from a digital training run")
    state_keys = tuple(model.state_dict())
    model = model.float().to(args.device)

    # By default all eligible layers are converted. For only fc1 and fc2:
    # args.include = ["fc1", "fc2"]   # fc3 remains a software Linear layer.
    engine = None
    if args.stage != "train" and not args.digital:
        engine = simulator(args)
        model = convert_model(
            model,
            engine,
            include=args.include,
            exclude=args.exclude,
            training=args.stage == "hat",
            gradient_mode=args.gradient_mode,
            inplace=True,
        )

    history = []
    if args.stage in ("train", "hat"):
        optimizer = optimizer_for(model, args)
        batches = loader(mnist_dataset(args, training=True), args, training=True)
        for _ in range(1 if args.smoke else args.epochs):
            history.append(classification_epoch(model, batches, args.device, optimizer=optimizer))
        save_checkpoint(
            model,
            args.save_checkpoint,
            state_keys=state_keys,
            architecture="three_layer_mlp",
            hidden_sizes=args.hidden_sizes,
            stage=args.stage,
            backward_dtype="float32",
        )
    evaluation = classification_epoch(model, loader(mnist_dataset(args), args), args.device)
    # Digital training is evaluated digitally; conversion is a separate inference run.
    args.digital = args.stage == "train" or args.digital
    report(
        args,
        dict(model="three_layer_mlp", stage=args.stage, training=history, evaluation=evaluation),
        engine,
    )


if __name__ == "__main__":
    main()
