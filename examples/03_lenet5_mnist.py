"""LeNet-5: digital training, checkpoint restoration, CIM inference, and FP32 HAT."""

from pathlib import Path

from memfast import convert_model
from memfast.models import LeNet5

try:
    from .common import (
        classification_epoch,
        load_checkpoint,
        loader,
        optimizer_for,
        parser,
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
        report,
        save_checkpoint,
        setup,
        simulator,
        training_options,
    )
    from common.mnist import mnist_dataset


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints/lenet5_mnist.pt"


def main():
    command = parser(__doc__)
    training_options(command)
    command.set_defaults(batch_size=128, epochs=8)
    args = command.parse_args()
    setup(args)
    if args.stage != "train" and not args.smoke and args.checkpoint is None:
        args.checkpoint = DEFAULT_CHECKPOINT
    if args.stage in ("train", "hat") and args.save_checkpoint is None and not args.smoke:
        args.save_checkpoint = (
            Path("examples/output/lenet5_mnist.pt")
            if args.stage == "train"
            else Path("examples/output/lenet5_mnist_hat.pt")
        )
    # Architecture is defined in memfast/models/lenet.py; load its digital state first.
    model = LeNet5()
    load_checkpoint(model, args.checkpoint)
    state_keys = tuple(model.state_dict())
    model = model.float().to(args.device)
    engine = None
    if args.stage != "train" and not args.digital:
        engine = simulator(args)
        # To use only the first four eligible layers, retaining the final classifier:
        # include = ["features.0", "features.3", "classifier.0", "classifier.2"]
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
        batches = loader(mnist_dataset(args, training=True), args, training=True)
        optimizer = optimizer_for(model, args)
        for _ in range(1 if args.smoke else args.epochs):
            history.append(classification_epoch(model, batches, args.device, optimizer=optimizer))
        save_checkpoint(
            model,
            args.save_checkpoint,
            state_keys=state_keys,
            architecture="lenet5_mnist",
            normalization=dict(mean=[0.5], std=[0.5]),
            seed=args.seed,
            epochs=args.epochs,
            stage=args.stage,
            backward_dtype="float32",
        )
    evaluation = classification_epoch(model, loader(mnist_dataset(args), args), args.device)
    args.digital = args.stage == "train" or args.digital
    report(
        args,
        dict(model="lenet5_mnist", stage=args.stage, training=history, evaluation=evaluation),
        engine,
    )


if __name__ == "__main__":
    main()
