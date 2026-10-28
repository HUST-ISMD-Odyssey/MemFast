"""Run VGG hardware-aware training on CIFAR with analog forward and FP32 backward."""

from pathlib import Path

from memfast import convert_model
from memfast.models import cifar_model

if __package__:
    from .common import (
        classification_epoch,
        load_checkpoint,
        loader,
        optimizer_for,
        report,
        save_checkpoint,
        simulator,
    )
    from .common.vision import configure_vision, vision_data, vision_parser
else:
    from common import (
        classification_epoch,
        load_checkpoint,
        loader,
        optimizer_for,
        report,
        save_checkpoint,
        simulator,
    )
    from common.vision import configure_vision, vision_data, vision_parser


def main():
    command = vision_parser(
        __doc__, task="cifar", family="vgg", default_model="vgg11_bn", training=True
    )
    command.set_defaults(stage="hat", epochs=1)
    args = command.parse_args()
    if args.stage != "hat" or args.digital:
        command.error("this example runs HAT; use 05_vgg_cifar.py for inference")
    configure_vision(command, args, inference_only=False)
    if args.save_checkpoint is None and not args.smoke:
        args.save_checkpoint = Path("examples/output") / (
            f"{args.model}_cifar{args.num_classes}_hat.pt"
        )
    model = cifar_model(args.model, num_classes=args.num_classes).float()
    load_checkpoint(model, args.checkpoint)
    state_keys = tuple(model.state_dict())
    engine = simulator(args)
    model = convert_model(
        model,
        engine,
        include=args.include,
        exclude=args.exclude,
        training=True,
        gradient_mode=args.gradient_mode,
        inplace=True,
    )
    training_data, evaluation_data = vision_data(args, model)
    training_batches = loader(training_data, args, training=True)
    optimizer = optimizer_for(model, args)
    history = []
    for _ in range(1 if args.smoke else args.epochs):
        history.append(
            classification_epoch(model, training_batches, args.device, optimizer=optimizer)
        )
    save_checkpoint(
        model,
        args.save_checkpoint,
        state_keys=state_keys,
        architecture=args.model,
        num_classes=args.num_classes,
        stage="hat",
        classes=getattr(training_data, "classes", None),
        seed=args.seed,
        epochs=1 if args.smoke else args.epochs,
        backward_dtype="float32",
    )
    evaluation = classification_epoch(model, loader(evaluation_data, args), args.device)
    report(
        args,
        dict(task="cifar", model=args.model, stage="hat", training=history, evaluation=evaluation),
        engine,
    )


if __name__ == "__main__":
    main()
