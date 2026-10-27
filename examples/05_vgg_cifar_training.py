"""Train a software VGG on CIFAR-10/100 and save its FP32 inference checkpoint."""

from memfast.models import cifar_model

if __package__:
    from .common import (
        classification_epoch,
        load_checkpoint,
        loader,
        optimizer_for,
        report,
        save_checkpoint,
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
    )
    from common.vision import configure_vision, vision_data, vision_parser


def main():
    command = vision_parser(
        __doc__, task="cifar", family="vgg", default_model="vgg11_bn", training=True
    )
    command.set_defaults(stage="train", digital=True)
    args = command.parse_args()
    if args.stage != "train":
        command.error("this tutorial trains digitally; use 06_vgg_cifar.py for inference")
    configure_vision(command, args, inference_only=False)
    model = cifar_model(args.model, num_classes=args.num_classes).float().to(args.device)
    load_checkpoint(model, args.checkpoint)
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
        architecture=args.model,
        num_classes=args.num_classes,
        stage="train",
        classes=getattr(training_data, "classes", None),
        backward_dtype="float32",
    )
    evaluation = classification_epoch(model, loader(evaluation_data, args), args.device)
    report(
        args,
        dict(
            task="cifar", model=args.model, stage="train", training=history, evaluation=evaluation
        ),
    )


if __name__ == "__main__":
    main()
