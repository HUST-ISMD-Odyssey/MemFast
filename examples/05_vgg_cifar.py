"""Run VGG inference on CIFAR-10 or CIFAR-100 from a trained checkpoint."""

if __package__:
    from .common import classification_epoch, load_checkpoint, loader, report, simulator
    from .common.vision import configure_vision, vision_data, vision_parser
else:
    from common import classification_epoch, load_checkpoint, loader, report, simulator
    from common.vision import configure_vision, vision_data, vision_parser

from memfast import convert_model
from memfast.models import cifar_model


def main():
    command = vision_parser(__doc__, task="cifar", family="vgg", default_model="vgg11_bn")
    args = command.parse_args()
    configure_vision(command, args)
    model = cifar_model(args.model, num_classes=args.num_classes).float()
    load_checkpoint(model, args.checkpoint)
    _, evaluation_data = vision_data(args, model)
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
    metrics = classification_epoch(model, loader(evaluation_data, args), args.device)
    report(
        args,
        dict(task="cifar", model=args.model, stage=args.stage, evaluation=metrics),
        engine,
    )


if __name__ == "__main__":
    main()
