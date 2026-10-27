"""VGG ImageNet inference with upstream weights or a local checkpoint."""

if __package__:
    from .common import classification_epoch, load_checkpoint, loader, report, simulator
    from .common.vision import configure_vision, vision_data, vision_parser
else:
    from common import classification_epoch, load_checkpoint, loader, report, simulator
    from common.vision import configure_vision, vision_data, vision_parser

from memfast import convert_model
from memfast.models import imagenet_model


def main():
    command = vision_parser(__doc__, task="imagenet", family="vgg", default_model="vgg11_bn")
    args = command.parse_args()
    configure_vision(command, args)
    model = imagenet_model(args.model, weights=args.weights, num_classes=args.num_classes).float()
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
        dict(
            task="imagenet",
            model=args.model,
            stage=args.stage,
            weights=args.weights,
            evaluation=metrics,
        ),
        engine,
    )


if __name__ == "__main__":
    main()
