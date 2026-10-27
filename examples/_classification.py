"""Shared CIFAR and ImageNet data, architecture and evaluation workflows."""

from torchvision import datasets, models, transforms

from memfast.models import CIFAR_MODELS, IMAGENET_MODELS, cifar_model, imagenet_model

if __package__:
    from ._common import (
        classification_epoch,
        load_checkpoint,
        loader,
        optimizer_for,
        parser,
        positive,
        prepare_model,
        report,
        save_checkpoint,
        setup,
        synthetic,
        training_options,
    )
else:
    from _common import (
        classification_epoch,
        load_checkpoint,
        loader,
        optimizer_for,
        parser,
        positive,
        prepare_model,
        report,
        save_checkpoint,
        setup,
        synthetic,
        training_options,
    )


def image_transform(args, model, *, training=False):
    if args.task == "cifar":
        mean = (0.4914, 0.4822, 0.4465) if args.num_classes == 10 else (0.5071, 0.4867, 0.4408)
        std = (0.2023, 0.1994, 0.2010) if args.num_classes == 10 else (0.2675, 0.2565, 0.2761)
        augment = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        return transforms.Compose(
            (augment if training else []) + [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
    if args.model.startswith("deit_"):
        from timm.data import create_transform, resolve_model_data_config

        return create_transform(**resolve_model_data_config(model), is_training=training)
    if args.weights:
        return models.get_model_weights(args.model)[args.weights].transforms()
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def dataset(args, model, *, training=False):
    if args.smoke:
        size = 32 if args.task == "cifar" else 224
        return synthetic((3, size, size), args.num_classes, args.limit, args.seed + int(training))
    transform = image_transform(args, model, training=training)
    if args.task == "cifar":
        factory = datasets.CIFAR10 if args.num_classes == 10 else datasets.CIFAR100
        return factory(args.data_root, train=training, download=args.download, transform=transform)
    images = datasets.ImageFolder(args.data_root, transform=transform)
    if len(images.classes) != args.num_classes:
        raise ValueError(
            f"ImageFolder has {len(images.classes)} classes; expected {args.num_classes}"
        )
    return images


def main(task):
    command = parser(f"{task.upper()} classification with upstream vision architectures.")
    command.add_argument(
        "--model",
        choices=CIFAR_MODELS if task == "cifar" else IMAGENET_MODELS,
        default="vgg11_bn" if task == "cifar" else "resnet18",
    )
    command.add_argument("--num-classes", type=positive, default=10 if task == "cifar" else 1000)
    if task == "cifar":
        command.add_argument("--train", action="store_true", help="train or fine-tune digitally")
        training_options(command)
    else:
        command.add_argument(
            "--weights", help="explicit torchvision weight enum, or DEFAULT for DeiT"
        )
    args = command.parse_args()
    args.task = task
    args.weights = getattr(args, "weights", None)
    setup(args)
    if task == "cifar" and args.num_classes not in (10, 100):
        command.error("CIFAR num-classes must be 10 or 100")
    if args.checkpoint and args.weights:
        command.error("select either --checkpoint or --weights")
    training = getattr(args, "train", False)
    if not args.smoke and not training and not args.checkpoint and not args.weights:
        command.error("evaluation requires --checkpoint or ImageNet --weights")
    model = (
        cifar_model(args.model, num_classes=args.num_classes)
        if task == "cifar"
        else imagenet_model(args.model, weights=args.weights, num_classes=args.num_classes)
    )
    load_checkpoint(model, args.checkpoint)
    # Resolve preprocessing from the upstream model before conversion.
    test_data = dataset(args, model)
    history = []
    if training:
        train_batches = loader(dataset(args, model, training=True), args, training=True)
        model.to(args.device)
        optimizer = optimizer_for(model, args)
        for _ in range(1 if args.smoke else args.epochs):
            history.append(
                classification_epoch(model, train_batches, args.device, optimizer=optimizer)
            )
        save_checkpoint(
            model, args.save_checkpoint, architecture=args.model, num_classes=args.num_classes
        )
    model, engine = prepare_model(model, args)
    result = classification_epoch(model, loader(test_data, args), args.device)
    report(
        args,
        dict(
            task=task,
            model=args.model,
            weights=args.weights,
            untrained=not training and args.checkpoint is None and args.weights is None,
            training=history,
            evaluation=result,
        ),
        engine,
    )
