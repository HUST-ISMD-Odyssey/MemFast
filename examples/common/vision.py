"""Datasets and preprocessing shared by the vision examples."""

from pathlib import Path

from torchvision import datasets, models, transforms

from .runtime import (
    parser,
    positive,
    setup,
    synthetic,
    training_options,
)


def vision_parser(description, *, task, family, default_model, training=False):
    command = parser(description)
    if training:
        training_options(command)
    choices = {
        "vgg": ("vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"),
        "resnet": ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"),
        "mobilenet": ("mobilenet_v2",),
        "deit": (
            "deit_tiny_patch16_224",
            "deit_small_patch16_224",
            "deit_base_patch16_224",
        ),
    }[family]
    if task == "imagenet" and family == "vgg":
        choices = ("vgg11_bn", "vgg16_bn")
    if task == "imagenet" and family == "resnet":
        choices = ("resnet18", "resnet50", "resnet101")
    command.add_argument("--model", choices=choices, default=default_model)
    command.add_argument("--num-classes", type=positive, default=10 if task == "cifar" else 1000)
    command.add_argument("--weights", help="explicit upstream weight enum, or DEFAULT")
    if training:
        command.add_argument("--train-root", type=Path)
    command.add_argument("--validation-root", type=Path)
    command.set_defaults(task=task)
    return command


def configure_vision(command, args, *, inference_only=True):
    setup(args)
    if inference_only and args.stage != "inference":
        command.error("this example supports inference only")
    if args.task == "cifar" and args.num_classes not in (10, 100):
        command.error("CIFAR num-classes must be 10 or 100")
    if args.task == "cifar" and args.weights:
        command.error("CIFAR requires task-specific --checkpoint weights")
    if args.weights and args.checkpoint:
        command.error("choose either --weights or --checkpoint")
    if args.stage != "train" and not args.smoke and not args.checkpoint and not args.weights:
        command.error("inference requires --checkpoint or explicit ImageNet --weights")
    args.digital = args.digital or args.stage == "train"


def image_transform(args, model, *, training=False):
    if args.task == "cifar":
        mean = (0.4914, 0.4822, 0.4465) if args.num_classes == 10 else (0.5071, 0.4867, 0.4408)
        std = (0.2023, 0.1994, 0.2010) if args.num_classes == 10 else (0.2675, 0.2565, 0.2761)
        augmentation = (
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            if training
            else []
        )
        return transforms.Compose(
            augmentation + [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
    if args.model.startswith("deit_"):
        from timm.data import create_transform, resolve_model_data_config

        return create_transform(**resolve_model_data_config(model), is_training=training)
    if args.weights:
        preset = models.get_model_weights(args.model)[args.weights].transforms()
        if not training:
            return preset
        size, mean, std = preset.crop_size[0], preset.mean, preset.std
        interpolation = preset.interpolation
    else:
        size, mean, std = 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        interpolation = transforms.InterpolationMode.BILINEAR
    spatial = (
        [
            transforms.RandomResizedCrop(size, interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
        ]
        if training
        else [transforms.Resize(256), transforms.CenterCrop(size)]
    )
    return transforms.Compose(spatial + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def vision_dataset(args, model, *, training=False):
    if args.smoke:
        size = 32 if args.task == "cifar" else 224
        return synthetic(
            (3, size, size), args.num_classes, args.limit or 4, args.seed + int(training)
        )
    transform = image_transform(args, model, training=training)
    if args.task == "cifar":
        factory = datasets.CIFAR10 if args.num_classes == 10 else datasets.CIFAR100
        return factory(args.data_root, train=training, download=args.download, transform=transform)
    explicit = getattr(args, "train_root", None) if training else args.validation_root
    root = Path(explicit or args.data_root)
    split = root / ("train" if training else "val")
    if explicit is None and split.is_dir():
        root = split
    result = datasets.ImageFolder(root, transform=transform)
    if len(result.classes) != args.num_classes:
        raise ValueError(
            f"ImageFolder has {len(result.classes)} classes; expected {args.num_classes}"
        )
    return result


def vision_data(args, model):
    evaluation = vision_dataset(args, model)
    training = vision_dataset(args, model, training=True) if args.stage != "inference" else None
    if training is not None and hasattr(training, "class_to_idx"):
        if training.class_to_idx != evaluation.class_to_idx:
            raise ValueError("training and validation class orders differ")
    return training, evaluation
