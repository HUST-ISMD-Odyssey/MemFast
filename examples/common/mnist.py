"""MNIST input preparation, shared by the MLP and LeNet tutorials."""

from torchvision import datasets, transforms

from .runtime import synthetic


def mnist_dataset(args, *, training=False):
    if args.smoke:
        return synthetic((1, 28, 28), 10, args.limit, args.seed + int(training))
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return datasets.MNIST(
        args.data_root, train=training, transform=transform, download=args.download
    )
