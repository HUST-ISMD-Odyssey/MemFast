"""Evaluate or fine-tune VGG and ResNet on CIFAR-10 or CIFAR-100."""

if __package__:
    from ._classification import main
else:
    from _classification import main

if __name__ == "__main__":
    main("cifar")
