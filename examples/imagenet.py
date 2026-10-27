"""Evaluate VGG, ResNet, MobileNetV2 and DeiT on a local ImageFolder split."""

if __package__:
    from ._classification import main
else:
    from _classification import main

if __name__ == "__main__":
    main("imagenet")
