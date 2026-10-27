"""Hardware-aware MNIST training with trainable master weights."""

if __package__:
    from ._mnist import main
else:
    from _mnist import main

if __name__ == "__main__":
    main(hat=True)
