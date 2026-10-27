"""Distributed hardware-aware MNIST training; launch using torchrun."""

if __package__:
    from ._mnist import main
else:
    from _mnist import main

if __name__ == "__main__":
    main(hat=True, distributed=True)
