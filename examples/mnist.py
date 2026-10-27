"""Train or evaluate a MNIST multilayer perceptron."""

if __package__:
    from ._mnist import main
else:
    from _mnist import main

if __name__ == "__main__":
    main()
