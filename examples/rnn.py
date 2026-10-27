"""Train or evaluate a recurrent classifier on MNIST row sequences."""

from memfast.models import RNNClassifier

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
        training_options,
    )
    from ._mnist import mnist_dataset
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
        training_options,
    )
    from _mnist import mnist_dataset


def row_batches(batches):
    for images, targets in batches:
        yield images.squeeze(1), targets


def main():
    command = parser(__doc__)
    training_options(command)
    command.add_argument("--hidden-size", type=positive, default=128)
    command.add_argument("--num-layers", type=positive, default=2)
    command.add_argument("--train", action="store_true")
    args = command.parse_args()
    setup(args)
    if args.smoke:
        args.hidden_size, args.num_layers, args.epochs = 8, 1, 1
    if not args.smoke and not args.train and args.checkpoint is None:
        command.error("evaluation requires --checkpoint; use --train to train a new model")
    model = RNNClassifier(hidden_size=args.hidden_size, num_layers=args.num_layers)
    load_checkpoint(model, args.checkpoint)
    model.to(args.device)
    history = []
    if args.train or args.smoke:
        optimizer = optimizer_for(model, args)
        for _ in range(args.epochs):
            history.append(
                classification_epoch(
                    model,
                    row_batches(loader(mnist_dataset(args, training=True), args, training=True)),
                    args.device,
                    optimizer=optimizer,
                )
            )
        save_checkpoint(
            model,
            args.save_checkpoint,
            architecture="rnn",
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )
    model, engine = prepare_model(model, args)
    result = classification_epoch(
        model, row_batches(loader(mnist_dataset(args), args)), args.device
    )
    report(
        args,
        dict(
            task="mnist_rows",
            model="rnn",
            recurrent_execution="torch.nn.RNN" if args.digital else "memfast.nn.AnalogRNN",
            analog_modules=[] if args.digital else ["rnn", "classifier"],
            training=history,
            evaluation=result,
        ),
        engine,
    )


if __name__ == "__main__":
    main()
