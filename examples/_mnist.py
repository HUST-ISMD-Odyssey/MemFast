"""MNIST inference and training, including hardware-aware distributed training."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision import datasets, transforms

from memfast.models import MLP

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


def mnist_dataset(args, *, training=False):
    if args.smoke:
        return synthetic((1, 28, 28), 10, args.limit, args.seed + int(training))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return datasets.MNIST(
        args.data_root, train=training, download=args.download, transform=transform
    )


def main(*, hat=False, distributed=False):
    command = parser(
        "MNIST MLP hardware-aware training." if hat else "Train or evaluate a MNIST MLP."
    )
    training_options(command)
    command.add_argument("--hidden-sizes", type=positive, nargs="+", default=[512, 128])
    if not hat:
        command.add_argument(
            "--train", action="store_true", help="train digitally before evaluation"
        )
    args = command.parse_args()
    setup(args)
    if args.smoke:
        args.hidden_sizes = [16]
        args.epochs = 1
    train = hat or getattr(args, "train", False)
    if not train and not args.smoke and args.checkpoint is None:
        command.error("inference requires --checkpoint; use --train to train a new model")
    rank, world_size, initialized = 0, 1, False
    if distributed:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            if args.device.startswith("cuda"):
                local_rank = int(os.environ["LOCAL_RANK"])
                args.device = f"cuda:{local_rank}"
                torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl" if args.device.startswith("cuda") else "gloo")
            initialized = True
            rank = dist.get_rank()
    try:
        # Only rank zero may download shared dataset files.
        if initialized and args.download and not args.smoke:
            if rank == 0:
                mnist_dataset(args, training=True)
                mnist_dataset(args)
            dist.barrier()
            args.download = False
        model = MLP(hidden_sizes=args.hidden_sizes)
        load_checkpoint(model, args.checkpoint)
        state_keys = tuple(model.state_dict())
        engine = None
        if hat:
            model, engine = prepare_model(model, args, training=True)
        else:
            model.to(args.device)
        bare_model = model
        if initialized:
            model = DistributedDataParallel(
                model,
                device_ids=[int(os.environ["LOCAL_RANK"])]
                if args.device.startswith("cuda")
                else None,
                broadcast_buffers=False,
            )
        history = []
        if train:
            batches = loader(
                mnist_dataset(args, training=True),
                args,
                training=True,
                rank=rank,
                world_size=world_size,
            )
            optimizer = optimizer_for(model, args)
            for epoch in range(args.epochs):
                if hasattr(batches.sampler, "set_epoch"):
                    batches.sampler.set_epoch(epoch)
                history.append(
                    classification_epoch(
                        model, batches, args.device, optimizer=optimizer, distributed=initialized
                    )
                )
            if rank == 0:
                save_checkpoint(
                    bare_model,
                    args.save_checkpoint,
                    state_keys=state_keys,
                    architecture="mlp",
                    hidden_sizes=args.hidden_sizes,
                    epochs=args.epochs,
                )
        if not hat:
            bare_model, engine = prepare_model(bare_model, args)
        test_batches = loader(mnist_dataset(args), args, rank=rank, world_size=world_size)
        metrics = classification_epoch(
            bare_model, test_batches, args.device, distributed=initialized
        )
        if rank == 0:
            report(
                args,
                dict(
                    task="mnist_hat" if hat else "mnist",
                    model="mlp",
                    untrained=not train and args.checkpoint is None,
                    world_size=world_size,
                    gradient_mode=args.gradient_mode if hat and not args.digital else None,
                    training=history,
                    evaluation=metrics,
                ),
                engine,
            )
    finally:
        if initialized:
            dist.destroy_process_group()
