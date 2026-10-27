"""Program matrices with configurable slices, INT/BFP, geometry, and scale groups."""

import torch

if __package__:
    from .common import parser, positive, report, setup, simulator
else:
    from common import parser, positive, report, setup, simulator


def main():
    command = parser(__doc__)
    command.add_argument("--vectors", type=positive, default=8)
    command.add_argument("--in-features", type=positive, default=128)
    command.add_argument("--out-features", type=positive, default=80)
    args = command.parse_args()
    setup(args)
    if args.smoke:
        args.vectors = min(args.vectors, 2)
        args.in_features = min(args.in_features, 12)
        args.out_features = min(args.out_features, 8)
    for name, bits in (("input_slice", args.activation_bits), ("weight_slice", args.weight_bits)):
        widths = getattr(args, name)
        signed = name == "input_slice" or args.weight_signed
        if widths is not None and ((signed and widths[0] != 1) or sum(widths) != bits):
            command.error(f"--{name.replace('_', '-')} must start with 1 and sum to {bits}")
    weight = torch.randn(args.out_features, args.in_features, device=args.device)
    x = torch.randn(args.vectors, args.in_features, device=args.device)
    engine = None if args.digital else simulator(args)
    if engine is not None and not engine.config.weight_signed:
        weight = weight.abs()
    expected = x @ weight.T
    if engine is None:
        result = expected
        replay_equal = None
    else:
        array = engine.map(weight, name="example")
        result = array(x)
        array.reset_read()
        replay_equal = torch.equal(result, array(x))
        if not replay_equal:
            raise AssertionError("reset_read did not replay the matrix output")
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite matrix output")
    error = (result - expected).norm() / expected.norm().clamp_min(1e-12)
    report(
        args,
        dict(shape=list(result.shape), relative_l2=float(error), read_replay_equal=replay_equal),
        engine,
    )


if __name__ == "__main__":
    main()
