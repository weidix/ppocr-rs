#!/usr/bin/env python3
"""Write an ONNX copy whose only graph input has a fixed NCHW shape."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def set_shape(value_info: onnx.ValueInfoProto, shape: list[int]) -> None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("elem_type"):
        raise ValueError(f"{value_info.name!r} is not a tensor input")
    dims = tensor_type.shape.dim
    del dims[:]
    for value in shape:
        dims.add().dim_value = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="dynamic ONNX model")
    parser.add_argument("output", type=Path, help="fixed-shape ONNX output")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=4,
        metavar=("N", "C", "H", "W"),
        required=True,
        help="fixed NCHW input shape",
    )
    args = parser.parse_args()
    if any(value <= 0 for value in args.shape):
        parser.error("--shape values must be positive")

    model = onnx.load_model(args.input, load_external_data=True)
    if len(model.graph.input) != 1:
        parser.error(f"expected one graph input, found {len(model.graph.input)}")
    set_shape(model.graph.input[0], args.shape)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.output)
    print(f"wrote {args.output} with input shape {args.shape}")


if __name__ == "__main__":
    main()
