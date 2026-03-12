#!/usr/bin/env python3
"""Convert a MioCodec PyTorch model to TFLite format for QNN/Hexagon NPU.

Uses ai_edge_torch (Google's PyTorch -> TFLite converter) to produce a
.tflite model that can be loaded with the QNN delegate on Qualcomm devices.

Usage:
    python convert_miocodec.py \
        --checkpoint path/to/miocodec.safetensors \
        --config path/to/config.yaml \
        --output miocodec.tflite \
        [--seq-len 512]

The resulting .tflite model has:
    Input 0: codec_tokens  (int32,   [1, seq_len])
    Input 1: speaker_embedding (float32, [1, 128])
    Output 0: waveform     (float32, [1, audio_len])
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MioCodec PyTorch model to TFLite"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to MioCodec .safetensors or .pt checkpoint",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to MioCodec config.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("miocodec.tflite"),
        help="Output .tflite path (default: miocodec.tflite)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Fixed sequence length for codec tokens (default: 512)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic-range quantization for smaller model size",
    )
    return parser.parse_args()


def load_miocodec_model(
    checkpoint_path: Path, config_path: Path
) -> torch.nn.Module:
    """Load and prepare the MioCodec model for export.

    This is a placeholder that should be adapted to the actual MioCodec
    model class and weight format used in candle-miotts.
    """
    # Import the MioCodec model class.
    # Adjust this import to match the actual Python model definition.
    try:
        from candle_miotts.models.miocodec import MioCodec
    except ImportError:
        print(
            "ERROR: Could not import MioCodec model class.\n"
            "Make sure candle-miotts Python package is installed or\n"
            "adjust the import in this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = MioCodec(**config)

    # Load weights from safetensors or PyTorch checkpoint
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path))
    else:
        state_dict = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=True
        )

    model.load_state_dict(state_dict)
    model.eval()
    return model


def convert_to_tflite(
    model: torch.nn.Module,
    seq_len: int,
    quantize: bool,
) -> bytes:
    """Convert the PyTorch model to TFLite using ai_edge_torch."""
    import ai_edge_torch

    # Create sample inputs matching the model's expected shapes
    sample_tokens = torch.zeros(1, seq_len, dtype=torch.int32)
    sample_speaker = torch.randn(1, 128, dtype=torch.float32)

    sample_inputs = (sample_tokens, sample_speaker)

    print(f"Converting model with seq_len={seq_len}...")

    # Convert using ai_edge_torch
    edge_model = ai_edge_torch.convert(
        model,
        sample_inputs,
    )

    if quantize:
        print("Applying dynamic-range quantization...")
        edge_model = edge_model.quantize()

    tflite_model = edge_model.export()
    return tflite_model


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading MioCodec from {args.checkpoint}...")
    model = load_miocodec_model(args.checkpoint, args.config)

    tflite_bytes = convert_to_tflite(model, args.seq_len, args.quantize)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(tflite_bytes)
    print(f"TFLite model saved to {args.output} ({len(tflite_bytes)} bytes)")


if __name__ == "__main__":
    main()
