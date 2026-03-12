#!/usr/bin/env python3
"""Convert MioCodec safetensors weights to a CoreML .mlpackage.

This script loads the MioCodec decoder from safetensors, traces it through
PyTorch, and converts the traced model to CoreML format using coremltools.
The resulting .mlpackage can be compiled to .mlmodelc and loaded by the
any-miotts CoreML backend for inference on Apple Neural Engine (ANE).

Usage:
    python convert_miocodec.py \
        --model-dir /path/to/miocodec/ \
        --output miocodec.mlpackage \
        --seq-len 512

Requirements:
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file


class MioCodecDecoderWrapper(nn.Module):
    """Wrapper that presents MioCodec decoder with the expected I/O signature.

    This wrapper takes codec tokens (int32) and a speaker embedding (float32)
    and produces a waveform (float32).  The internal architecture mirrors the
    MioCodec decoder from candle-miotts but in PyTorch for tracing.

    NOTE: The actual model architecture must match whatever was trained.
    This is a *template* -- users must adapt the architecture to match
    their specific MioCodec checkpoint.
    """

    def __init__(self, config: dict, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self.config = config

        # Build decoder layers from config.
        # This is a simplified placeholder architecture.  Real MioCodec
        # architecture should be imported from the training codebase.
        embed_dim = config.get("embed_dim", 512)
        num_codebooks = config.get("num_codebooks", 8)
        vocab_size = config.get("vocab_size", 1024)
        speaker_dim = config.get("speaker_dim", 128)
        upsample_ratios = config.get("upsample_ratios", [8, 5, 4, 2])

        # Codec token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Codebook projection (num_codebooks embeddings summed)
        self.codebook_proj = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for _ in range(num_codebooks)
        ])

        # Speaker conditioning projection
        self.speaker_proj = nn.Linear(speaker_dim, embed_dim)

        # Upsampling decoder (transposed convolutions)
        channels = embed_dim
        decoder_layers = []
        for ratio in upsample_ratios:
            out_channels = channels // 2 if channels > 32 else channels
            decoder_layers.extend([
                nn.ConvTranspose1d(channels, out_channels, kernel_size=ratio * 2, stride=ratio,
                                   padding=ratio // 2),
                nn.GELU(),
            ])
            channels = out_channels
        decoder_layers.append(nn.Conv1d(channels, 1, kernel_size=7, padding=3))
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

        # Load weights (best-effort: skip missing keys)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: {len(missing)} missing keys (may need architecture adjustment)")
            for k in missing[:10]:
                print(f"  - {k}")
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected keys in checkpoint")
            for k in unexpected[:10]:
                print(f"  - {k}")

    def forward(
        self,
        codec_tokens: torch.Tensor,   # [1, seq_len] int32
        speaker_embedding: torch.Tensor,  # [1, 128] float32
    ) -> torch.Tensor:
        """Decode codec tokens + speaker embedding into waveform.

        Returns:
            Waveform tensor of shape [1, num_samples].
        """
        # Sum codebook embeddings (each codebook slice of the token sequence)
        # For simplicity, treat all tokens as from codebook 0.
        # Real implementation should de-interleave tokens across codebooks.
        num_codebooks = len(self.codebook_proj)
        seq_len = codec_tokens.shape[1]
        tokens_per_cb = seq_len // num_codebooks

        embedded = torch.zeros(1, tokens_per_cb, self.codebook_proj[0].embedding_dim,
                               device=codec_tokens.device)
        for i, proj in enumerate(self.codebook_proj):
            start = i * tokens_per_cb
            end = start + tokens_per_cb
            cb_tokens = codec_tokens[:, start:end]
            embedded = embedded + proj(cb_tokens)

        # Add speaker conditioning
        spk = self.speaker_proj(speaker_embedding)  # [1, embed_dim]
        embedded = embedded + spk.unsqueeze(1)       # broadcast over seq

        # Transpose for conv: [batch, embed_dim, seq_len]
        x = embedded.transpose(1, 2)

        # Decode to waveform
        waveform = self.decoder(x)  # [1, 1, num_samples]
        return waveform.squeeze(1)   # [1, num_samples]


def load_config(model_dir: Path) -> dict:
    """Load MioCodec config from the model directory."""
    # Try JSON first, then YAML
    json_path = model_dir / "config.json"
    yaml_path = model_dir / "config.yaml"

    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    elif yaml_path.exists():
        try:
            import yaml
            with open(yaml_path) as f:
                return yaml.safe_load(f)
        except ImportError:
            print("Warning: PyYAML not installed, cannot read config.yaml")
            return {}
    else:
        print("Warning: No config file found, using defaults")
        return {}


def convert(
    model_dir: Path,
    output_path: Path,
    seq_len: int = 512,
    compute_precision: str = "float16",
) -> None:
    """Convert MioCodec to CoreML .mlpackage."""
    print(f"Loading MioCodec from {model_dir}")

    # Load config
    config = load_config(model_dir)
    print(f"Config: {config}")

    # Load safetensors weights
    safetensors_files = list(model_dir.glob("*.safetensors"))
    if not safetensors_files:
        print(f"Error: No .safetensors files found in {model_dir}", file=sys.stderr)
        sys.exit(1)

    state_dict = {}
    for sf in safetensors_files:
        print(f"  Loading {sf.name}")
        state_dict.update(load_file(str(sf)))

    print(f"Loaded {len(state_dict)} tensors")

    # Build model
    model = MioCodecDecoderWrapper(config, state_dict)
    model.eval()

    # Create example inputs for tracing
    num_codebooks = config.get("num_codebooks", 8)
    # Total tokens = seq_len * num_codebooks (interleaved)
    total_tokens = seq_len * num_codebooks

    example_tokens = torch.zeros(1, total_tokens, dtype=torch.int32)
    example_spk = torch.randn(1, 128, dtype=torch.float32)

    print(f"Tracing model with seq_len={seq_len} ({total_tokens} total tokens)...")

    with torch.no_grad():
        traced = torch.jit.trace(model, (example_tokens, example_spk))

    print("Converting to CoreML...")

    # Define input types
    precision = ct.precision.FLOAT16 if compute_precision == "float16" else ct.precision.FLOAT32

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="codec_tokens",
                shape=(1, total_tokens),
                dtype=np.int32,
            ),
            ct.TensorType(
                name="speaker_embedding",
                shape=(1, 128),
                dtype=np.float32,
            ),
        ],
        outputs=[
            ct.TensorType(name="waveform"),
        ],
        compute_precision=precision,
        minimum_deployment_target=ct.target.iOS16,
    )

    # Save
    print(f"Saving to {output_path}")
    mlmodel.save(str(output_path))
    print("Done!")

    # Print model info
    spec = mlmodel.get_spec()
    print(f"\nModel spec:")
    print(f"  Inputs: {[inp.name for inp in spec.description.input]}")
    print(f"  Outputs: {[out.name for out in spec.description.output]}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MioCodec safetensors to CoreML .mlpackage"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing MioCodec safetensors and config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("miocodec.mlpackage"),
        help="Output .mlpackage path (default: miocodec.mlpackage)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length per codebook (default: 512)",
    )
    parser.add_argument(
        "--precision",
        choices=["float16", "float32"],
        default="float16",
        help="Compute precision (default: float16 for ANE)",
    )

    args = parser.parse_args()

    if not args.model_dir.is_dir():
        print(f"Error: {args.model_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    convert(
        model_dir=args.model_dir,
        output_path=args.output,
        seq_len=args.seq_len,
        compute_precision=args.precision,
    )


if __name__ == "__main__":
    main()
