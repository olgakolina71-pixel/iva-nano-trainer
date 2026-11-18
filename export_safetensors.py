import argparse

import torch

from iva_nano.hf_compat import export_safetensors_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Export an IVA Nano .pt checkpoint to safe, weights-only Safetensors."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    output = export_safetensors_checkpoint(checkpoint, args.output)
    print(f"Safetensors exported: {output}")


if __name__ == "__main__":
    main()
